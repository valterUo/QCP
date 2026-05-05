import torch


def _to_2d_tensor(array, device, dtype):
    tensor = torch.as_tensor(array, device=device, dtype=dtype)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def _combine_generators_and_params_torch(params, circuit, init_coefs=None):
    generators = torch.as_tensor(
        circuit.generators, device=params.device, dtype=params.dtype
    )
    effective_params = params

    if circuit.init_generators is not None:
        if init_coefs is None:
            raise ValueError("init_coefs must be provided if init_gates are used")

        init_generators = torch.as_tensor(
            circuit.init_generators, device=params.device, dtype=params.dtype
        )
        init_coefs_tensor = torch.as_tensor(
            init_coefs, device=params.device, dtype=params.dtype
        )
        generators = torch.cat([generators, init_generators], dim=0)
        effective_params = torch.cat([effective_params, init_coefs_tensor], dim=0)

    return generators, effective_params


class ExpvalFunction:
    @staticmethod
    def apply(
        params,
        circuit,
        ops,
        n_samples=1000,
        init_coefs=None,
        z_samples=None,
        return_samples=False,
    ):
        single_op = torch.as_tensor(ops).ndim == 1
        ops_tensor = _to_2d_tensor(ops, params.device, params.dtype)
        generators, effective_params = _combine_generators_and_params_torch(
            params, circuit, init_coefs
        )

        ops_gen = torch.remainder(ops_tensor @ generators.T, 2.0)
        par_ops_gates = 2.0 * effective_params.unsqueeze(0) * ops_gen

        if circuit.bitflip:
            expvals = torch.prod(
                torch.where(
                    ops_gen > 0,
                    torch.cos(par_ops_gates),
                    torch.ones_like(par_ops_gates),
                ),
                dim=-1,
            )

            if circuit.spin_sym:
                odd_ops = 1.0 - 2.0 * torch.remainder(ops_tensor.sum(dim=-1), 2.0)
                expvals = 0.5 * expvals + 0.5 * odd_ops * expvals

            if return_samples:
                expvals = expvals.unsqueeze(-1)

        else:
            if n_samples <= 0:
                raise ValueError("n_samples must be positive")

            if z_samples is None:
                z_tensor = torch.randint(
                    0,
                    2,
                    (n_samples, circuit.n_qubits),
                    device=params.device,
                    dtype=torch.int64,
                ).to(params.dtype)
            else:
                z_tensor = _to_2d_tensor(z_samples, params.device, params.dtype)

            samples_gates = 1.0 - 2.0 * torch.remainder(z_tensor @ generators.T, 2.0)
            expvals = torch.cos(par_ops_gates @ samples_gates.T)

            if circuit.spin_sym:
                ops_sum = ops_tensor.sum(dim=-1, keepdim=True)
                samples_sum = z_tensor.sum(dim=-1).unsqueeze(0)
                expvals = (
                    2.0
                    - torch.remainder(ops_sum, 2.0)
                    - 2.0 * torch.remainder(samples_sum, 2.0)
                ) * expvals

            if not return_samples:
                expvals = expvals.mean(dim=-1)

        return expvals if not single_op else expvals[0]


def expvals_torch(
    params,
    circuit,
    ops,
    n_samples=1000,
    init_coefs=None,
    z_samples=None,
    return_samples=False,
):
    return ExpvalFunction.apply(
        params,
        circuit,
        ops,
        n_samples=n_samples,
        init_coefs=init_coefs,
        z_samples=z_samples,
        return_samples=return_samples,
    )

# Corresponds to https://github.com/XanaduAI/iqpopt/blob/bf207502c317364746bd56ae0dd1b4d04e492c7b/src/iqpopt/gen_qml/iqp_methods.py#L11
def mmd_loss_torch(params, circuit, ground_truth, ops, n_samples=1000, init_coefs=None):
    expval_samples = ExpvalFunction.apply(
        params,
        circuit,
        ops,
        n_samples=n_samples,
        init_coefs=init_coefs,
        return_samples=True,
    )
    expval_samples = _to_2d_tensor(expval_samples, params.device, params.dtype)
    expvals = expval_samples.mean(dim=-1)

    m = len(ground_truth)
    if m < 2:
        raise ValueError("ground_truth must contain at least 2 samples")
    if not circuit.bitflip and expval_samples.shape[-1] <= 1:
        raise ValueError("n_samples must be greater than 1 for IQP circuits")

    ground_truth_tensor = _to_2d_tensor(ground_truth, params.device, params.dtype)
    ops_tensor = _to_2d_tensor(ops, params.device, params.dtype)
    data_vals = 1.0 - 2.0 * torch.remainder(ground_truth_tensor @ ops_tensor.T, 2.0)
    tr_data = data_vals.mean(dim=0)

    if circuit.bitflip:
        res = expvals * expvals - 2.0 * expvals * tr_data + (m * tr_data * tr_data - 1.0) / (m - 1)
    else:
        correction = expval_samples.square().mean(dim=-1) / expval_samples.shape[-1]
        n_iqp_samples = expval_samples.shape[-1]
        res = (
            (expvals * expvals - correction) * n_iqp_samples / (n_iqp_samples - 1)
            - 2.0 * expvals * tr_data
            + (m * tr_data * tr_data - 1.0) / (m - 1)
        )

    return torch.mean(res)
