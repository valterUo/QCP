from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector
import numpy as np

# Note: init_gates and init_coeffs will not probably be used in this project,
# but are included here just in case because they were included in the original Iqpopt package.

# Note: The sampling part uses a perfect simulator (StatevectorSampler) at the moment. 

# params is a list/array of angles, one per gate.
# gates is a list of which qubits each gate acts on.

class IqpCircuitQiskit:
    def __init__(self, n_qubits, gates, init_gates=None, spin_sym=False, bitflip=False):
        self.n_qubits = n_qubits
        self.gates = gates
        self.init_gates = init_gates
        self.spin_sym = spin_sym
        self.bitflip = bitflip

        # Build generator matrix (IQPopt-style)
        self.generators = self._build_generators(self.gates)
        self.init_generators = (
            self._build_generators(self.init_gates)
            if self.init_gates is not None
            else None
        )

    
    # Build generator matrix
    def _build_generators(self, gates):
        """
        Convert gates into binary generator matrix:
        shape = (n_gates, n_qubits)
        """
        if gates is None:
            return np.zeros((0, self.n_qubits), dtype=np.int8)

        n_gates = len(gates)
        generators = np.zeros((n_gates, self.n_qubits), dtype=np.int8)

        for k, gate in enumerate(gates):
            qubits = gate[0]  # because format is [[i]] or [[i,j]]
            for q in qubits:
                generators[k, q] = 1

        return generators
    
    # Corresponds to https://github.com/XanaduAI/iqpopt/blob/bf207502c317364746bd56ae0dd1b4d04e492c7b/src/iqpopt/iqp_optimizer.py#L45
    # to https://github.com/XanaduAI/iqpopt/blob/bf207502c317364746bd56ae0dd1b4d04e492c7b/src/iqpopt/iqp_optimizer.py#L94
    def _combine_generators_and_params(self, params, init_coefs=None):
        params = np.asarray(params, dtype=np.float64)

        if self.init_generators is None:
            return self.generators, params

        if init_coefs is None:
            raise ValueError("init_coefs must be provided if init_gates are used")

        init_coefs = np.asarray(init_coefs, dtype=np.float64)
        generators = np.concatenate([self.generators, self.init_generators], axis=0)
        effective_params = np.concatenate([params, init_coefs], axis=0)
        return generators, effective_params

    # Corresponds to https://github.com/XanaduAI/iqpopt/blob/bf207502c317364746bd56ae0dd1b4d04e492c7b/src/iqpopt/iqp_optimizer.py#L279
    def _draw_z_samples(self, n_samples, z_samples=None):
        if z_samples is not None:
            z_samples = np.asarray(z_samples, dtype=np.int8)
            if z_samples.ndim == 1:
                z_samples = z_samples.reshape(1, -1)
            if z_samples.shape[1] != self.n_qubits:
                raise ValueError("z_samples must have shape (n_samples, n_qubits)")
            return z_samples

        n_samples = int(n_samples)
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")

        return np.random.randint(0, 2, size=(n_samples, self.n_qubits), dtype=np.int8)

    def _sample_bitflip(self, params, init_coefs=None, shots=1024):
        params = np.asarray(params, dtype=np.float64)
        shots = int(shots)
        samples = np.zeros((shots, self.n_qubits), dtype=np.int8)

        if self.spin_sym:
            global_flips = np.random.binomial(1, 0.5, size=shots).astype(np.int8)
            samples = (samples + global_flips[:, None]) % 2

        if self.init_gates is not None:
            if init_coefs is None:
                raise ValueError("init_coefs must be provided if init_gates are used")

            for par, gate in zip(init_coefs, self.init_gates):
                flips = np.random.binomial(1, np.sin(par) ** 2, size=shots).astype(np.int8)
                if np.any(flips):
                    samples[:, gate[0]] = (samples[:, gate[0]] + flips[:, None]) % 2

        for par, gate in zip(params, self.gates):
            flips = np.random.binomial(1, np.sin(par) ** 2, size=shots).astype(np.int8)
            if np.any(flips):
                samples[:, gate[0]] = (samples[:, gate[0]] + flips[:, None]) % 2

        return samples

    def _bitflip_op_expval(self, params, ops, init_coefs=None):
        generators, effective_params = self._combine_generators_and_params(
            params, init_coefs
        )
        overlap = ((ops @ generators.T) % 2).astype(np.int8)
        cos_terms = np.cos(2.0 * effective_params)

        expvals = np.prod(np.where(overlap, cos_terms, 1.0), axis=1)

        if self.spin_sym:
            odd_mask = (ops.sum(axis=1) & 1).astype(bool)
            expvals[odd_mask] = 0.0

        return expvals

    def _iqp_op_expval(
        self,
        params,
        ops,
        init_coefs=None,
        n_samples=1000,
        z_samples=None,
        return_samples=False,
    ):
        generators, effective_params = self._combine_generators_and_params(
            params, init_coefs
        )
        # Corresponds to https://github.com/XanaduAI/iqpopt/blob/bf207502c317364746bd56ae0dd1b4d04e492c7b/src/iqpopt/iqp_optimizer.py#L279
        z_samples = self._draw_z_samples(n_samples, z_samples)

        ops_work = np.asarray(ops, dtype=np.int32)
        generators_work = np.asarray(generators, dtype=np.int32)
        z_work = np.asarray(z_samples, dtype=np.int32)

        # Corresponds to https://github.com/XanaduAI/iqpopt/blob/bf207502c317364746bd56ae0dd1b4d04e492c7b/src/iqpopt/iqp_optimizer.py#L352
        ops_gen = (ops_work @ generators_work.T) % 2
        # Corresponds to https://github.com/XanaduAI/iqpopt/blob/bf207502c317364746bd56ae0dd1b4d04e492c7b/src/iqpopt/iqp_optimizer.py#L353
        samples_gates = 1 - 2 * ((z_work @ generators_work.T) % 2)
        # Corresponds to https://github.com/XanaduAI/iqpopt/blob/bf207502c317364746bd56ae0dd1b4d04e492c7b/src/iqpopt/iqp_optimizer.py#L371
        par_ops_gates = 2.0 * effective_params * ops_gen
        # Corresponds to https://github.com/XanaduAI/iqpopt/blob/bf207502c317364746bd56ae0dd1b4d04e492c7b/src/iqpopt/iqp_optimizer.py#L372
        expvals = np.cos(par_ops_gates @ samples_gates.T)

        # Corresponds to code block after https://github.com/XanaduAI/iqpopt/blob/bf207502c317364746bd56ae0dd1b4d04e492c7b/src/iqpopt/iqp_optimizer.py#L359
        if self.spin_sym:
            ops_sum = ops_work.sum(axis=-1, keepdims=True)
            samples_sum = z_work.sum(axis=-1, keepdims=True).T
            ini_spin_sym = 2 - (ops_sum % 2) - 2 * (samples_sum % 2)
            expvals = ini_spin_sym * expvals

        if return_samples:
            return expvals

        # Corresponds to https://github.com/XanaduAI/iqpopt/blob/bf207502c317364746bd56ae0dd1b4d04e492c7b/src/iqpopt/iqp_optimizer.py#L378
        return expvals.mean(axis=-1)

    def iqp_circuit(self, params, init_coefs=None):
        qc = QuantumCircuit(self.n_qubits)

        # Spin symmetry
        if self.spin_sym:
            qc.h(0)
            for i in range(1, self.n_qubits):
                qc.cx(0, i)

        # Hadamards
        for i in range(self.n_qubits):
            qc.h(i)

        # General multi-qubit Pauli-Z rotation: e^{-i*theta * Z_{q0}...Z_{qk-1}}
        # Decomposed as: cascade CX onto target, RZ(2*theta), reverse cascade CX.
        # For 1 qubit: just RZ(2*theta). For 2 qubits: CX(i,j), RZ, CX(i,j). Etc.
        def apply_pauli_z_rotation(qubits, theta):
            target = qubits[-1]
            controls = qubits[:-1]
            for ctrl in controls:
                qc.cx(ctrl, target)
            qc.rz(2 * theta, target)
            for ctrl in reversed(controls):
                qc.cx(ctrl, target)

        # Initial gates
        if self.init_gates is not None:
            if init_coefs is None:
                raise ValueError("init_coefs must be provided if init_gates are used")

            for par, gate in zip(init_coefs, self.init_gates):
                apply_pauli_z_rotation(gate[0], par)

        # Trainable gates
        # Loop over each gate in the ansatz and its corresponding parameter
        for par, gate in zip(params, self.gates):
            apply_pauli_z_rotation(gate[0], par)

        # Final Hadamards
        for i in range(self.n_qubits):
            qc.h(i)

        return qc

    def sample(self, params, init_coefs=None, shots=1024):
        if self.bitflip:
            return self._sample_bitflip(params, init_coefs, shots)

        qc = self.iqp_circuit(params, init_coefs)

        qc.measure_all()

        sampler = StatevectorSampler()
        job = sampler.run([qc], shots=shots)
        result = job.result()

        pub_result = result[0]
        counts = pub_result.data.meas.get_counts()

        samples = []
        for bitstring, cnt in counts.items():
            arr = np.array([int(b) for b in reversed(bitstring)])
            samples.extend([arr] * cnt)

        return np.array(samples)

    def probs(self, params, init_coefs=None):
        if self.bitflip:
            raise NotImplementedError("probs not implemented for bitflip circuits")

        qc = self.iqp_circuit(params, init_coefs)

        # Use Statevector directly
        state = Statevector.from_instruction(qc)
        probs = np.abs(state.data) ** 2

        return probs
    
    # IQPopt expectation values computation for the training loop
    def op_expval(
        self,
        params,
        ops,
        init_coefs=None,
        n_samples=1000,
        z_samples=None,
        return_samples=False,
    ):
        """
        Estimate Pauli-Z expectation values for the current model.

        For IQP circuits, this uses the paper-style Monte Carlo estimator over
        uniformly sampled bitstrings z. For bitflip models, it uses the exact
        closed-form product-of-cosines expression.
        """
        ops_array = np.asarray(ops, dtype=np.int8)
        single_op = ops_array.ndim == 1
        ops_array = np.atleast_2d(ops_array)

        if self.bitflip:
            expvals = self._bitflip_op_expval(params, ops_array, init_coefs)
            if return_samples:
                expvals = expvals[:, None]
        else:
            expvals = self._iqp_op_expval(
                params,
                ops_array,
                init_coefs=init_coefs,
                n_samples=n_samples,
                z_samples=z_samples,
                return_samples=return_samples,
            )

        if return_samples:
            return expvals if not single_op else expvals[0]

        return expvals if not single_op else expvals[0]