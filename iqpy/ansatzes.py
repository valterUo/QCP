import itertools
import networkx as nx


def nearest_neighbour_IQP_ansatz(n_qubits):
    """
    Nearest-neighbour IQP ansatz.

    Params:
        n_qubits: number of qubits in the circuit

    Returns:
        gates: list of gates in IQP format [[[i,j]], ...]
    """
    gates = []

    # Single-qubit Z terms
    for i in range(n_qubits):
        gates.append([[i]])

    # Nearest-neighbour ZZ interactions
    for i in range(n_qubits - 1):
        gates.append([[i, i + 1]])

    return gates


def fully_connected_IQP_ansatz(n_qubits, max_weight=2):
    """
    Fully-connected IQP ansatz.

    Params:
        n_qubits: number of qubits in the circuit

    Returns:
        gates: list of gates in IQP format [[[i,j]], ...]
    """
    gates = []

    for weight in range(1, max_weight + 1):
        for combo in itertools.combinations(range(n_qubits), weight):
            gates.append([list(combo)])

    # Single-qubit Z terms
    #for i in range(n_qubits):
    #    gates.append([[i]])

    # Fully-connected ZZ interactions
    #for i in range(n_qubits):
    #    for j in range(i + 1, n_qubits):
    #        gates.append([[i, j]])

    return gates


def nodes_within_distance(G, source, distance):
    """
    Given a graph G, returns all nodes on within a specified distance of a source node.

    Params:
        G: A networkx graph object
        source: source node
        distance: maximum distance from source node

    Returns:
        list of nodes within specified distance
    """

    current_level = {source}
    visited = {source}
    all_nodes_within_distance = set(current_level)

    for _ in range(distance):
        next_level = set()
        for node in current_level:
            for neighbor in G.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_level.add(neighbor)
        current_level = next_level
        all_nodes_within_distance.update(current_level)

    all_nodes_within_distance.discard(source)
    return list(all_nodes_within_distance)


def nearest_neighbour_from_graph_IQP_ansatz(G, distance: int, max_weight=2):
    """
    For every node on a graph G, find all other nodes within a specified distance, and create all gates
    up to a maximum weight from these nodes.

    Params:
        G: networkx graph that determines the distances, or a path reference to a networkx adjacency list file
        distance: maximum distance to consider
        max_weight: maximum weight of the generators of the gates

    Returns:
        list of lists specifying the gates
    """

    if isinstance(G, str):
        G = nx.read_adjlist(G, nodetype=int)

    gates = []
    for source in list(G.nodes):
        neighbours = nodes_within_distance(G, source, distance)
        for weight in range(0, max_weight):
            for combo in itertools.combinations(neighbours, weight):
                new_gate = list(combo) + [source]
                new_gate = [int(elem) for elem in new_gate]
                perms = list(itertools.permutations(new_gate))
                perms = [[list(perm)] for perm in perms]
                if not any([perm in gates for perm in perms]):
                    gates.append([new_gate])

    return gates
