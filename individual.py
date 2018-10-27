import numpy as np
from plugins.genome_visualizer import make_dot_genome


class Phase:
    def __init__(self, genome):
        self.genome = genome
        self.key = self.phase_key_encoder(self.genome)
        self.active_nodes = self.count_phase_active_nodes(self.genome)
        self.connections = self.count_phase_connections(self.genome)

    @staticmethod
    def phase_key_encoder(genome):
        genome_key = []
        for op in genome:
            op = [int(x) for x in op]
            genome_key.append("".join(map(str, op)))
        return '-'.join(genome_key)

    @staticmethod
    def count_phase_active_nodes(genome):
        active_nodes = len(genome)
        for i in range(len(genome) - 1):
            if np.sum(genome[i]) < 1:
                non_active_node = 1
                for j in range(i + 1, len(genome) - 1):
                    if genome[j][i + 1] > 0:
                        non_active_node = 0
                        break
                active_nodes = active_nodes - non_active_node

        first_node_redundant = 1
        for k in range(len(genome) - 1):
            if genome[k][0] > 0:
                first_node_redundant = 0
                break
        return active_nodes - first_node_redundant

    @staticmethod
    def count_phase_connections(genome):
        connections = sum([sum(node) for node in genome])
        # calculate output connection for each node
        output_connection = [0] * len(genome)
        for bit in range(len(genome) - 1):
            for node in range(bit, len(genome) - 1):
                if genome[node][bit] > 0:
                    output_connection[bit] = 1
                    break
        # calculate input connection for each node
        input_connection = [0]
        for node in range(len(genome) - 1):
            if sum(genome[node]) > 0:
                input_connection.append(1)
            else:
                input_connection.append(0)
        for bit in range(len(output_connection)):
            if output_connection[bit] != input_connection[bit]:
                connections += 1

        return connections


class Network:
    def __init__(self, nid, genome, n_phases=3, n_nodes=4):
        self.id = nid
        self.genome = genome
        self.n_phases = n_phases
        self.n_nodes = n_nodes
        self.phases, self.key, self.active_nodes = self.encoder(self.genome, self.n_phases)
        self.n_params = None
        self.n_FLOPs = None

    @staticmethod
    def encoder(genome, n_phases):
        # start decoding for each phase
        phases = []
        key = []
        active_nodes = []
        for p in range(n_phases):
            phase = Phase(genome[p])
            phases.append(phase)
            key.append(phase.key)
            active_nodes.append(phase.active_nodes)

        return phases, key, active_nodes

    @staticmethod
    def render_networks(genome, save_pth, save_format='png'):
        """
        Renders the graphviz and image files of network architecture defined by a genome.
        :param population: list of nsga individuals.
        :param nsga_details: bool, true if we want rank and crowding distance in the title.
        :param show_genome: bool, true if we want the genome in the title.
        """

        viz = make_dot_genome(genome, format=save_format)
        viz.render(save_pth, view=False)
