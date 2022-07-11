import numpy as np
from joblib import Parallel, delayed


class SomUnitP:
    def __init__(self, id, x, y, weight_dim):
        self.id = id
        self.x = x
        self.y = y
        self.weight_dim = weight_dim
        self.isNeighbor = False
        self.weights = np.random.rand(weight_dim)

    def get_id(self):
        return self.id

    def get_coordinates(self):
        return [self.x, self.y]

    def get_weights(self):
        return self.weights


def make_node(x, y, node_id, weight_dim):
    som_node = SomUnitP(node_id, x, y, weight_dim)
    return som_node


def euclidean_distance(node, target):
    pw = np.power(node - target, 2)
    distance = np.sqrt(np.sum(pw))
    return distance


class SomGridP:
    def __init__(self, grid_rows, grid_cols, weight_dim, N_jobs):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.weight_dim = weight_dim
        self.N_jobs = N_jobs
        self.nodes_id = []
        self.nodes = []
        self.make_grid()

    def build_grid(self, i, j):
        node_id = i * self.grid_cols + j
        self.nodes.append(make_node(i, j, node_id, self.weight_dim))
        self.nodes_id.append(node_id)

    def make_grid(self):
        Parallel(n_jobs=self.N_jobs, backend="threading", prefer="threads")([delayed(self.build_grid)(i, j)
                                                                             for j in range(self.grid_cols) for i in
                                                                             range(self.grid_rows)])

    def print_grid(self):
        for i in range(len(self.nodes)):
            print(self.nodes[i].weights)


class SomLayerP:
    def __init__(self, grid_rows, grid_cols, weight_dim, lrate, N_jobs):
        self.som_grid = SomGridP(grid_rows, grid_cols, weight_dim, N_jobs)
        self.init_lrate = lrate
        self.init_radius = np.max([grid_rows, grid_cols]) / 2
        self.distances = []  # list of tuples [(distance, nodeID)]
        self.indexes = []
        self.BMU = None
        self.N_jobs = N_jobs

    def bmu_cycle(self, index, inpt):
        neuron = self.som_grid.nodes[index]
        distance = euclidean_distance(neuron.weights, inpt)
        return distance, neuron.get_id()

    def bmu(self, inpt):
        self.distances = Parallel(n_jobs=self.N_jobs, backend="threading", prefer="threads")(
            [delayed(self.bmu_cycle)(i, inpt) for i in range(len(self.som_grid.nodes))])

        # Find the tuple in list with the minimum value in position 0
        min_dist_tuple = min(self.distances, key=lambda t: t[0])
        # BMU index is the value in position 1 of the tuple
        BMU_index = min_dist_tuple[1]
        return BMU_index

    def cycle_adjust_weight(self, neuron, bmu_index, radius, inp_vec, lrate):
        if neuron.isNeighbor:
            bmu = self.som_grid.nodes[bmu_index]
            coordinates_neuron = np.array([neuron.x, neuron.y])
            coordinates_bmu = np.array([bmu.x, bmu.y])
            dist_bmu = euclidean_distance(coordinates_neuron, coordinates_bmu)
            theta = np.exp(-dist_bmu / (2 * np.power(radius, 2)))
            neuron.weights = neuron.weights + theta * lrate * (inp_vec - neuron.weights)

    def adjust_weight(self, radius, bmu_index, inp_vec, lrate):
        Parallel(n_jobs=self.N_jobs, backend="threading", prefer="threads")(
            [delayed(self.cycle_adjust_weight)(neuron, bmu_index, radius, inp_vec, lrate)
             for neuron in self.som_grid.nodes])

    def cycle_neighborhood(self, neuron, bmu_index, radius):
        bmu = self.som_grid.nodes[bmu_index]
        coordinates_neuron = np.array([neuron.x, neuron.y])
        coordinates_bmu = np.array([bmu.x, bmu.y])
        dist = euclidean_distance(coordinates_neuron, coordinates_bmu)
        if dist <= radius:
            neuron.isNeighbor = True

    def neighborhood(self, bmu_index, radius):
        Parallel(n_jobs=self.N_jobs, backend="threading", prefer="threads")(
            [delayed(self.cycle_neighborhood)(neuron, bmu_index, radius)
             for neuron in self.som_grid.nodes])

    def cycle_reset_grid(self, neuron):
        neuron.isNeighbor = False

    def reset_grid(self):
        Parallel(n_jobs=self.N_jobs, backend="threading", prefer="threads")(
            [delayed(self.cycle_reset_grid)(neuron) for neuron in self.som_grid.nodes])

    def training_som(self, epochs, inp_vec):
        alpha = epochs / np.log(self.init_lrate)
        for epoch in range(epochs):
            print("================= EPOCH " + str(epoch + 1) + "/" + str(epochs) + " =================")
            radius = self.init_radius * np.exp(-epoch / alpha)
            learning_rate = self.init_lrate * np.exp(-epoch / epochs)
            for i in range(len(inp_vec)):
                bmu_index = self.bmu(inp_vec[i, :])
                self.neighborhood(bmu_index, radius)
                self.adjust_weight(radius, bmu_index, inp_vec[i, :], learning_rate)
                self.reset_grid()
