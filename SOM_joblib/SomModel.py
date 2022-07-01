import numpy as np
import time


class SomUnit:
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
    som_node = SomUnit(node_id, x, y, weight_dim)
    return som_node


def euclidean_distance(node, target):
    pw = np.power(node - target, 2)
    distance = np.sqrt(np.sum(pw))
    return distance


class SomGrid:
    def __init__(self, grid_rows, grid_cols, weight_dim):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.weight_dim = weight_dim
        self.nodes_id = []
        self.nodes = []

        self.make_grid()

    def make_grid(self):
        node_id = 0
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                self.nodes.append(make_node(i, j, node_id, self.weight_dim))
                self.nodes_id.append(node_id)
                node_id += 1

    def print_grid(self):
        for i in range(len(self.nodes)):
            print(self.nodes[i].weights)


class SomLayer:
    def __init__(self, grid_rows, grid_cols, weight_dim, lrate):
        self.som_grid = SomGrid(grid_rows, grid_cols, weight_dim)
        self.init_lrate = lrate
        self.init_radius = np.max([grid_rows, grid_cols]) / 2
        # self.alpha = alpha
        self.BMU = None

    def bmu(self, inpt):
        distances = []
        indexes = []
        for neuron in self.som_grid.nodes:
            distance = euclidean_distance(neuron.weights, inpt)
            distances.append(distance)
            indexes.append(neuron.get_id())
        BMU_index = indexes[np.argmin(distances)]
        return BMU_index

    def adjust_weight(self, lrate, radius, bmu_index, inp_vec):
        for neuron in self.som_grid.nodes:
            if neuron.isNeighbor:
                bmu = self.som_grid.nodes[bmu_index]
                coordinates_neuron = np.array([neuron.x, neuron.y])
                coordinates_bmu = np.array([bmu.x, bmu.y])
                dist_bmu = euclidean_distance(coordinates_neuron, coordinates_bmu)
                theta = np.exp(-dist_bmu / (2 * np.power(radius, 2)))
                neuron.weights = neuron.weights + theta * lrate * (inp_vec - neuron.weights)

    def neighborhood(self, bmu_index, radius):
        for neuron in self.som_grid.nodes:
            bmu = self.som_grid.nodes[bmu_index]
            coordinates_neuron = np.array([neuron.x, neuron.y])
            coordinates_bmu = np.array([bmu.x, bmu.y])
            dist = euclidean_distance(coordinates_neuron, coordinates_bmu)
            if dist <= radius:
                neuron.isNeighbor = True

    def reset_grid(self):
        for neuron in self.som_grid.nodes:
            neuron.isNeighbor = False

    def training_som(self, epochs, inp_vec):
        alpha = epochs / np.log(self.init_lrate)
        for epoch in range(epochs):
            print("================= EPOCH " + str(epoch + 1) + "/" + str(epochs) + " =================")
            radius = self.init_radius * np.exp(-epoch / alpha)
            learning_rate = self.init_lrate * np.exp(-epoch / epochs)
            for i in range(inp_vec.shape[0]):
                bmu_index = self.bmu(inp_vec[i, :])
                self.neighborhood(bmu_index, radius)
                # TODO: si può mettere l'adjust weights nella funzione prima e risparmiarci un ciclo per cercare i nodi vicini
                self.adjust_weight(learning_rate, radius, bmu_index, inp_vec[i, :])
                # print(neuron.x, neuron.y)
                self.reset_grid()
