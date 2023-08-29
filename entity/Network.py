# Classe d'un réseau de neurones

import numpy as np


class Network:

    def __init__(self, input_layer, hidden_layer, output_layer):
        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer

    def train(self, input_data, output_data, iterations):
        for i in range(iterations):
            for j in range(len(input_data)):
                self.input_layer.set_values(input_data[j])
                self.hidden_layer.set_values(self.input_layer)
                self.output_layer.set_values(self.hidden_layer)
                self.output_layer.update_weights(output_data[j])
                self.hidden_layer.update_weights(self.output_layer)
                self.input_layer.update_weights(self.hidden_layer)

    def predict(self, input_data):
        self.input_layer.set_values(input_data)
        self.hidden_layer.set_values(self.input_layer)
        self.output_layer.set_values(self.hidden_layer)
        return self.output_layer.get_values()

    def __str__(self):
        return "Network:\n" + str(self.input_layer) + "\n" + str(self.hidden_layer) + "\n" + str(self.output_layer) + "\n"

    # fonctions d'import et d'export des poids du réseau
    def export(self, filename):
        with open(filename, "w") as file:
            file.write(str(self))

    def import_(self, filename):
        with open(filename, "r") as file:
            lines = file.readlines()
            self.input_layer.weights = np.array(
                eval(lines[1].replace("Layer:\n", "")))
            self.hidden_layer.weights = np.array(
                eval(lines[3].replace("Layer:\n", "")))
            self.output_layer.weights = np.array(
                eval(lines[5].replace("Layer:\n", "")))
