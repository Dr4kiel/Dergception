'''
author: Tristan GAUTIER
date: 29-08-2023

Initiation aux réseaux de neurones (Identification d'image)
'''

# Importation des librairies
import numpy as np

# Importation des entités
from entity.Network import Network
from entity.Layer import Layer
import os

'''
Génération d'un réseau de neurones permettant d'identifier des images (500x500 pixels) :
- 1 couche d'entrée (500x500 = 250 000 neurones)
- 1 couche cachée (100 neurones)
- 1 couche de sortie (1 neurone)

input_data : matrice de données d'entrée (250 000 colonnes) (1 ligne = 1 image)
output_data : matrice de données de sortie (1 colonne) (1 ligne = 1 image) (1 si l'image est un dragon, 0 sinon)

'''

# Création des couches
input_layer = Layer(250000, 100)
hidden_layer = Layer(100, 1)
output_layer = Layer(1, 1)

# Création du réseau
network = Network(input_layer, hidden_layer, output_layer)

# Importation des poids du réseau
# network.import_("data/network.txt")

'''
Identification d'une image
'''

# Importation de l'image
image = np.loadtxt("data/image.txt")

# Identification de l'image
print(network.predict(image))

'''
Entraînement du réseau
'''

# Importation des données d'entraînement
input_data = []
for filename in os.listdir("data/input_data"):
    input_data.append(np.loadtxt("data/input_data/" + filename))
output_data = np.matrix(np.loadtxt("data/output_data.txt")).T

# Entraînement du réseau
network.train(input_data, output_data, 5)

# Exportation des poids du réseau
network.export("data/network.txt")

'''
Identification d'une image
'''

# Importation de l'image
image = np.loadtxt("data/image.txt")

# Identification de l'image
print(network.predict(image))


'''
Résultats attendus :
- 0.9999999999999999

Fin du programme
'''
