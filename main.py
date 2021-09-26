import networkx as nx 
import cv2
import numpy as np

import tensorflow as tf
import sonnet as snt

from graph_nets.demos_tf2 import models
from graph_nets import utils_np
from graph_nets import utils_tf
from graph_nets.demos_tf2 import models

import time
import matplotlib.pyplot as plt

SEED = 1
np.random.seed(SEED)
tf.random.set_seed(SEED)


#adj, adjcorner, gausssian
neighbors = "adjcorner" 

def coords(coord, height):
    counter = coord[0] + coord[1] * height
    return counter

def createGraph(mode, im):
    G = nx.MultiDiGraph()

    height = im.shape[0]
    width = im.shape[1]
    counter = 0
    for y in range(height):
        for x in range(width):
            G.add_node(counter, features=im[y][x], pos=(x, y))
            counter += 1
    
    for node in list(G.nodes):
        i = [node - ((node // im.shape[0]) * im.shape[0]), node // im.shape[0]]
        if mode == "adj" or mode == "adjcorner":
            if i[0] < im.shape[0] - 1:
                G.add_edge(node, coords((i[0] + 1, i[1]), im.shape[0]), features=[])
            if i[0] > 0:
                G.add_edge(node, coords((i[0] - 1, i[1]), im.shape[0]), features=[])
            if i[1] < im.shape[1] - 1: 
                G.add_edge(node, coords((i[0], i[1] + 1), im.shape[0]), features=[])
            if i[1] > 0:
                G.add_edge(node, coords((i[0], i[1] - 1), im.shape[0]), features=[])
        if mode == "adjcorner":
            if i[0] < im.shape[0] - 1 and i[1] < im.shape[1] - 1:
                G.add_edge(node, coords((i[0] + 1, i[1] + 1), im.shape[0]), features=[])
            if i[0] > 0 and i[1] > 0:
                G.add_edge(node, coords((i[0] - 1, i[1] - 1), im.shape[0]), features=[])
            if i[0] < im.shape[0] - 1 and i[1] > 0:
                G.add_edge(node, coords((i[0] + 1, i[1] - 1), im.shape[0]), features=[])
            if i[0] > 0 and i[1] < im.shape[1] - 1:
                G.add_edge(node, coords((i[0] - 1, i[1] + 1), im.shape[0]), features=[])
    return G

def loadExamples(xPath, yPath, numExamples=-1, mode="adjcorner"):
    x, y = np.load(xPath)[:numExamples], np.load(yPath)[:numExamples]
    xGraphTuples = [utils_np.networkxs_to_graphs_tuple([createGraph(mode, i)]) for i in x]
    yGraphTuples = [utils_np.networkxs_to_graphs_tuple([createGraph(mode, i)]) for i in y]
    return xGraphTuples, yGraphTuples


def createLossFunc():
    mse = tf.keras.losses.MeanSquaredError()
    def loss(input, outputs):
        loss = [
            mse(target.nodes, output.nodes) for output in outputs
        ]
    return loss

x, y = loadExamples("x.npy", "y.npy", numExamples=3, mode=neighbors)
loss = createLossFunc()

learning_rate = 1e-3
optimizer = snt.optimizers.Adam(learning_rate)

model = models.EncodeProcessDecode(edge_output_size=0, node_output_size=3)
last_iteration = 0
logged_iterations = []
losses_tr = []
corrects_tr = []
solveds_tr = []
losses_ge = []
corrects_ge = []
solveds_ge = []


# Training.
def update_step(inputs_tr, targets_tr):
    with tf.GradientTape() as tape:
        outputs_tr = model(inputs_tr, num_processing_steps_tr)
        # Loss.
        loss_tr = create_loss(targets_tr, outputs_tr)
        loss_tr = tf.math.reduce_sum(loss_tr) / num_processing_steps_tr

    gradients = tape.gradient(loss_tr, model.trainable_variables)
    optimizer.apply(gradients, model.trainable_variables)
    return outputs_tr, loss_tr


#graph_tuple = utils_np.networkxs_to_graphs_tuple([graph])
#pos = nx.get_node_attributes(graph, 'pos')
#nx.draw(graph, pos, with_labels=True)
#plt.show()