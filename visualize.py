import qecp
import numpy as np
import json
from analysis import *
from simulator_to_hypergraph import *
from hypergraph_to_decoder import *

d = 3
noisy_measurements = 3
p = 0.01

code_type = qecp.CodeType.RotatedPlanarCode
code_size = qecp.CodeSize(noisy_measurements, d, d)
simulator = qecp.Simulator(code_type, code_size)

noise_model_builder = qecp.NoiseModelBuilder.Phenomenological
noise_model = qecp.NoiseModel(simulator)
noise_model_builder.apply(simulator, noise_model, p)

model_hypergraph = qecp.create_hypergraph(simulator, noise_model)

hypergraph, num_vertices = qecp_to_hypergraph(simulator, noise_model)

visualize_graph(hypergraph_to_graph(hypergraph, num_vertices))