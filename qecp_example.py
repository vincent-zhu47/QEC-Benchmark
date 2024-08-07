import qecp
from simulator_to_hypergraph import *
from hypergraph_to_decoder import *
import numpy as np
import json

d = 3
noisy_measurements = 3
p = 0.05

code_type = qecp.CodeType.RotatedPlanarCode
code_size = qecp.CodeSize(noisy_measurements, d, d)
simulator = qecp.Simulator(code_type, code_size)

noise_model_builder = qecp.NoiseModelBuilder.Phenomenological
noise_model = qecp.NoiseModel(simulator)
noise_model_builder.apply(simulator, noise_model, p)

model_hypergraph = qecp.create_hypergraph(simulator, noise_model)

hypergraph, num_vertices = qecp_to_hypergraph(simulator, noise_model)

latencies = np.empty(shape=1)

mwpf_hypergraph = hypergraph_to_mwpf(hypergraph)

for i in range(1) :
    simulator.generate_random_errors(noise_model)
    simulator.generate_sparse_error_pattern()
    measurement = simulator.generate_sparse_measurement()

    defect_measurement_indices = [model_hypergraph.vertex_index(v) for v in measurement.defects]
    
    hyperion, hyperion_subgraph, latency = run_mwpf(mwpf_hypergraph, num_vertices, defect_measurement_indices)
    latencies[i] = latency
    
    (logical_i, logical_j) = simulator.validate_correction(hyperion_subgraph)
    is_qec_failed = logical_i or logical_j
    print(is_qec_failed)
    
    print("Trial ", i, ": ", latency)

# with open('data.txt', 'r') as file:
#     data = file.read().rstrip()
# latencies = json.loads("output.txt")

                       
# print(hyperion_subgraph)  # out: [3, 5], weighted 160
# _, bound = hyperion.subgraph_range()
# print((bound.lower, bound.upper))  # out: (Fraction(160, 1), Fraction(160, 1))

# fusion_subgraph, total_weight = hypergraph_to_fb(hypergraph, num_vertices, defect_measurement_indices)

# print(fusion_subgraph)
# print(total_weight)