import qecp
from simulator_to_hypergraph import *
from hypergraph_to_decoder import *

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

simulator.generate_random_errors(noise_model)
simulator.generate_sparse_error_pattern()
measurement = simulator.generate_sparse_measurement()

defect_measurement_indices = [model_hypergraph.vertex_index(v) for v in measurement.defects]

hypergraph, num_vertices = qecp_to_hypergraph(simulator, noise_model)

hyperion, hyperion_subgraph, latency = hypergraph_to_mwpf(hypergraph, num_vertices, defect_measurement_indices, True)

print(latency)
print(hyperion_subgraph)  # out: [3, 5], weighted 160
_, bound = hyperion.subgraph_range()
print((bound.lower, bound.upper))  # out: (Fraction(160, 1), Fraction(160, 1))

fusion_subgraph, total_weight = hypergraph_to_fb(hypergraph, num_vertices, defect_measurement_indices)

print(fusion_subgraph)
print(total_weight)