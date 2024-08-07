import qecp
import numpy as np
import json
from analysis import *
from simulator_to_hypergraph import *
from hypergraph_to_decoder import *

get_from_file = False;

if not get_from_file:
    d = 5
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

    latencies = np.empty(shape=10000)

    fb_graph = hypergraph_to_fb(hypergraph, num_vertices)

    for i in range(10000):
        simulator.generate_random_errors(noise_model)
        simulator.generate_sparse_error_pattern()
        measurement = simulator.generate_sparse_measurement()

        defect_measurement_indices = [model_hypergraph.vertex_index(v) for v in measurement.defects]
        
        hyperion, hyperion_subgraph, latency = run_fb(fb_graph, num_vertices, defect_measurement_indices)
        latencies[i] = latency
        
        print("Trial ", i, ": ", latency)

    with open('output.json', 'w') as filehandle:
        json.dump(latencies.tolist(), filehandle)

with open("output.json", "r") as f: 
    latencies = json.load(f)

latencies = np.array(latencies)

bins = np.logspace(np.log10(max(latencies)), np.log10(np.min(latencies[np.nonzero(latencies)])), 2000)

binplace = np.digitize(latencies, bins)

counts = np.zeros(shape=2000)
for i in range(2000):
    counts[i] = len(np.where(binplace == i)[0])

plt.loglog(bins, counts, ".-")
plt.xlim(1e-7, 1e-1)
plt.ylim(0.5, 1e6)
plt.ylabel("Sample Count")
plt.xlabel("Latency (s)")
plt.title("Latency Distribution")
# plt.savefig("fb_latency.pdf")
plt.show()

# create_error_plots(bins, counts, 5)