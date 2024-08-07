import stim
from hypergraph_to_decoder import *
from simulator_to_hypergraph import *
import time

circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_x",
    rounds=3,
    distance=3,
    after_clifford_depolarization=0.01,
    after_reset_flip_probability=0.01,
    before_measure_flip_probability=0.01,
    before_round_data_depolarization=0.01
)

sampler = circuit.compile_detector_sampler()

detection_events, observable_flips = sampler.sample(1, separate_observables=True)

detector_error_model = circuit.detector_error_model(decompose_errors=False)
# For dem file:
# detector_error_model = stim.DetectorErrorModel.from_file("")

hypergraph, num_vertices = stim_to_hypergraph(detector_error_model)

detection_events = [i for i, x in enumerate(detection_events[0]) if x == True]

mwpf_hypergraph = hypergraph_to_mwpf(hypergraph)

hyperion, hyperion_subgraph, latency = run_mwpf(mwpf_hypergraph, num_vertices, detection_events)
      
print(hyperion_subgraph)  # out: [3, 5], weighted 160
_, bound = hyperion.subgraph_range()
print((bound.lower, bound.upper))  # out: (Fraction(160, 1), Fraction(160, 1))
print(latency)

# start = time.time()
# fusion_subgraph, total_weight, latency = hypergraph_to_fb(hypergraph, num_vertices, detection_events)
# end = time.time()
# print(end-start-latency)

# print(fusion_subgraph)
# print(total_weight)
# print(latency)
