import stim
import time
from hypergraph_to_decoder import *
from simulator_to_hypergraph import *

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

hypergraph = stim_to_hypergraph(detector_error_model)

visualize_hypergraph(hypergraph)

latency = []

for _ in range(1):
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(1, separate_observables=True)

    detector_error_model = circuit.detector_error_model(decompose_errors=False)

    hypergraph = stim_to_hypergraph(detector_error_model)
    
    start = time.time()
    hyperion, hyperion_subgraph = hypergraph_to_mwpf(hypergraph, detector_error_model.num_detectors, detection_events)
    end = time.time()
    latency.append(end-start)
    
print(latency)
print(sum(latency)/len(latency))
      
# print(hyperion_subgraph)  # out: [3, 5], weighted 160
# _, bound = hyperion.subgraph_range()
# print((bound.lower, bound.upper))  # out: (Fraction(160, 1), Fraction(160, 1))

latency = []

for _ in range(1):
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(1, separate_observables=True)

    detector_error_model = circuit.detector_error_model(decompose_errors=False)

    hypergraph = stim_to_hypergraph(detector_error_model)
    
    start = time.time()
    fusion_subgraph, total_weight = hypergraph_to_fb(hypergraph, detector_error_model.num_detectors, detection_events)
    end = time.time()
    latency.append(end-start)
    
print(latency)
print(sum(latency)/len(latency))

#print(fusion_subgraph)
#print(total_weight)
