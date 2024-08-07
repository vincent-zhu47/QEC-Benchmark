import stim
import pymatching
import numpy as np
import time
import matplotlib.pyplot as plt
from analysis import *
    
circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_x",
    rounds=1,
    distance=9,
    after_clifford_depolarization=0.001,
    after_reset_flip_probability=0.001,
    before_measure_flip_probability=0.001,
    before_round_data_depolarization=0.001
)

sampler = circuit.compile_detector_sampler()
detection_events, observable_flips = sampler.sample(1, separate_observables=True)

detector_error_model = circuit.detector_error_model(decompose_errors=True)
matcher = pymatching.Matching.from_detector_error_model(detector_error_model)
print(matcher.edges())

latency = np.empty(shape=1000000)

for i in range(1000000):
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(1, separate_observables=True)
    
    start = time.time()
    predictions = matcher.decode_batch(detection_events)
    end = time.time()
    
    latency[i] = end - start

bins = np.logspace(np.log10(max(latency)), np.log10(min(latency)), 2000)

binplace = np.digitize(latency, bins)

counts = np.zeros(shape=2000)
for i in range(2000):
    counts[i] = len(np.where(binplace == i)[0])

plt.loglog(bins, counts, ".-")
plt.xlim(1e-6, 1e-2)
plt.ylim(0.5, 1e6)
plt.ylabel("Sample Count")
plt.xlabel("Latency (s)")
plt.show()
# plt.savefig("pymatching_latency.pdf")

create_error_plots(bins, counts, 9)