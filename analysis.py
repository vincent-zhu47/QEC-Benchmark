import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Single Conditional Gate
def single_cond_gate(latency: list, count: list, distance: int) -> tuple[np.ndarray, np.ndarray]:
    # Latency should be in seconds

    # Convert to numpy array for easier manipulation
    latency = np.array(latency)
    count = np.array(count)
    
    # Calculate error rate
    error_rate = (1+latency*1e6/distance)
    
    # Change counts to probabilities
    total = np.sum(count)
    prob = count/total
    
    return error_rate, prob

# Multi-Conditioned Gate
def multi_cond_gate(latency: list, count: list, trials: int, n: int, distance: int) -> tuple[np.ndarray, np.ndarray]:
    
    # Convert to numpy array for easier manipulation
    latency = np.array(latency)
    count = np.array(count)
    
    # Change counts to probabilities       
    total = sum(count)
    prob = count/total
    
    # Size up latencies to integers for use in discrete distribution
    int_latency = latency*1e11
    int_latency = np.round(int_latency)
    
    # Create discrete distribution with integer indices representing values
    dist = stats.rv_discrete(values=(int_latency, prob))
    
    # Take "trials" max samples of "n" values from distribution
    values = np.empty(shape=trials)
    for i in range(0,trials):
        num = np.max(dist.rvs(size=n))
        values[i] = num
        
    # Convert back to original latencies
    values = values/1e11
    
    # Take counts of unique values
    sampled_latency, sampled_count = np.unique(values, return_counts=True)
    
    # Calculate error rate and probabilities from distribution sample
    error_rate = (1+sampled_latency*1e6/distance)
    prob = sampled_count/trials
    
    return error_rate, prob

# Consecutive Conditioned Gate
def consec_cond_gate(latency: list, count: list, trials: int, n: int, distance: int) -> tuple[np.ndarray, np.ndarray]:
    
    # Convert to numpy array for easier manipulation
    latency = np.array(latency)
    count = np.array(count)
    
    # Change counts to probabilities       
    total = sum(count)
    prob = count/total
    
    # Size up latencies to integers for use in discrete distribution
    int_latency = latency*1e11
    int_latency = np.round(int_latency)
    
    # Create discrete distribution with integer indices representing values
    dist = stats.rv_discrete(values=(int_latency, prob))
    
    # Take "trials" sum samples of "n" values from distribution
    values = np.empty(shape=trials)
    for i in range(0,trials):
        num = np.sum(dist.rvs(size=n))
        values[i] = num
        
    # Convert back to original latencies
    values = values/1e11
    
    # Take counts of unique values
    sampled_latency, sampled_count = np.unique(values, return_counts=True)
    
    # Calculate error rate and probabilities from distribution sample
    error_rate = (1+sampled_latency*1e6/distance)
    prob = sampled_count/trials
    
    return error_rate, prob

def create_error_plots(latency: list, count: list, distance: int):
    single_error_rate, single_prob = single_cond_gate(latency, count, distance)

    plt.cla()
    plt.loglog(single_error_rate, single_prob, ".-")
    plt.xlim(1e0, 1e7)
    plt.ylim(1e-9, 1e0)
    plt.title("Logical Error Rate Distribution for \n Single Conditional Gate")
    plt.ylabel("Probability")
    plt.xlabel("Logical Error Rate")
    plt.savefig("mwpf_single.pdf")
    plt.show()

    trials = 1000000
    n = 3
    multi_error_rate, multi_prob = multi_cond_gate(latency, count, trials, n, distance)

    plt.cla()
    plt.loglog(multi_error_rate, multi_prob, ".-")
    plt.xlim(1e0, 1e7)
    plt.ylim(1e-9, 1e0)
    if n == 1:
        plt.title("Logical Error Rate Distribution for \n Single Conditional Gate")
    else:
        plt.title(f"Logical Error Rate Distribution for \n {n} Multi-Conditioned Gates")
    plt.ylabel("Probability")
    plt.xlabel("Logical Error Rate")
    plt.savefig("mwpf_multi.pdf")
    plt.show()
    
    consec_error_rate, consec_prob = consec_cond_gate(latency, count, trials, n, distance)

    plt.cla()
    plt.loglog(consec_error_rate, consec_prob, ".-")
    plt.xlim(1e0, 1e7)
    plt.ylim(1e-9, 1e0)
    if n == 1:
        plt.title("Logical Error Rate Distribution for \n Single Conditional Gate")
    else:
        plt.title(f"Logical Error Rate Distribution for \n {n} Consecutive Conditional Gates")
    plt.ylabel("Probability")
    plt.xlabel("Logical Error Rate")
    plt.savefig("mwpf_consec.pdf")
    plt.show()