import numpy as np
from scipy import stats

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