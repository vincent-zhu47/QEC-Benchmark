import mwpf
import fusion_blossom as fb
import hypernetx as hnx
import matplotlib.pyplot as plt
from math import ceil, log10, floor
import time
import numpy as np

def round_to_even(num: float):
    return ceil(num/2)*2

def hypergraph_to_graph(hypergraph: list[tuple[list[int], float]], num_vertices: int):  
    multi_factor = floor(log10(hypergraph[0][1]))
    
    weighted_edges = []
    for edge in hypergraph:
        if len(edge[0]) == 1:
            weighted_edges.append((edge[0][0], num_vertices, round_to_even(edge[1]*(10**(-1*multi_factor+4)))))
        elif len(edge[0]) == 2:
            weighted_edges.append((edge[0][0], edge[0][1], round_to_even(edge[1]*(10**(-1*multi_factor+4)))))
            
    return weighted_edges

# hypergraph must be a list of edges in the form: ([vertex indices], weight)
# num_vertices: the number of vertices in the hypergraph (aka number of detectors)
# detection_events: list of vertex indices which detected an error
def hypergraph_to_mwpf(hypergraph: list[tuple[list[int], float]], num_vertices: int, detection_events: list):
    multi_factor = floor(log10(hypergraph[0][1]))
    
    weighted_edges = []
    for edge in hypergraph:
        weighted_edges.append(mwpf.HyperEdge(edge[0], round(edge[1]*(10**(-1*multi_factor+4)))))

    initializer = mwpf.SolverInitializer(num_vertices, weighted_edges)
    hyperion = mwpf.SolverSerialJointSingleHair(initializer)

    start = time.time()
    print("start")
    hyperion.solve(mwpf.SyndromePattern(detection_events)) 
    end = time.time()
    print("end")
    latency = end - start

    hyperion_subgraph = hyperion.subgraph()
    
    return hyperion, hyperion_subgraph, latency

def hypergraph_to_fb(hypergraph: list[tuple[list[int], float]], num_vertices: int, detection_events: list):
    weighted_edges = hypergraph_to_graph(hypergraph, num_vertices)
            
    initializer = fb.SolverInitializer(num_vertices + 1, weighted_edges, [num_vertices])
    fusion = fb.SolverSerial(initializer)

    start = time.time()
    fusion.solve(fb.SyndromePattern(detection_events))
    end = time.time()
    
    latency = end-start
    
    fusion_subgraph = fusion.subgraph()
    
    total_weight = 0
    for x in fusion_subgraph:
        total_weight+=weighted_edges[x][2]
    
    return fusion_subgraph, total_weight, latency

def hypergraph_to_bposd():
    return

def visualize_hypergraph(hypergraph: list[tuple[list[int], float]]):
    only_vertices = []
    
    for edge in hypergraph:
        only_vertices.append(edge[0])
    
    H = hnx.Hypergraph(only_vertices)
    
    hnx.draw(H)
    plt.show()
    
def visualize_graph(graph: list[list[int]]):
    only_vertices = []
    
    for edge in graph:
        only_vertices.append(edge[0:2])
        
    H = hnx.Hypergraph(only_vertices)
    
    hnx.draw(H)
    plt.show()