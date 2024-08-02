import stim
import qecp
            
def stim_to_hypergraph(dem: stim.DetectorErrorModel):
    hyperedges = []
    
    for instruction in dem:
        if instruction.type == 'error':
            dets = []
            p = instruction.args_copy()[0]
            for t in instruction.targets_copy():
                if t.is_relative_detector_id():
                    dets.append(t.val)
            hyperedges.append((dets, p))
                    
    return hyperedges, dem.num_detectors

def qecp_to_hypergraph(simulator: qecp.Simulator, noise_model: qecp.NoiseModel):
    hypergraph = qecp.create_hypergraph(simulator, noise_model)
    hyperedges = []

    for w_e in hypergraph.weighted_edges:
        hyperedges.append(([hypergraph.vertex_index(v) for v in w_e[0].vertices], w_e[1].hyperedge.weight))
    
    return hyperedges, len(hypergraph.vertex_indices)

def ldpc_to_hypergraph():
    
    return