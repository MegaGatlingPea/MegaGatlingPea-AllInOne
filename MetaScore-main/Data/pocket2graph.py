import torch
import numpy as np
from torch_geometric.data import Data
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.edges.distance import add_distance_threshold
from functools import partial
import esm
import networkx as nx

# Load ESM-2 model
protein_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
protein_model.eval()

# Configure Graphein
config = ProteinGraphConfig(
    edge_construction_functions=[
        partial(add_distance_threshold, long_interaction_threshold=0, threshold=8)
    ]
)

def get_esm2_embedding(seq, pdb_id):
    # Handle sequences longer than 1022 residues
    seq_feat = []
    for i in range(0, len(seq), 1022):
        data = [(pdb_id, seq[i:i+1022])]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        with torch.no_grad():
            results = protein_model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]
        seq_feat.append(token_representations.squeeze(0)[1:len(data[0][1])+1])
    seq_feat = torch.cat(seq_feat, dim=0)
    
    return seq_feat

def adj2edge_index(adj):
    # Convert adjacency matrix to edge index format
    return adj.nonzero().t().contiguous()

def pocket2graph(pdb_path):
    # Extract pdb_id from pdb_path
    pdb_id = pdb_path.split("/")[-1].split("_")[0]
    
    # Construct graph using Graphein
    try:
        g = construct_graph(config=config, path=pdb_path)
    except Exception as e:
        print(f"Error constructing graph for {pdb_id}: {e}")
        return None
    
    # Extract sequence
    seq = ''.join(g.graph[key] for key in g.graph.keys() if key.startswith("sequence_"))
    
    # Get node features using ESM-2
    node_feat = get_esm2_embedding(seq, pdb_id)
    
    # Build edges
    A = nx.to_numpy_array(g, nonedge=0, weight='distance')
    edge_index = adj2edge_index(torch.tensor(A))
    
    # Add extra structural information
    node_ids = list(g.nodes())
    
    '''
    extra_features = torch.tensor([
        [g.nodes[node_id].get('is_helix', 0),
         g.nodes[node_id].get('is_sheet', 0),
         g.nodes[node_id].get('is_turn', 0),
         g.nodes[node_id].get('solvent_accessibility', 0)]
        for node_id in node_ids
    ])
    
    node_feat = torch.cat([node_feat, extra_features], dim=1)
    '''
    
    # Create Data object
    pos = torch.tensor([g.nodes[node_id]['coords'] for node_id in node_ids])
    data = Data(x=node_feat, edge_index=edge_index, pos=pos)
    
    return data