import torch
import torch.nn as nn

class InteractionModule(nn.Module):
    def __init__(self, protein_dim, ligand_dim, hidden_dim):
        super(InteractionModule, self).__init__()
        self.protein_proj = nn.Linear(protein_dim, hidden_dim)
        self.ligand_proj = nn.Linear(ligand_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, protein_repr, ligand_repr):
        protein_proj = self.protein_proj(protein_repr).unsqueeze(0)
        ligand_proj = self.ligand_proj(ligand_repr).unsqueeze(0)
        
        attn_output, _ = self.attention(protein_proj, ligand_proj, ligand_proj)
        
        combined = torch.cat([attn_output.squeeze(0), protein_repr], dim=-1)
        return self.mlp(combined)