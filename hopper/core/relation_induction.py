"""
Relation Induction Layer.
Given entity pairs (e_i, e_j), produces typed relation vectors.

Uses bilinear scoring — proven asymmetric in theoretical verification.
f(e_i, e_j) ≠ f(e_j, e_i) for general W_rel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationInductionLayer(nn.Module):

    def __init__(self, d_model: int, num_relation_types: int = 16):
        """
        d_model           : hidden dimension
        num_relation_types: number of discrete relation type embeddings K
        """
        super().__init__()
        self.d_model            = d_model
        self.num_relation_types = num_relation_types

        # Bilinear scoring matrix
        self.W_rel = nn.Parameter(torch.randn(d_model, d_model) * 0.01)

        # Relation type classifier: maps bilinear score → K type logits
        self.W_type = nn.Linear(d_model, num_relation_types, bias=True)

        # K relation type embeddings
        self.type_embeddings = nn.Embedding(num_relation_types, d_model)

        # Output projection
        self.W_out = nn.Linear(d_model, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, entities: torch.Tensor):
        """
        entities : (B, M, d)

        Returns:
            relations : (B, M, d)   relation vector for each hop slot
                        — relation of hop j = relation between entity j
                          and its most related neighbour
        """
        B, M, d = entities.shape

        # For each slot j, compute relation to all other slots
        # Then aggregate via attention-weighted sum

        # Bilinear scores: (B, M, M)
        # score[b,i,j] = e_i @ W_rel @ e_j
        e_left  = entities @ self.W_rel          # (B, M, d)
        scores  = torch.bmm(e_left, entities.transpose(1, 2))  # (B, M, M)

        # Mask self-relations (diagonal)
        mask = torch.eye(M, device=entities.device).bool().unsqueeze(0)
        scores = scores.masked_fill(mask, -1e9)

        # Attention over neighbours for each slot
        attn = F.softmax(scores / (d ** 0.5), dim=-1)  # (B, M, M)

        # Attended entity context
        context = torch.bmm(attn, entities)  # (B, M, d)

        # Relation type distribution for each slot
        type_logits = self.W_type(context)                    # (B, M, K)
        type_dist   = F.softmax(type_logits, dim=-1)          # (B, M, K)

        # Relation vector: convex combination of type embeddings
        type_embs = self.type_embeddings.weight               # (K, d)
        relations = torch.einsum('bmk,kd->bmd', type_dist, type_embs)  # (B, M, d)

        relations = self.layer_norm(self.W_out(relations))

        return relations, type_dist   # type_dist kept for interpretability