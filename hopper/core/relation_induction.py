import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationInductionLayer(nn.Module):

    def __init__(self, d_model: int, num_relation_types: int = 16):
        super().__init__()
        self.d_model            = d_model
        self.num_relation_types = num_relation_types

        self.W_rel          = nn.Parameter(torch.randn(d_model, d_model) * 0.01)
        self.W_type         = nn.Linear(d_model, num_relation_types, bias=True)
        self.type_embeddings= nn.Embedding(num_relation_types, d_model)
        self.W_out          = nn.Linear(d_model, d_model, bias=False)
        self.layer_norm     = nn.LayerNorm(d_model)

    def forward(self, entities: torch.Tensor):
        B, M, d = entities.shape

        e_left  = entities @ self.W_rel                              # (B, M, d)
        scores  = torch.bmm(e_left, entities.transpose(1, 2))       # (B, M, M)

        # Mask diagonal — float16 safe value
        mask   = torch.eye(M, device=entities.device, dtype=torch.bool)
        mask   = mask.unsqueeze(0).expand(B, -1, -1)
        scores = scores.masked_fill(mask, -1e4)

        attn    = F.softmax(scores / (d ** 0.5), dim=-1)             # (B, M, M)
        context = torch.bmm(attn, entities)                          # (B, M, d)

        type_logits = self.W_type(context)                           # (B, M, K)
        type_dist   = F.softmax(type_logits, dim=-1)                 # (B, M, K)

        type_embs = self.type_embeddings.weight                      # (K, d)
        relations = torch.einsum('bmk,kd->bmd', type_dist, type_embs)

        relations = self.layer_norm(self.W_out(relations))

        return relations, type_dist