import torch
import torch.nn as nn
import torch.nn.functional as F


class EntityInductionLayer(nn.Module):

    def __init__(self, d_model: int, num_slots: int, tau: float = 0.5):
        super().__init__()
        self.d_model   = d_model
        self.num_slots = num_slots
        self.tau       = tau

        self.W_r        = nn.Linear(d_model, num_slots, bias=False)
        self.W_e        = nn.Linear(d_model, d_model,   bias=False)
        self.layer_norm = nn.LayerNorm(d_model)

        nn.init.xavier_uniform_(self.W_r.weight)
        nn.init.xavier_uniform_(self.W_e.weight)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None):
        B, T, d = x.shape

        routing_logits = self.W_r(x)   # (B, T, M)

        # Mask padding — use -1e4 not -1e9 (float16 safe)
        if attention_mask is not None:
            routing_logits = routing_logits.masked_fill(
                attention_mask.unsqueeze(-1) == 0, -1e4
            )

        if self.training:
            gumbel = -torch.log(
                -torch.log(torch.rand_like(routing_logits) + 1e-10) + 1e-10
            )
            routing = F.softmax(
                (routing_logits + gumbel) / self.tau, dim=1
            )
        else:
            routing = F.softmax(routing_logits / self.tau, dim=1)

        x_proj   = self.W_e(x)
        entities = torch.bmm(routing.transpose(1, 2), x_proj)  # (B, M, d)
        entities = self.layer_norm(entities)

        return entities, routing

    def set_temperature(self, tau: float):
        self.tau = tau