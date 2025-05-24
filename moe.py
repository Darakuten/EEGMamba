import torch
import torch.nn as nn
import torch.nn.functional as F
from debug_utils import debug_tensor, debug_print, timing

class MoE(nn.Module):
    def __init__(self, num_experts, input_dim, expert_hidden_size, k=2):
        """
        Initialize the MoE module.
        Args:
            num_experts (int): Number of experts.
            input_dim (int): Input dimension.
            expert_hidden_size (int): Expert hidden size
            k (int): Number of top-k experts to activate.
        """
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.k = k

        # Define experts
        expert = nn.Sequential(
                nn.Linear(input_dim, expert_hidden_size),
                nn.ReLU(),
                nn.Linear(expert_hidden_size, input_dim)
            )
        self.experts = nn.ModuleList([expert for _ in range(num_experts)])
        self.universal_expert = expert

        # Gating network
        self.gating_network = nn.Linear(input_dim, num_experts)

    @timing("moe_forward")
    def forward(self, x):
        r"""
        Input: 
            x: (B, N, D)
        Output:
            output: (B, N, D)
        """
        B, N, D = x.shape
        debug_tensor(x, "moe_input")
        
        x = x.reshape(B * N, D)
        # Compute gating values
        gating_logits = self.gating_network(x)  # Shape: (batch_size, num_experts)
        debug_tensor(gating_logits, "moe_gating_logits")

        # Apply Top-k masking
        topk_values, topk_indices = torch.topk(gating_logits, self.k, dim=-1)
        mask = torch.full_like(gating_logits, -float('inf'))
        mask.scatter_(1, topk_indices, topk_values)

        # Compute SoftMax over masked values
        gating_values = F.softmax(mask, dim=-1)  # Shape: (batch_size, num_experts)
        debug_tensor(gating_values, "moe_gating_weights")
        
        # Log expert utilization
        expert_usage = (gating_values > 0.01).float().mean(dim=0)
        debug_print(f"Expert utilization: {expert_usage.tolist()}", 'yellow')

        # Compute expert outputs
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # Shape: (batch_size, num_experts, output_dim)
        debug_tensor(expert_outputs, "moe_expert_outputs")

        max_gate_value = torch.max(gating_values, dim=1, keepdim=True)[0]
        omega = 1 - max_gate_value
        universal_expert_output = omega * self.universal_expert(x)
        debug_tensor(universal_expert_output, "moe_universal_expert_output")

        # Weighted sum of expert outputs
        output = torch.sum(gating_values.unsqueeze(-1) * expert_outputs, dim=1)  # Shape: (batch_size, output_dim)
        output += universal_expert_output
        debug_tensor(output, "moe_combined_output")

        output = output.reshape(B, N, D)
        debug_tensor(output, "moe_final_output")

        return output
