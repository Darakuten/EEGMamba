import torch
import torch.nn as nn
from st_adaptive import STAdaptive
from bi_mammba import BidirectionalMamba
from matlab_utils.load_meg import roi
from moe import MoE
from debug_utils import debug_tensor, debug_print, timing

class EEGMamba(nn.Module):
    def __init__(self, args):
        super().__init__()
        r"""
        Input:
            x: (B, C_i, L_i)
              C_i: input channels
        Middle:
            y_SA: (B, D, L_i)
              D: hidden dim
            T: (B, N+1, D)
            T_k: (B, N+1, D)
            T^*: (B, N+1, D)
        Output:
            output: (B, C_o, L_o)
        """

        roi_channels = roi(args)
        num_channels = len(roi_channels)
        
        self.st_adaptive = STAdaptive(num_channels, args.D)
        
        self.blocks = nn.ModuleList([
            nn.Sequential(
                BidirectionalMamba(args.D),  # Mamba による系列変換
                MoE(args.num_experts, args.D, args.F, args.topk)  # Mixture of Experts
            )
            for _ in range(args.num_blocks)
        ])

        self.classifier = nn.Linear(args.D, args.num_class, bias=True)
      
    @timing("forward_pass")
    def forward(self, x):
        debug_print("=== EEGMamba Forward Pass ===", 'blue')
        debug_tensor(x, "input", step=0)
        
        # Spatial-Temporal Adaptive Processing
        T = self.st_adaptive(x)
        debug_tensor(T, "st_adaptive_output", step=1)
        debug_print(f"ST Adaptive output shape: {T.shape}", 'green')

        # Process through Mamba blocks with MoE
        for i, block in enumerate(self.blocks):
            T_prev = T.clone()
            T = block(T)
            debug_tensor(T, f"mamba_block_{i}_output", step=i+2)
            
            # Check for gradient flow issues
            if torch.allclose(T, T_prev, atol=1e-6):
                debug_print(f"WARNING: Block {i} output unchanged from input!", 'red')
                
        debug_print(f"Final Mamba blocks output shape: {T.shape}", 'green')

        # Extract class token
        class_logits = T[:, 0, :]
        debug_tensor(class_logits, "class_logits", step=len(self.blocks)+2)
        debug_print(f"Class logits shape: {class_logits.shape}", 'green')

        # Final classification
        output = self.classifier(class_logits)
        debug_tensor(output, "final_output", step=len(self.blocks)+3)
        debug_print(f"Final output shape: {output.shape}", 'green')
        
        # Check for NaN or Inf in output
        if torch.isnan(output).any():
            debug_print("ERROR: NaN detected in final output!", 'red')
        if torch.isinf(output).any():
            debug_print("ERROR: Inf detected in final output!", 'red')

        return output