import torch
import torch.nn as nn
from st_adaptive import STAdaptive
from bi_mammba import BidirectionalMamba
from matlab_utils.load_meg import roi
from moe import MoE

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
      
    def forward(self, x):
        
        T = self.st_adaptive(x)
        print("ST Adaptive: ", T.shape)

        for block in self.blocks:
            T = block(T)       
        print("Mamba Block: ", T.shape)

        class_logits = T[:, 0, :]
        print("class_logits: ", class_logits.shape)

        output = self.classifier(class_logits)
        print("classifier: ", output.shape)

        return output