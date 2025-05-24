#!/usr/bin/env python3
"""
Test script to demonstrate EEGMamba debugging functionality
EEGMambaのデバッグ機能をテストするスクリプト
"""

import torch
import torch.nn as nn
from debug_utils import (enable_debug, disable_debug, debug_print, 
                        debug_tensor, check_gradients, debug_summary, timing)
from debug_config import get_debug_config

# Simple test model to demonstrate debugging
class SimpleTestModel(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, output_dim=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    @timing("model_forward")
    def forward(self, x):
        debug_tensor(x, "input")
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            debug_tensor(x, f"layer_{i}_output")
            
        return x

def test_debugging_functionality():
    """Test the debugging functionality"""
    
    debug_print("=== EEGMamba Debug Test ===", 'blue')
    
    # Enable debugging with advanced configuration
    debug_cfg = get_debug_config('advanced')
    enable_debug(save_tensors=debug_cfg['save_debug_tensors'], 
                save_dir=debug_cfg['debug_save_dir'])
    
    # Create test model and data
    model = SimpleTestModel()
    batch_size = 4
    input_dim = 128
    
    debug_print(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    x = torch.randn(batch_size, input_dim)
    debug_print("Running forward pass...", 'green')
    
    output = model(x)
    debug_tensor(output, "final_output")
    
    # Test backward pass and gradient checking
    target = torch.randint(0, 10, (batch_size,))
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(output, target)
    
    debug_print("Running backward pass...", 'green')
    loss.backward()
    
    # Check gradients
    grad_norms = check_gradients(model)
    
    # Test with problematic scenarios
    debug_print("Testing NaN detection...", 'yellow')
    nan_tensor = torch.tensor([1.0, float('nan'), 3.0])
    debug_tensor(nan_tensor, "nan_test_tensor")
    
    debug_print("Testing Inf detection...", 'yellow')
    inf_tensor = torch.tensor([1.0, float('inf'), 3.0])
    debug_tensor(inf_tensor, "inf_test_tensor")
    
    # Print summary
    debug_summary()
    
    debug_print("Debug test completed!", 'green')

if __name__ == "__main__":
    test_debugging_functionality()