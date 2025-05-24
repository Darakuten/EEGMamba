import torch
import torch.nn as nn
import numpy as np
import time
from functools import wraps
import os
from termcolor import colored

class EEGMambaDebugger:
    def __init__(self, enable_debug=True, save_tensors=False, save_dir="debug_outputs"):
        self.enable_debug = enable_debug
        self.save_tensors = save_tensors
        self.save_dir = save_dir
        self.step_count = 0
        self.timing_stats = {}
        self.tensor_stats = {}
        
        if self.save_tensors and not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def debug_print(self, message, color='cyan'):
        if self.enable_debug:
            print(colored(f"[DEBUG] {message}", color))
    
    def log_tensor_stats(self, tensor, name, step=None):
        if not self.enable_debug:
            return
        
        if step is None:
            step = self.step_count
            
        with torch.no_grad():
            stats = {
                'shape': tuple(tensor.shape),
                'dtype': str(tensor.dtype),
                'device': str(tensor.device),
                'min': float(tensor.min()),
                'max': float(tensor.max()),
                'mean': float(tensor.mean()),
                'std': float(tensor.std()),
                'norm': float(torch.norm(tensor)),
                'nan_count': int(torch.isnan(tensor).sum()),
                'inf_count': int(torch.isinf(tensor).sum())
            }
            
            key = f"{name}_step_{step}"
            self.tensor_stats[key] = stats
            
            self.debug_print(f"{name}: shape={stats['shape']}, mean={stats['mean']:.4f}, std={stats['std']:.4f}, norm={stats['norm']:.4f}")
            
            if stats['nan_count'] > 0:
                self.debug_print(f"WARNING: {name} contains {stats['nan_count']} NaN values!", 'red')
            if stats['inf_count'] > 0:
                self.debug_print(f"WARNING: {name} contains {stats['inf_count']} Inf values!", 'red')
                
            if self.save_tensors:
                save_path = os.path.join(self.save_dir, f"{key}.pt")
                torch.save(tensor.cpu(), save_path)
    
    def timing_decorator(self, func_name):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enable_debug:
                    return func(*args, **kwargs)
                
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                
                duration = end_time - start_time
                if func_name not in self.timing_stats:
                    self.timing_stats[func_name] = []
                self.timing_stats[func_name].append(duration)
                
                self.debug_print(f"{func_name} took {duration:.4f}s", 'yellow')
                return result
            return wrapper
        return decorator
    
    def check_gradients(self, model, threshold=1e-6):
        if not self.enable_debug:
            return
        
        self.debug_print("=== Gradient Check ===", 'magenta')
        grad_norms = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms[name] = grad_norm
                
                if grad_norm < threshold:
                    self.debug_print(f"Small gradient: {name} = {grad_norm:.8f}", 'yellow')
                elif grad_norm > 1.0:
                    self.debug_print(f"Large gradient: {name} = {grad_norm:.4f}", 'red')
                else:
                    self.debug_print(f"Normal gradient: {name} = {grad_norm:.6f}", 'green')
        
        total_grad_norm = torch.sqrt(sum(g**2 for g in grad_norms.values()))
        self.debug_print(f"Total gradient norm: {total_grad_norm:.6f}", 'cyan')
        
        return grad_norms
    
    def visualize_attention_weights(self, attention_weights, name="attention"):
        if not self.enable_debug or attention_weights is None:
            return
        
        self.debug_print(f"=== {name} Analysis ===", 'blue')
        with torch.no_grad():
            if len(attention_weights.shape) >= 3:
                # Batch dimension exists
                avg_attention = attention_weights.mean(dim=0)
                self.debug_print(f"Average attention shape: {avg_attention.shape}")
                self.debug_print(f"Attention entropy: {self._compute_entropy(avg_attention):.4f}")
            
    def _compute_entropy(self, weights):
        # Compute entropy of attention weights
        weights = weights.flatten()
        weights = weights / weights.sum()
        entropy = -(weights * torch.log(weights + 1e-10)).sum()
        return entropy.item()
    
    def memory_usage(self):
        if not self.enable_debug or not torch.cuda.is_available():
            return
        
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3  # GB
        self.debug_print(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB", 'yellow')
    
    def step(self):
        self.step_count += 1
        self.memory_usage()
    
    def summary(self):
        if not self.enable_debug:
            return
        
        self.debug_print("=== DEBUG SUMMARY ===", 'magenta')
        
        # Timing summary
        if self.timing_stats:
            self.debug_print("Timing Statistics:", 'cyan')
            for func_name, times in self.timing_stats.items():
                avg_time = np.mean(times)
                total_time = np.sum(times)
                count = len(times)
                self.debug_print(f"  {func_name}: avg={avg_time:.4f}s, total={total_time:.4f}s, count={count}")
        
        # Tensor statistics summary
        if self.tensor_stats:
            self.debug_print("Tensor Statistics:", 'cyan')
            for name, stats in list(self.tensor_stats.items())[-5:]:  # Show last 5
                self.debug_print(f"  {name}: shape={stats['shape']}, norm={stats['norm']:.4f}")
        
        self.debug_print(f"Total steps: {self.step_count}", 'green')

# Global debugger instance
debugger = EEGMambaDebugger()

def enable_debug(save_tensors=False, save_dir="debug_outputs"):
    global debugger
    debugger = EEGMambaDebugger(enable_debug=True, save_tensors=save_tensors, save_dir=save_dir)

def disable_debug():
    global debugger
    debugger = EEGMambaDebugger(enable_debug=False)

def debug_tensor(tensor, name, step=None):
    debugger.log_tensor_stats(tensor, name, step)

def debug_print(message, color='cyan'):
    debugger.debug_print(message, color)

def check_gradients(model):
    return debugger.check_gradients(model)

def debug_step():
    debugger.step()

def debug_summary():
    debugger.summary()

def timing(func_name):
    return debugger.timing_decorator(func_name)