"""
Debug configuration example for EEGMamba
このファイルはデバッグ設定の例を示します
"""

# Basic debug configuration
DEBUG_CONFIG = {
    # Enable/disable debugging
    'debug_mode': True,
    
    # Save tensor outputs to disk for later analysis
    'save_debug_tensors': False,
    
    # Directory to save debug outputs
    'debug_save_dir': 'debug_outputs',
    
    # Gradient checking frequency (every N steps)
    'gradient_check_frequency': 10,
    
    # Memory usage reporting
    'monitor_memory': True,
    
    # Timing analysis
    'enable_timing': True,
}

# Advanced debug configuration for development
ADVANCED_DEBUG_CONFIG = {
    'debug_mode': True,
    'save_debug_tensors': True,
    'debug_save_dir': 'debug_outputs_detailed',
    'gradient_check_frequency': 5,
    'monitor_memory': True,
    'enable_timing': True,
    
    # Additional analysis
    'check_nan_inf': True,
    'log_expert_utilization': True,
    'analyze_attention': True,
}

# Production configuration (minimal debugging)
PRODUCTION_CONFIG = {
    'debug_mode': False,
    'save_debug_tensors': False,
    'gradient_check_frequency': 100,
    'monitor_memory': False,
    'enable_timing': False,
}

def get_debug_config(mode='basic'):
    """
    Get debug configuration for specified mode
    
    Args:
        mode (str): 'basic', 'advanced', or 'production'
    
    Returns:
        dict: Debug configuration
    """
    configs = {
        'basic': DEBUG_CONFIG,
        'advanced': ADVANCED_DEBUG_CONFIG,
        'production': PRODUCTION_CONFIG,
    }
    
    return configs.get(mode, DEBUG_CONFIG)

# Usage example:
# from debug_config import get_debug_config
# debug_cfg = get_debug_config('advanced')
# for key, value in debug_cfg.items():
#     setattr(args, key, value)