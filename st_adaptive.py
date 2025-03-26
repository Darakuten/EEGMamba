import torch
from torch import nn

class SpatialAdaptiveConvolution(nn.Module):
  def __init__(self, input_channels, hidden_dim):
    super().__init__()
    r"""
    input: 
            X = (B, C_i, L_i)
    output: 
            y_SA = (B, D, L_i)
    """

    self.conv = nn.Conv1d(input_channels, hidden_dim, kernel_size=3, stride=1, padding=1)
  
  def forward(self, x):
    y_SA = self.conv(x)
    return y_SA


class Tokenizer(nn.Module):
  def __init__(self, input_channels):
    super().__init__()
    r"""
    input: 
            y_SA = (B, D, L_i)
    output: 
            T = (B, N, D)
    """
    self.conv1_s = nn.Sequential(
            nn.Conv1d( input_channels, input_channels, kernel_size=15, stride=2 ),
            nn.GELU(),
            nn.MaxPool1d(7, stride=2),
            nn.Dropout(0.5),
        )
  
    self.conv2_s = nn.Sequential(
        nn.Conv1d( input_channels, input_channels, kernel_size=7, stride=1 ),
        nn.GELU(),
      )
    
    self.conv3_s = nn.Sequential(
        nn.Conv1d( input_channels, input_channels, kernel_size=7, stride=1 ),
        nn.GELU(),
        nn.MaxPool1d(7, stride=2),
      )

    self.conv1_w = nn.Sequential(
            nn.Conv1d( input_channels, input_channels, kernel_size=49, stride=8 ),
            nn.GELU(),
            nn.MaxPool1d(3, stride=2),
            nn.Dropout(0.5),
        )
  
    self.conv2_w = nn.Sequential(
        nn.Conv1d( input_channels, input_channels, kernel_size=7, stride=1 ),
        nn.GELU(),
      )
    
    self.conv3_w = nn.Sequential(
        nn.Conv1d( input_channels, input_channels, kernel_size=7, stride=1 ),
        nn.GELU(),
        nn.MaxPool1d(3, stride=2),
      )
    
    self.cls_token = nn.Parameter(torch.randn(1, 1, input_channels))
  
  def forward(self, x):
    B = x.size(0)

    # get small token sequence.
    z_s = self.conv1_s(x) 
    z_s = self.conv2_s(z_s)
    z_s = self.conv3_s(z_s)

    # get wide token sequence.
    z_w = self.conv1_w(x)
    z_w = self.conv2_w(z_w)
    z_w = self.conv3_w(z_w)

    cls_token = self.cls_token.expand(B, -1, -1)

    T = torch.cat([cls_token, z_s.transpose(1, 2), z_w.transpose(1, 2)], dim=1)

    return T

class STAdaptive(nn.Module):
  def __init__(self, input_channels, hidden_dim):
    super().__init__()
    self.conv = SpatialAdaptiveConvolution(input_channels, hidden_dim)
    self.tokenizer = Tokenizer(hidden_dim)
  
  def forward(self, x):
    y_SA = self.conv(x) #(B, C, L) --> (B, D, L)
    T = self.tokenizer(y_SA) # (B, D, L) --> (B, N+1, D)
    return T