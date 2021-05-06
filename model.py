import torch
import torch.nn as nn
import math, copy

class DDQN(nn.Module):
    """
    Description: 
        Deep Q Network with fixed target network implementation written in PyTorch.
    """
    
    def __init__(self, input_dim, output_dim, conv_params, linear_sizes):
        super().__init__()
        
        self.online = DuelingNet(input_dim, output_dim, conv_params, linear_sizes)
        self.online.apply(DuelingNet.init_weights)
        
        self.target = copy.deepcopy(self.online)
        
        # freeze target network parameters
        for p in self.target.parameters():
            p.requires_grad = False
            
    def forward(self, x, version="online"):
        if version == "online":
            return torch.squeeze(self.online(x), -1)
        elif version == "target":
            return torch.squeeze(self.target(x), -1)
        
class DuelingNet(nn.Module):
    """
    Description: 
        Dueling Deep Q Network implementation written in PyTorch.
    """
    
    def __init__(self, input_dim, output_dim, conv_params, linear_sizes):
        super().__init__()
        c, h, w = input_dim
        n_convs = conv_params.shape[0]
        n_linears = len(linear_sizes)
        
        conv_layers = []
        
        ## convolutional layers
        h_in = h
        w_in = w
        c_in = c
        for i in range(n_convs):
            c_out = conv_params[i][0]
            kernel_size = conv_params[i][1]
            stride = conv_params[i][2]
            padding = conv_params[i][3]
            
            h_out = math.floor(((h_in + 2*padding - (kernel_size-1) - 1) / stride) + 1)
            w_out = math.floor(((w_in + 2*padding - (kernel_size-1) - 1) / stride) + 1)
            
            conv_layers.append(nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
            conv_layers.append(nn.ReLU())
            h_in = h_out
            w_in = w_out
            c_in = c_out
        conv_layers.append(nn.Flatten())
        
        self.conv = nn.Sequential(*conv_layers)
        
        ## linear layers
        # value
        value_layers = []
        for i in range(n_linears):
            if i == 0:
                value_layers.append(nn.Linear(c_out * w_out * h_out, linear_sizes[i]))
            else:
                value_layers.append(nn.Linear(linear_sizes[i-1], linear_sizes[i]))
            value_layers.append(nn.ReLU())
        value_layers.append(nn.Linear(linear_sizes[-1], 1))
        
        self.value_fc = nn.Sequential(*value_layers)
        
        # advantage
        adv_layers = []
        for i in range(n_linears):
            if i == 0:
                adv_layers.append(nn.Linear(c_out * w_out * h_out, linear_sizes[i]))
            else:
                adv_layers.append(nn.Linear(linear_sizes[i-1], linear_sizes[i]))
            adv_layers.append(nn.ReLU())
        adv_layers.append(nn.Linear(linear_sizes[-1], output_dim))
        
        self.adv_fc = nn.Sequential(*adv_layers)
            
    def forward(self, x):
        x = x / 255
        x = self.conv(x)
        x_value = self.value_fc(x)
        x_adv = self.adv_fc(x)
        
        return x_value + (x_adv - torch.mean(x_adv, axis=1, keepdim=True))
    
    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            module.bias.data.fill_(0.0)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")