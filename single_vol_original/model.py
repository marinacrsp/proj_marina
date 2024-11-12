import math

import numpy as np
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self, coord_dim=4, hidden_dim=512, n_layers=8, out_dim=2, L=10, gamma=1.0
    ) -> None:
        super().__init__()
        self.L = L

        # Precompute the scaling factors for the coordinate encoding.
        L_mult = torch.pow(2, torch.arange(self.L)) * math.pi
        self.register_buffer("L_mult", L_mult)
        fourier_dim = self.L * 2 * coord_dim

        self.linear_layers = [nn.Linear(fourier_dim, hidden_dim)]
        self.linear_layers.extend(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 1)]
        )
        self.linear_layers = nn.ModuleList(self.linear_layers)

        self.output_layer = nn.Linear(hidden_dim, out_dim)
        self.activation = nn.Tanh()
        self.gamma = gamma

    def forward(self, coords):
        # Coordinate Encoding
        x = coords.unsqueeze(-1) * self.L_mult
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        x = x.view(x.size(0), -1)

        # Forward pass
        for layer in self.linear_layers:
            x = layer(x)
            x = self.activation(x * self.gamma)

        return self.output_layer(x)



# class Siren(nn.Module):
#     def __init__(
#         self,
#         coord_dim=4,
#         hidden_dim=512,
#         n_layers=8,
#         out_dim=2,
#         omega_0=30,
#         L=10,
#         dropout_rate=0.20,
#     ) -> None:
#         super().__init__()
#         self.L = L

#         # Precompute the scaling factors for the coordinate encoding.
#         L_mult = torch.pow(2, torch.arange(self.L)) * math.pi
#         self.register_buffer("L_mult", L_mult)
#         fourier_dim = self.L * 2 * coord_dim

#         self.sine_layers = [
#             SineLayer(fourier_dim, hidden_dim, is_first=True, omega_0=omega_0)
#         ]
        
#         # NOTE : Introducing residual connection
#         for layer_idx in range(n_layers-1):
#             if layer_idx == n_layers//2 - 1:
#                 self.res_connection = layer_idx + 1 
#                 self.sine_layers.append(
#                 SineLayer(hidden_dim + fourier_dim, hidden_dim, is_first=False, omega_0=omega_0)
#             )
#             else:
#                 self.sine_layers.append(
#                 SineLayer(hidden_dim, hidden_dim, is_first=False, omega_0=omega_0)
#             )
#         self.sine_layers = nn.ModuleList(self.sine_layers)

#         self.output_layer = nn.Linear(hidden_dim, out_dim)
#         with torch.no_grad():
#             self.output_layer.weight.uniform_(
#                 -np.sqrt(6 / hidden_dim) / omega_0, np.sqrt(6 / hidden_dim) / omega_0
#             )

#     def forward(self, coords):
#         x = coords.unsqueeze(-1) * self.L_mult
#         x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
#         x = x.view(x.size(0), -1)
#         x0 = x.clone()

#         for layer_idx, layer in enumerate(self.sine_layers):
#             if layer_idx == self.res_connection:
#                 x = torch.cat([x0, x], dim=-1)
#             x = layer(x)

#         return self.output_layer(x)

class Siren(nn.Module):
    def __init__(
        self,
        coord_dim=4,
        hidden_dim=512,
        n_layers=8,
        out_dim=2,
        omega_0=30,
        L=10,
        dropout_rate=0.20,
    ) -> None:
        super().__init__()
        self.L = L

        # Precompute the scaling factors for the coordinate encoding.
        L_mult = torch.pow(2, torch.arange(self.L)) * math.pi
        self.register_buffer("L_mult", L_mult)
        fourier_dim = self.L * 2 * coord_dim

        self.sine_layers = [SineLayer(fourier_dim, hidden_dim, is_first=True, omega_0=omega_0)]
        
        # NOTE : Introducing residual connection
        for _ in range(n_layers-1):
            self.sine_layers.append(
                SineLayer(hidden_dim, hidden_dim, is_first=False, omega_0=omega_0))
        #
        
        self.output_layer = nn.Linear(hidden_dim, out_dim)
        with torch.no_grad():
            self.output_layer.weight.uniform_(
                -np.sqrt(6 / hidden_dim) / omega_0, np.sqrt(6 / hidden_dim) / omega_0
            )

    def forward(self, coords):
        x = coords.unsqueeze(-1) * self.L_mult
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        x = x.view(x.size(0), -1)
        
        for layer_idx, layer in enumerate(self.sine_layers):
            x = layer(x)
        return self.output_layer(x)
        

# NOTE: Siren Network, using Fourier Features (instead of Positional Encoding).
# class Siren(nn.Module):
#     def __init__(self, coord_dim=4, fourier_dim=512, hidden_dim=512, n_layers=8, out_dim=2, omega_0=30) -> None:
#         super().__init__()
#         # Random Gaussian matrix used in the computation of Fourier features.
#         B = torch.randn((coord_dim, fourier_dim//2), dtype=torch.float32)
#         self.register_buffer('B', B)

#         self.net = [SineLayer(fourier_dim, hidden_dim, is_first=True, omega_0=omega_0)]
#         for i in range(n_layers-1):
#             self.net.append(SineLayer(hidden_dim, hidden_dim, is_first=False, omega_0=omega_0))
#         final_linear = nn.Linear(hidden_dim, out_dim)

#         with torch.no_grad():
#             final_linear.weight.uniform_(-np.sqrt(6 / hidden_dim) / omega_0,
#                                           np.sqrt(6 / hidden_dim) / omega_0)
#         self.net.append(final_linear)
#         self.net = nn.Sequential(*self.net)

#     def forward(self, coords):
#         fourier_features = torch.cat([torch.sin(coords @ self.B), torch.cos(coords @ self.B)], dim=-1)

#         return self.net(fourier_features)


class SineLayer(nn.Module):
    """Linear layer with sine activation. Adapted from Siren repo"""

    def __init__(
        self, in_features, out_features, bias=True, is_first=False, omega_0=30
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        # self.linear = nn.utils.weight_norm(nn.Linear(in_features, out_features, bias=bias))

        # self.layer_norm = nn.LayerNorm(out_features)
        # self.batch_norm = nn.BatchNorm1d(out_features)

        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, x):
        # NOTE: Uncomment when using batch (or layer) normalization.
        # x = self.linear(x)
        # x = self.layer_norm(x)
        # x = self.batch_norm(x)
        # return torch.sin(self.omega_0 * x)

        return torch.sin(self.omega_0 * self.linear(x))
