import math
from hash_encoding_batch import *
import numpy as np
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self, coord_dim=4, hidden_dim=256, n_layers=4, out_dim=2, L=10, gamma=1.0
    ) -> None:
        super().__init__()
        self.L = L

        # Precompute the scaling factors for the coordinate encoding.
        L_mult = torch.pow(2, torch.arange(self.L)) * math.pi
        self.register_buffer("L_mult", L_mult)
        fourier_dim = self.L * 2 * coord_dim

        # fourier_dim = 84
        # hidden dim = 256
        self.firstlinear_layers = [nn.Linear(fourier_dim, hidden_dim)]
        self.firstlinear_layers.extend(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 1)]
        )
        self.firstlinear_layers = nn.ModuleList(self.firstlinear_layers)
        
        self.secondset_layers = [nn.Linear(fourier_dim+hidden_dim, hidden_dim)]
        self.secondset_layers.extend([[nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 2)]])
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


# class coor_embedding(nn.Module):
#     def __init__(
#         self,
#     )->None:
#         super().__init__()      
        
#         self.embedd_fn = hash_encoder(levels=10, log2_hashmap_size=12, n_features_per_level=2, n_max=320, n_min=16)
        
#         # self.x_embedding = nn.Embedding(num_x_coords, embedding_dim, padding_idx=0)
#         # self.y_embedding = nn.Embedding(num_y_coords, embedding_dim, padding_idx=0)
        
#     def forward(self, coors_kspace):
#         # kx_embedded = self.x_embedding(coors_kspace[:,0].long()) 
#         # ky_embedded = self.y_embedding(coors_kspace[:,1].long()) 
#         # coord_features = torch.cat((torch.cat((kx_embedded, ky_embedded), dim=1), coors_kspace[:,2:]), dim = 1 )
#         coord_features = self.embedd_fn(coors_kspace)

#         return coord_features
    

class Siren_skip_emb(nn.Module):
    def __init__(
        self,
        hidden_dim=512,
        levels = 10,
        n_layers=4,
        out_dim=2,
        omega_0=30,
        dropout_rate=0.20,
        device=None,
    ) -> None:
        super().__init__()        
        self.n_flayer = n_layers // 2
        self.n_slayer = n_layers - self.n_flayer
                
        # Layer containing trainable parameters for the embedding
        
        self.embed_fn = hash_encoder(levels=levels, log2_hashmap_size=12, n_features_per_level=2, n_max=320, n_min=16)
        
        coor_embedd_dim = levels*2 + 2
                
        # First set of layers (before the first skip connection)
        self.firstlayers = nn.ModuleList([SineLayer(coor_embedd_dim, hidden_dim, is_first=True, omega_0=omega_0)])
        for _ in range(self.n_flayer-1):
            self.firstlayers.append(SineLayer(hidden_dim, hidden_dim, is_first=False, omega_0=omega_0))
        
        # Second set of layers (after the first skip connection)
        self.secondlayers = nn.ModuleList([SineLayer(coor_embedd_dim + hidden_dim, hidden_dim, is_first=False, omega_0=omega_0)])
        for _ in range(self.n_slayer-1):
            self.secondlayers.append(SineLayer(hidden_dim, hidden_dim, is_first=False, omega_0=omega_0))

        self.output_layer = SineLayer(hidden_dim, out_dim, is_first=False, omega_0=omega_0)
        
    def forward(self, coords):
        
        # Coordinate encoding (Fourier)
        h0 = self.embed_fn(coords)
        
        # First set of layers
        h1 = h0.clone()
        
        for layer in self.firstlayers:
            h1 = layer(h1)
        
        # First skip connection
        h2 = torch.cat([h1, h0], dim=-1)
        
        # Second set of layers
        for layer in self.secondlayers:
            h2 = layer(h2)
        
        out_x = self.output_layer(h2)
        return out_x


class Siren_skip(nn.Module):
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
        
        self.n_flayer = n_layers // 2
        self.n_slayer = n_layers - self.n_flayer
        
        # self.n_slayer = self.n_flayer + 2 * n_layers // 3
        
        # Precompute the scaling factors for the coordinate encoding.
        L_mult = torch.pow(2, torch.arange(self.L)) * math.pi
        self.register_buffer("L_mult", L_mult)
        fourier_dim = self.L * 2 * coord_dim + coord_dim
        # fourier_dim = self.L * 2 * coord_dim
        
        # First set of layers (before the first skip connection)
        self.firstlayers = nn.ModuleList([SineLayer(fourier_dim, hidden_dim, is_first=True, omega_0=omega_0)])
        for _ in range(self.n_flayer-1):
            self.firstlayers.append(SineLayer(hidden_dim, hidden_dim, is_first=False, omega_0=omega_0))
        
        # Second set of layers (after the first skip connection)
        self.secondlayers = nn.ModuleList([SineLayer(fourier_dim + hidden_dim, hidden_dim, is_first=False, omega_0=omega_0)])
        for _ in range(self.n_slayer-1):
            self.secondlayers.append(SineLayer(hidden_dim, hidden_dim, is_first=False, omega_0=omega_0))
        
        # self.output_layer = nn.Linear(hidden_dim, out_dim)
        # Don't keep track of gradients here, just initialization of weights 
        # with torch.no_grad():
        #     self.output_layer.weight.uniform_(
        #         -np.sqrt(6 / hidden_dim) / omega_0, np.sqrt(6 / hidden_dim) / omega_0
        #     )
        self.output_layer = SineLayer(hidden_dim, out_dim, is_first=False, omega_0=omega_0)

    def forward(self, coords):
        
        # Coordinate encoding (Fourier)
        h0 = coords.unsqueeze(-1) * self.L_mult
        h0 = torch.cat([torch.sin(h0), torch.cos(h0)], dim=-1)
        h0 = h0.view(h0.size(0), -1)
        # Concatenate encoding with original coordinates
        h0 = torch.cat([h0, coords], dim=-1)
        
        # First set of layers
        h1 = h0
        for layer in self.firstlayers:
            h1 = layer(h1)
        
        # First skip connection
        h2 = torch.cat([h1, h0], dim=-1)
        
        # Second set of layers
        for layer in self.secondlayers:
            h2 = layer(h2)
        
        out_x = self.output_layer(h2)
        return out_x

class Siren(nn.Module):
    def __init__(
        self,
        coord_dim=4,
        hidden_dim=512,
        n_layers=10,
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

        # Initialization, no track of gradients here
        self.sine_layers = [
            SineLayer(fourier_dim, hidden_dim, is_first=True, omega_0=omega_0)
        ]
        for _ in range(n_layers - 1):
            self.sine_layers.append(
                SineLayer(hidden_dim, hidden_dim, is_first=False, omega_0=omega_0)
            )
        self.sine_layers = nn.ModuleList(self.sine_layers)
        self.output_layer = nn.Linear(hidden_dim, out_dim)
        
        # Don't keep track of gradients here, just initialization of weights 
        with torch.no_grad():
            self.output_layer.weight.uniform_(
                -np.sqrt(6 / hidden_dim) / omega_0, np.sqrt(6 / hidden_dim) / omega_0
            )

        # self.dropout = nn.Dropout(dropout_rate)

    def forward(self, coords):
        x = coords.unsqueeze(-1) * self.L_mult
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        x = x.view(x.size(0), -1)
        x = torch.cat([x], dim=-1)

        for layer in self.sine_layers:
            x = layer(x)
            # x = self.dropout(x)

        return self.output_layer(x)
    
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

        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
                self.linear.weight *= self.omega_0
                
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
        # layer_out = torch.sin(self.omega_0 * self.linear(x))
        layer_out = torch.sin(self.linear(x))

        return layer_out

                
