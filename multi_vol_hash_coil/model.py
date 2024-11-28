import math
from hash_encoding_batch import *
import numpy as np
import torch
from torch import nn


class Siren(nn.Module):
    def __init__(
        self,
        coord_dim=3,
        levels = 10,
        n_min = 16,
        size_hashtable = 12,
        vol_embedding_dim=256,
        coil_embedding_dim = 128,
        hidden_dim=512,
        n_layers=4,
        n_features=3,
        out_dim=2,
        omega_0=30,
        dropout_rate=0.20,
    ) -> None:
        super().__init__()
        
        self.n_flayer = n_layers // 2
        self.n_slayer = n_layers - self.n_flayer
        self.embed_fn = hash_encoder(levels=levels, log2_hashmap_size=size_hashtable, n_features_per_level=n_features, n_max=320, n_min=n_min)
        coord_encoding_dim = levels*n_features + coord_dim-2 # NOTE: We need to append the kz and coilID coordinates 

        self.sine_layers = [
            SineLayer(
                coord_encoding_dim + vol_embedding_dim + coil_embedding_dim,
                hidden_dim,
                is_first=True,
                omega_0=omega_0,
            )
        ]
        
        for layer_idx in range(n_layers - 1):
            # We have a residual connection at this layer (hence the different input dimension).
            if layer_idx == n_layers // 2 - 1:
                self.res_layer_idx = layer_idx + 1
                self.sine_layers.append(
                    SineLayer(hidden_dim + coord_encoding_dim + vol_embedding_dim + coil_embedding_dim, hidden_dim, is_first=False, omega_0=omega_0))
            else:
                self.sine_layers.append(
                    SineLayer(hidden_dim, hidden_dim, is_first=False, omega_0=omega_0)
                )
                # self.sine_layers.append(nn.LayerNorm(hidden_dim))
                # self.sine_layers.append(nn.BatchNorm1d(hidden_dim))
        self.sine_layers = nn.ModuleList(self.sine_layers)
        
        self.output_layer = nn.Linear(hidden_dim, out_dim)
        with torch.no_grad():
            self.output_layer.weight.uniform_(
                -np.sqrt(6 / hidden_dim) / omega_0, np.sqrt(6 / hidden_dim) / omega_0
            )
        # self.output_layer = SineLayer(hidden_dim, out_dim, is_first=False, omega_0=omega_0)

        # self.dropout = nn.Dropout(dropout_rate)

    def forward(self, coords, latent_vol, latent_coil):
        # Hash encode the input coordinates.
        x = self.embed_fn(coords)
        
        # Concatenate embeddings and positional encodings.
        x = torch.cat([x, latent_vol, latent_coil], dim=-1)
        x0 = x.clone()
        
        for layer_idx, layer in enumerate(self.sine_layers):
            # Residual connection.
            if layer_idx == self.res_layer_idx:
                x = torch.cat([x, x0], dim=-1)

            x = layer(x)

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
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))
