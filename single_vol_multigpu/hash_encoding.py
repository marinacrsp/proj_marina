import torch
from torch import nn, Tuple
import numpy as np



class hash_encoder:
    """
    Class that computes the encoding for a given point xi, based on concatenation of bounding boxes embeddings at differnt resolution levels L.
    
    """
    def __init__(self, levels=10, log2_hashmap_size=12, n_features_per_level=2, n_max=320, n_min=16):
        self.l_max = levels # Total number of levels of resolution 
        self.log2_hashmap_size = log2_hashmap_size # size of hash table
        self.n_features_per_level = n_features_per_level # number of features per level
        self.n_max = n_max # Finest resolution in grid
        self.n_min = n_min # Coarsest resolution in grid
        self.b = np.exp((np.log(self.n_max) - np.log(self.n_min)) / (self.l_max - 1)) # Stepsize of resolution increment

        # Initialize embeddings for each level
        # Contains gradients!!
        self.embeddings = nn.ModuleList([
            nn.Embedding(self._get_number_of_embeddings(i), self.n_features_per_level)
            for i in range(self.l_max)
        ])
    
    def _get_number_of_embeddings(self, level_idx: int) -> int:
        # Computes the number of embeddings at a particular resolution level L
        max_size = 2 ** self.log2_hashmap_size
        n_l = int(self.n_min * (self.b ** level_idx).item())
        n_l_embeddings = (n_l + 2) ** 2  # Added padding as specified
        return min(max_size, n_l_embeddings)

    def bilinear_interp(self, x: torch.Tensor, box_indices: torch.Tensor, box_embedds: torch.Tensor) -> torch.Tensor:
        # bilinear interpolation of the embedding of a point xi, based on its surrounding bounding box vertices

        if box_indices.shape[0] > 2:
            weights = [np.linalg.norm(box_indices[i] - x) for i in range(4)]
            den = sum(weights)
            weights = [w / den for w in weights]
            xi_embedding = sum(weights[i] * box_embedds[i] for i in range(4))
        else:
            xi_embedding = box_embedds
        return xi_embedding
    
    def _get_box_idx(self, point: torch.Tensor , n_l: int) -> tuple:
        # Given the coordinates of a point xi, and the resolution level of interest, function that returns the coordinates of the bounding box
        
        x = point[:,0]
        y = point[:,1]
        if self.n_max == n_l: # NOTE: If the resolution of the grid is just the finest resolution possible, the bounding box is just the point
            box_idx = torch.tensor((x, y))
        else:
            box_width = self.n_max // n_l
            box_height = self.n_max // n_l
            x_min = max(0, (x // box_width) * box_width)
            y_min = max(0, (y // box_height) * box_height)
            x_max = min(self.n_max, x_min + box_width)
            y_max = min(self.n_max, y_min + box_height)
            box_idx = torch.tensor([[x_min, y_min], [x_max, y_min], [x_min, y_max], [x_max, y_max]])
        
        # Derive the embeddings of the vertices by indexing a hash table
        # Compute the hash value for the vertices corners
        max_hashtable_size = 2 ** self.log2_hashmap_size
        n_l_embeddings = (n_l + 2) ** 2
        
        if max_hashtable_size > n_l_embeddings:
            hashed_box_idx, box_idx_scaled = self._to_1D(box_idx, n_l)
        else:
            hashed_box_idx = self._hash(box_idx)
            box_idx_scaled = box_idx
        
        return box_idx, box_idx_scaled, hashed_box_idx   
    
    
    ## Hash encoders
    def _to_1D(self, coors: torch.Tensor, n_l: int) -> tuple:
        # For the non collision case, hash value can be computed directly with this function
        # Normalization of the bounding box coordinates is done
        scale_factor = self.n_max // n_l
        scaled_coords = torch.div(coors, scale_factor, rounding_mode="floor").int()
        x, y = scaled_coords[:, 0], scaled_coords[:, 1]
        return y * n_l + x, scaled_coords

    def _hash(self, coords: torch.Tensor) -> torch.Tensor:
        # When collisions start to happen, calculate the hash values with this function
        # No normalization is done
        primes = torch.tensor([1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737], dtype=torch.int64)
        
        xor_result = torch.zeros_like(coords, dtype=torch.int64)[..., 0]
        
        for i in range(coords.shape[-1]):
            xor_result ^= coords[..., i].to(torch.int64) * primes[i]
            
        return (1 << self.log2_hashmap_size) - 1 & xor_result
    
    
    ## Forward pass of the class
    def forward(self, point):
        x_embedded_all = []
        
        for i in range(self.l_max):
            n_l = int(self.n_min * self.b**i)
            box_idx, box_idx_scaled, hashed_box_idx = self._get_box_idx(point, n_l)
            
            # print(f"New level: \n {box_idx}")
            # print(box_idx_scaled)
            
            box_embedds = self.embeddings[i](hashed_box_idx)
            x_embedded = self.bilinear_interp(point, box_idx, box_embedds)
            x_embedded_all.append(x_embedded)
        
        return torch.cat(x_embedded_all, dim=-1)
    
    def __call__(self, point_coors: torch.Tensor) -> torch.Tensor:
        return self.forward(point_coors)