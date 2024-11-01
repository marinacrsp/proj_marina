import torch
import numpy as np


def normalize (data, norm_factor):
    """Function that normalizes a data matrix to the range [-1,1]"""
    n_data = (2*data) / (norm_factor - 1) - 1 
    return n_data

def denormalize (n_data, norm_factor):
    """Function that reverts a normalized data matrix to the original range, specified by norm_factor"""
    data = ((n_data + 1) * (norm_factor - 1))/2
    return data

def split_batch (data, size_minibatch):
    """Function that performs the random spliting of the dataloader batch into Ns subsets of generally the same size"""
    total_batch = data.shape[0]
    iter = total_batch//size_minibatch
    sample_batch = []
    last_idx = 0
    
    for i in range(iter):
        if i == 0:
            minibatch = data[:size_minibatch,...]
        elif i==iter-1:
            minibatch = data[last_idx+1:,...]
        else:
            minibatch = data[last_idx+1:last_idx+size_minibatch,...]
        
        sample_batch.append(minibatch)
        last_idx += size_minibatch
    return sample_batch, iter

def compute_Lsquares (X, Y, alpha):
    """Solves the Least Squares giving matrix W"""
    P_TxP = torch.matmul(X.T, X)
    P_TxT = torch.matmul(X.T, Y)

    reg = alpha * torch.eye(P_TxP.shape[0])
    W = torch.linalg.solve(P_TxP + reg, P_TxT)
    
    PxW = torch.matmul(X, W)
    
    elem1 = torch.linalg.norm((Y - PxW), ord=2) 
    elem2 = torch.linalg.norm(W, ord=2) 

    return W, elem1, elem2


# def complex_distance(W1, W2):
#     """
#     Computes the L2 distance between two complex matrices W1 and W2.
#     It compares both real and imaginary parts.
#     """

#     # Compute the squared differences for both real and imaginary parts
#     real_diff = torch.norm(W1.real - W2.real)
#     # print(real_diff)
#     imag_diff = torch.norm(W1.imag - W2.imag)
#     # Return the combined distance
#     return real_diff + imag_diff


def L_pisco (Ws):
    """Function to compute the Pisco loss
    Inputs:
    - Ws (list) : contains the corresponding Ws computed from Least squares
    
    """
    # Compare the Ws, obtain the Pisco loss
    total_loss = 0
    Ns = len(Ws)
    for i in range(Ns):
        for j in range(i+1, Ns):
            diff = Ws[i].flatten() - Ws[j].flatten()
            pisco = torch.linalg.norm(diff, ord=1)
            total_loss += pisco
                
    return (1/Ns**2) * total_loss

def get_grappa_matrixes (inputs, shape):
    """Function that generates two matrixes out of the input coordinates of the batch points     
    - n_r_kcoors : normalized and reshaped matrix containing the kspace coordinates 
        dim -> (Nm x Nc x 4)
    - n_r_patch : normalized and reshaped matrix containing the kspace coordinates of the neighbourhood for each point in first matrix
        dim -> (Nm·Nn x Nc x 4)
    """
    n_slices, n_coils, height, width = shape
    k_coors = torch.zeros((inputs.shape[0], 4), dtype=torch.float)
    # k_coors[:,0] = denormalize(inputs[:,0], width)
    # k_coors[:,1] = denormalize(inputs[:,1], height)
    
    # NOTE Denormalize only the coordinates that contain normalized inputs
    k_coors[:,:2] = inputs[:,:2]
    k_coors[:,2] = denormalize(inputs[:,2], n_slices)
    k_coors[:,3] = denormalize(inputs[:,3], n_coils)
    
    # Remove the edges from the target coordinates
    leftmost_vedge = (k_coors[:, 1] == 0)
    rightmost_vedge = (k_coors[:, 1] == 319)
    upmost_vedge = (k_coors[:, 0] == 0)
    downmost_vedge = (k_coors[:, 0] == 319)

    edges = leftmost_vedge | rightmost_vedge | upmost_vedge | downmost_vedge
    k_coors_nedge = k_coors[~edges]

    #### Reshape:
    # Reshape input matrixes for coilID to be considered dim : n_points x N_coils x 4
    r_kcoors = np.repeat(k_coors_nedge[:, np.newaxis, :], n_coils, axis=1)
    r_kcoors[...,-1] = torch.arange(n_coils)
    
    ##### Reshape patches matrix to : n_points x n_neighbours x N_coils x 4
    build_neighbours = get_patch()
    patch_coors = build_neighbours(r_kcoors)
    
    # Reshape so that dim : n_points x N_n x Nc x 4 (kx,ky,kz, n_coils coordinates)
    r_patch = torch.zeros((patch_coors.shape[0],patch_coors.shape[1], r_kcoors.shape[2]))
    r_patch[...,:3] = patch_coors
    r_patch = np.repeat(r_patch[:, :, np.newaxis], n_coils, axis=2)
    r_patch[...,-1] = torch.arange(n_coils)

    ### For predicting, normalize coordinates back to [-1,1]
    # Normalize the NP neighbourhood coordinates
    n_r_patch = torch.zeros((r_patch.shape), dtype=torch.float)
    # n_r_patch[:,:,:,0] = normalize(r_patch[:,:,:,0], width)
    # n_r_patch[:,:,:,1] = normalize(r_patch[:,:,:,1], height)
    n_r_patch[...,:2] = r_patch[...,:2] # NOTE Normalize only the coordinates that contain normalized inputs
    n_r_patch[:,:,:,2] = normalize(r_patch[:,:,:,2], n_slices)
    n_r_patch[:,:,:,3] = normalize(r_patch[:,:,:,3], n_coils)
    
    # Flatten the first dimensions for the purpose of kvalue prediction
    Nn = n_r_patch.shape[1]
    n_r_patch = n_r_patch.view(-1, n_coils, 4)

    # Normalize the Nt targets coordinates
    n_r_koors = torch.zeros((r_kcoors.shape), dtype=torch.float)
    # n_r_koors[:,:,0] = normalize(r_kcoors[:,:,0], width)
    # n_r_koors[:,:,1] = normalize(r_kcoors[:,:,1], height)
    n_r_koors[:,:,:2] = r_kcoors[:,:,:2] # NOTE Normalize only the coordinates that contain normalized inputs
    n_r_koors[:,:,2] = normalize(r_kcoors[:,:,2], n_slices)
    n_r_koors[:,:,3] = normalize(r_kcoors[:,:,3], n_coils)
    
    return n_r_koors, n_r_patch, Nn


class get_patch:
    def __init__(
        self, 
        width = 320,
        height = 320,
        patch_size=9, 
        ):
        
        self.width = width
        self.height = height
        self.patch_size = patch_size
        
        super().__init__()
    
    def forward(self, batch_coors: torch.Tensor) -> torch.Tensor:
        """Returns the 3x3 neighbors for all points in a batch.
        Inputs : 
        - batch_coors : matrix of dimension batch_size x 4 denormalized coordinates (kx,ky,kz,coilid)
        """
        
        shifts = torch.tensor([[-1, -1], [0, -1], [1, -1],
                [ -1, 0], [ 1, 0],
                [ -1, 1], [ 0, 1], [ 1, 1]], device=batch_coors.device)  

        # Extract kx, ky from k_coor
        kx = batch_coors[:,:,0][:,0].unsqueeze(1)  # shape: (batch_size, 1)
        ky = batch_coors[:,:,1][:,1].unsqueeze(1)  # shape: (batch_size, 1)
        kz = batch_coors[:,:,2][:,0].unsqueeze(1)  # shape: (batch_size, 1)
        
        # Compute all neighbor shifts at once (apply shifts to kx, ky)
        kx_neighbors = torch.clamp(kx + shifts[:, 0], 0, self.width - 1)
        ky_neighbors = torch.clamp(ky + shifts[:, 1], 0, self.height - 1)
        
        # Ouput of neighbors dim : batch_size x nneighbors x 3 coordinates (kx, ky, kz)
        neighbors = torch.stack([kx_neighbors, ky_neighbors, kz.repeat(1, self.patch_size-1)], dim=-1)
        return neighbors
    
    def __call__(self, batch_coors: torch.Tensor) -> torch.Tensor:
        return self.forward(batch_coors)
    