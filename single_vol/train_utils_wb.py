import os
from pathlib import Path
from typing import Optional

import fastmri
import h5py
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from data_utils import *
from fastmri.data.transforms import tensor_to_complex_np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader, TensorDataset
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pisco import *
from helper_functions import *

# Initialize Weights and Biases logger

class Trainer:
    def __init__(
        self, dataloader, model, loss_fn, optimizer, scheduler, config
    ) -> None:
        self.device = torch.device(config["device"])
        self.n_epochs = config["n_epochs"]
        self.lossadded = config["l_pisco"]["type"]
        self.E_epoch = config["l_pisco"]["E_epoch"]
        self.minibatch = config["l_pisco"]["minibatch_size"]
        self.alpha = config["l_pisco"]["alpha"]
        self.factor = config["l_pisco"]["factor"]
        self.dataloader = dataloader
        self.model = model.to(self.device)
        

        if hasattr(loss_fn, "to"):
            self.loss_fn = loss_fn.to(self.device)
        else:
            self.loss_fn = loss_fn

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.log_interval = config["log_interval"]
        self.checkpoint_interval = config["checkpoint_interval"]
        self.path_to_out = Path(config["path_to_outputs"])
        self.timestamp = config["timestamp"]
        
        self.project = config["wandb"]["project_name"]
        self.run = wandb.init(project= self.project, name=self.timestamp )
        
        # Initialize Wandb logger
        self.wandb_logger = WandbLogger(project=self.project )
        self.wandb_logger.experiment.config.update(config)  # log config details
            
            
        # Ground truth for evaluation
        file = self.dataloader.dataset.metadata[0]["file"]
        with h5py.File(file, "r") as hf:
            self.ground_truth = hf["reconstruction_rss"][()][
                : config["dataset"]["n_slices"]
            ]

        # Metrics placeholders
        self.last_nmse = [0] * len(self.dataloader.dataset.metadata)
        self.last_psnr = [0] * len(self.dataloader.dataset.metadata)
        self.last_ssim = [0] * len(self.dataloader.dataset.metadata)

    def train(self):
        # Example of logging within training
        empirical_risk = 0
        pisco_empirical_risk = 0
        
        for epoch_idx in range(self.n_epochs):
            print(f"EPOCH {epoch_idx}    avg loss: {empirical_risk}\n")
            empirical_risk = self._train_one_epoch()
            
            wandb.log({"train_loss": empirical_risk})
            
            if epoch_idx > self.E_epoch:
                if self.lossadded == "addPisco":
                    print(f"EPOCH {epoch_idx}  Evaluating Pisco loss: {pisco_empirical_risk}\n")
                    pisco_empirical_risk, Ws = self._train_with_Lpisco()
                    
                    wandb.log({"Pisco Loss": pisco_empirical_risk})                   
                    
            if epoch_idx % self.log_interval == 0 and epoch_idx != 0:
                # Log loss and other metrics to wandb
                self._log_performance(epoch_idx) # log the performance, ssim, psnr, nmse evaluation results
                self._log_weight_info(epoch_idx) # 
                
        self._log_information(empirical_risk)        
        # wandb.log({"Grappa Matrix": [t.item() for t in Ws]})
        wandb.finish()
    
    def _train_with_Lpisco (self):
        self.model.train()
        n_obs = 0
        avg_loss = 0.0
        vol_id = 0
        shape = self.dataloader.dataset.metadata[vol_id]["shape"]
        _, n_coils, _, _ = shape
        
        for inputs, _ in self.dataloader:
            # inputs = inputs.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            
            #### Compute grid 
            t_coordinates, patch_coordinates, Nn = get_grappa_matrixes(inputs, shape)

            Ws = self.predict_ws(t_coordinates, patch_coordinates, n_coils, Nn)
            
            batch_Lp = L_pisco(Ws)
            batch_Lp *= self.factor
            assert batch_Lp.requires_grad, "batch_Lp does not require gradients."
            
            # Update the model based on the Lpisco loss
            batch_Lp.backward()
            self.optimizer.step()

            avg_loss += batch_Lp.item() * len(inputs)
            n_obs += len(inputs)
        
        self.scheduler.step()
        avg_loss = avg_loss / n_obs
        return avg_loss, Ws
        
    # Here we do want to calculate the gradients
    def _train_one_epoch(self):
        avg_loss = 0.0
        n_obs = 0

        self.model.train()
        for inputs, targets in self.dataloader:
            
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Initialize the gradients to zero
            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(inputs)
            batch_loss = self.loss_fn(outputs, targets)
            # print(batch_loss)
            batch_loss.backward()
            self.optimizer.step()
            
            avg_loss += batch_loss.item() * len(inputs)
            n_obs += len(inputs)
            
        self.scheduler.step()
        avg_loss = avg_loss / n_obs
        return avg_loss
    
    

            
    def predict_ws (self, t_coordinates, patch_coordinates, n_coils, Nn):
        
        t_predicted = torch.zeros((t_coordinates.shape[0], n_coils), dtype=torch.complex64)
        patch_predicted = torch.zeros((patch_coordinates.shape[0], n_coils), dtype=torch.complex64)
        t_coordinates, patch_coordinates = t_coordinates.to(self.device), patch_coordinates.to(self.device) 

        for coil_id in range(n_coils):
            t_predicted[:,coil_id] = torch.view_as_complex(self.model(t_coordinates[:,coil_id,:])).detach()
            patch_predicted[:,coil_id] = torch.view_as_complex(self.model(patch_coordinates[:,coil_id,:])).detach()
            
        torch.cuda.empty_cache()  # Clear unused cached memory
        
        # Reshape back the patch to its original shape : Nm x Nn x Nc
        patch_predicted = patch_predicted.view(t_coordinates.shape[0], Nn, n_coils)
        ##### Estimate the Ws for a random subset of values
        T_s, _ = split_batch(t_predicted, self.minibatch)
        P_s, _ = split_batch(patch_predicted, self.minibatch)
        
        # Estimate the Weight matrixes
        Ws = []
        for i, t_s in enumerate(T_s):
            p_s = P_s[i]
            p_s = torch.flatten(p_s, start_dim=1)
            Ws.append(compute_Lsquares(p_s, t_s, self.alpha))
        Ws = [W.requires_grad_() if not W.requires_grad else W for W in Ws]
        return Ws
            
###########################################################################
###########################################################################
###########################################################################
            
    @torch.no_grad()
    # Here we don't want to calculate the gradients
    def predict(self, vol_id, shape, left_idx, right_idx, center_vals):
        self.model.eval()
        n_slices, n_coils, height, width = shape

        kx_ids = torch.cat([torch.arange(left_idx), torch.arange(right_idx, width)])  # Predict everything but the center
        
        ky_ids = torch.arange(height)
        kz_ids = torch.arange(n_slices)
        coil_ids = torch.arange(n_coils)

        kspace_ids = torch.meshgrid(kx_ids, ky_ids, kz_ids, coil_ids, indexing="ij")
        kspace_ids = torch.stack(kspace_ids, dim=-1).reshape(-1, len(kspace_ids))

        dataset = TensorDataset(kspace_ids)
        dataloader = DataLoader(dataset, batch_size=60_000, shuffle=False, num_workers=3)

        volume_kspace = torch.zeros(
            (n_slices, n_coils, height, width, 2),
            device=self.device,
            dtype=torch.float32,
        )
        
        for point_ids in dataloader:
            point_ids = point_ids[0].to(self.device, dtype=torch.long)
            coords = torch.zeros_like(point_ids, dtype=torch.float32, device=self.device)

            coords[:, 0] = (2 * point_ids[:, 0]) / (width - 1) - 1
            coords[:, 1] = (2 * point_ids[:, 1]) / (height - 1) - 1
            coords[:, 2] = (2 * point_ids[:, 2]) / (n_slices - 1) - 1
            coords[:, 3] = (2 * point_ids[:, 3]) / (n_coils - 1) - 1

            outputs = self.model(coords)
            # n_slices, n_coils, height, width
            volume_kspace[point_ids[:, 2], point_ids[:, 3], point_ids[:, 1], point_ids[:, 0]] = outputs

        volume_kspace = volume_kspace * self.dataloader.dataset.metadata[vol_id]["norm_cste"]
        volume_kspace = tensor_to_complex_np(volume_kspace.detach().cpu())

        volume_kspace[..., left_idx:right_idx] = center_vals
        volume_img = rss(inverse_fft2_shift(volume_kspace))

        self.model.train()
        return volume_img
    
    ###########################################################################
    ###########################################################################
    ###########################################################################
    
    @torch.no_grad()
    def _log_performance(self, epoch_idx):
        vol_id=0
        shape = self.dataloader.dataset.metadata[vol_id]["shape"]
        center_data = self.dataloader.dataset.metadata[vol_id]["center"]
        left_idx, right_idx, center_vals = (
            center_data["left_idx"],
            center_data["right_idx"],
            center_data["vals"],
        )

        volume_img = self.predict(vol_id, shape, left_idx, right_idx, center_vals)
        volume_kspace = fft2_shift(volume_img)
        volume_kspace[..., left_idx:right_idx] = 0

        modulus = np.abs(volume_kspace)
        cste_mod = self.dataloader.dataset.metadata[vol_id]["plot_cste"]
        argument = np.angle(volume_kspace)
        cste_arg = np.pi / 180

        real_part = np.real(volume_kspace)
        imag_part = np.imag(volume_kspace)
        
        ##################################################
        # Log image space values
        ##################################################
        volume_img = np.abs(volume_img)

        for slice_id in range(shape[0]):
            
            fig = plt.figure()
            plt.subplot(1,3,1)
            plt.imshow(volume_img[slice_id])
            plt.title('Real fig')
            plt.axis('off')
            
            plt.subplot(1,3,2)
            plt.imshow(modulus[slice_id])
            plt.title('Modulus')
            plt.axis('off')
            
            plt.subplot(1,3,3)
            plt.imshow(argument[slice_id])
            plt.title('Angle')
            plt.axis('off')
            
            fig.tight_layout()
            plt.close(fig)
            
            wandb.log({"prediction output": fig})

        ssim_val = ssim(self.ground_truth, volume_img)
        psnr_val = psnr(self.ground_truth, volume_img)
        nmse_val = nmse(self.ground_truth, volume_img)

        wandb.log({
            "eval/ssim": ssim_val,
            "eval/psnr": psnr_val,
            "eval/nmse": nmse_val
        })

        self.last_nmse = nmse_val
        self.last_psnr = psnr_val
        self.last_ssim = ssim_val

    @torch.no_grad()
    def _log_weight_info(self, epoch_idx):
        for name, param in self.model.named_parameters():
            wandb.log({
                f"params/{name}/values": wandb.Histogram(param.data.cpu().numpy()),
                f"params/{name}/gradients": wandb.Histogram(param.grad.cpu().numpy()) if param.grad is not None else None
            })

    @torch.no_grad()
    def _save_checkpoint(self, epoch_idx):
        path = self.path_to_out / self.timestamp / "checkpoints"
        os.makedirs(path, exist_ok=True)
        path_to_file = path / f"epoch_{epoch_idx:04d}.pt"
        
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }
        torch.save(save_dict, path_to_file)

    @torch.no_grad()
    def _log_information(self, loss):
        hparam_metrics = {
            "hparam/loss": loss,
            "hparam/eval_metric/nmse": np.mean(self.last_nmse),
            "hparam/eval_metric/psnr": np.mean(self.last_psnr),
            "hparam/eval_metric/ssim": np.mean(self.last_ssim)}
        
        wandb.log(hparam_metrics)
        
        inputs, _ = next(iter(self.dataloader))
        inputs = inputs.to(self.device)
        wandb.watch(self.model, log="all", log_freq=100)