import os
from pathlib import Path
from typing import Optional

import fastmri
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from data_utils import *
from fastmri.data.transforms import tensor_to_complex_np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader, TensorDataset
from pisco import *
from helper_functions import *
from torch.utils.tensorboard import SummaryWriter # To print to tensorboard


class Trainer:
    def __init__(
        self, dataloader_consistency, dataloader_pisco, model, loss_fn, optimizer, scheduler, config
    ) -> None:
        self.device = torch.device(config["device"])
        self.n_epochs = config["n_epochs"]

        self.dataloader_consistency = dataloader_consistency
        self.dataloader_pisco = dataloader_pisco
        self.model = model.to(self.device)

        # If stateful loss function, move its "parameters" to `device`.
        if hasattr(loss_fn, "to"):
            self.loss_fn = loss_fn.to(self.device)
        else:
            self.loss_fn = loss_fn
            
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epsilon = config["dataset"]["epsilon"]
        
        self.log_interval = config["log_interval"]
        self.checkpoint_interval = config["n_epochs"] # Initialize the checkpoint interval as this value
        self.path_to_out = Path(config["path_to_outputs"])
        self.timestamp = config["timestamp"]
        
        self.lossadded = config["l_pisco"]["addpisco"]
        self.E_epoch = config["l_pisco"]["E_epoch"]
        self.minibatch = config["l_pisco"]["minibatch_size"]
        self.alpha = config["l_pisco"]["alpha"]
        self.factor = config["l_pisco"]["factor"]
        
        self.best_ssim = 0.84
        self.best_psnr = 40.0
        
        self.writer = SummaryWriter(self.path_to_out / self.timestamp)

        # Ground truth (used to compute the evaluation metrics).
        file = self.dataloader_consistency.dataset.metadata[0]["file"]
        with h5py.File(file, "r") as hf:
            self.ground_truth = hf["reconstruction_rss"][()][
                : config["dataset"]["n_slices"]
            ]

        # Scientific and nuissance hyperparameters.
        self.hparam_info = config["hparam_info"]
        self.hparam_info["n_layer"] = config["model"]["params"]["n_layers"]
        self.hparam_info["hidden_dim"] = config["model"]["params"]["hidden_dim"]
        # self.hparam_info["L_encoding"] = config["model"]["params"]["L"]
        self.hparam_info["embedding_dim"] = config["model"]["params"]["embedding_dim"]
        self.hparam_info["batch_size"] = config["dataloader"]["batch_size"]
        self.hparam_info["pisco_weightfactor"] = config["l_pisco"]["factor"]
        
        print(self.hparam_info)
        # self.hparam_info["loss"] = config["loss"]["id"]
        # self.hparam_info["acceleration"] = config["dataset"]["acceleration"]
        # self.hparam_info["center_frac"] = config["dataset"]["center_frac"]

        # Evaluation metrics for the last log.
        self.last_nmse = [0] * len(self.dataloader_consistency.dataset.metadata)
        self.last_psnr = [0] * len(self.dataloader_consistency.dataset.metadata)
        self.last_ssim = [0] * len(self.dataloader_consistency.dataset.metadata)
        
    ###########################################################################
    ###########################################################################
    ###########################################################################

    def train(self):
        """Train the model across multiple epochs and log the performance."""
        empirical_risk = 0
        empirical_pisco = 0
        for epoch_idx in range(self.n_epochs):
            empirical_risk = self._train_one_epoch(epoch_idx)
            
            if epoch_idx > self.E_epoch:
                
                if self.lossadded == "addPisco":
                    empirical_pisco, epoch_res1, epoch_res2 = self._train_with_Lpisco()
                    
                    print(f"EPOCH {epoch_idx}  Pisco loss: {empirical_pisco}\n")
                    self.writer.add_scalar("Residuals/Linear", epoch_res1, epoch_idx)
                    self.writer.add_scalar("Residuals/Regularizer", epoch_res2, epoch_idx)
                    
            print(f"EPOCH {epoch_idx}    avg loss: {empirical_risk}\n")
            # Log the errors
            self.writer.add_scalar("Loss/train", empirical_risk, epoch_idx)
            self.writer.add_scalar("Loss/Pisco", empirical_pisco, epoch_idx)
            
            # Log the average residuals
            # TODO: UNCOMMENT WHEN USING LR SCHEDULER.
            self.writer.add_scalar("Learning Rate", self.scheduler.get_last_lr()[0], epoch_idx)

            if (epoch_idx + 1) % self.log_interval == 0:
                self._log_performance(epoch_idx)
                self._log_weight_info(epoch_idx)

            if (epoch_idx + 1) % self.checkpoint_interval == 0:
                # Takes ~3 seconds.
                self._save_checkpoint(epoch_idx)

        self._log_information(empirical_risk)
        self.writer.close()
        
        
    def _train_one_epoch(self, epoch_idx):
        # Also known as "empirical risk".
        avg_loss = 0.0
        n_obs = 0
        
        self.model.train()
        for inputs, targets in self.dataloader_consistency:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(inputs)
            
            # Can be thought as a moving average (with "stride" `batch_size`) of the loss.
            batch_loss = self.loss_fn(outputs, targets)
            
            # NOTE: Uncomment for some of the loss functions (e.g. 'MSEDistLoss').
            # batch_loss = self.loss_fn(outputs, targets, inputs)

            batch_loss.backward()
            #########################################
            # At the beginning there's no Pisco loss
            if epoch_idx < self.E_epoch and self.lossadded == True:
                self.optimizer.step()
                # self.scheduler.step()
                
            # If there is no Pisco loss to train with at all 
            elif self.lossadded == False:
                self.optimizer.step()
            
            avg_loss += batch_loss.item() * len(inputs)
            n_obs += len(inputs)        
        avg_loss = avg_loss / n_obs
        return avg_loss
    
    
    @torch.no_grad()
    def predict(self, vol_id, shape, left_idx, right_idx, center_vals):
        """Reconstruct MRI volume (k-space)."""
        self.model.eval()
        n_slices, n_coils, height, width = shape

        # Create tensors of indices for each dimension
        kx_ids = torch.cat([torch.arange(left_idx), torch.arange(right_idx, width)])
        ky_ids = torch.arange(height)
        kz_ids = torch.arange(n_slices)
        coil_ids = torch.arange(n_coils)

        # Use meshgrid to create expanded grids
        kspace_ids = torch.meshgrid(kx_ids, ky_ids, kz_ids, coil_ids, indexing="ij")
        kspace_ids = torch.stack(kspace_ids, dim=-1).reshape(-1, len(kspace_ids))

        dataset = TensorDataset(kspace_ids)
        dataloader = DataLoader(
            dataset, batch_size=60_000, shuffle=False, num_workers=3
        )

        volume_kspace = torch.zeros(
            (n_slices, n_coils, height, width, 2),
            device=self.device,
            dtype=torch.float32,
        )
        
        for point_ids in dataloader:
            point_ids = point_ids[0].to(self.device, dtype=torch.long)
            coords = torch.zeros_like(
                point_ids, dtype=torch.float32, device=self.device
            )

            # coords[:, 0] = (2 * point_ids[:, 0]) / (width - 1) - 1
            # coords[:, 1] = (2 * point_ids[:, 1]) / (height - 1) - 1
            coords[:, :2] = point_ids[:, :2]
            coords[:, 2] = (2 * point_ids[:, 2]) / (n_slices - 1) - 1
            coords[:, 3] = (2 * point_ids[:, 3]) / (n_coils - 1) - 1

            outputs = self.model(coords)
            # "Fill in" the unsampled region.
            volume_kspace[
                point_ids[:, 2], point_ids[:, 3], point_ids[:, 1], point_ids[:, 0]
            ] = outputs

        # Multiply by the normalization constant.
        volume_kspace = (
            volume_kspace * self.dataloader_consistency.dataset.metadata[vol_id]["norm_cste"]
        )

        volume_kspace = volume_kspace.detach().cpu()

        # "Fill-in" center values.
        volume_kspace[..., left_idx:right_idx,:] = center_vals #NOTE Shape of volume kspace is [n_coils, n_slices, height, width, 2] it's in complex form
        
        ilog_volume_kspace = inv_transform_magnitude(volume_kspace, self.epsilon)
        
        ilog_volume_img = rss(inverse_fft2_shift(torch.view_as_complex(ilog_volume_kspace)))
        log_volume_img = rss(inverse_fft2_shift(torch.view_as_complex(volume_kspace)))
        

        self.model.train()
        return log_volume_img, ilog_volume_img

    ###########################################################################
    ###########################################################################
    ###########################################################################

    @torch.no_grad()
    def _log_performance(self, epoch_idx, vol_id=0):
        # Predict volume image.
        shape = self.dataloader_consistency.dataset.metadata[vol_id]["shape"]
        center_data = self.dataloader_consistency.dataset.metadata[vol_id]["center"]
        left_idx, right_idx, center_vals = (
            center_data["left_idx"],
            center_data["right_idx"],
            center_data["vals"],
        )

        volume_img_log, volume_img_ilog = self.predict(vol_id, shape, left_idx, right_idx, center_vals)
        
        volume_kspace_ilog = fft2_shift(volume_img_ilog)  # To get "single-coil" k-space.
        volume_kspace_log = fft2_shift(volume_img_log)
        
        volume_kspace_ilog[..., left_idx:right_idx] = 0
        volume_kspace_log[..., left_idx:right_idx] = 0

        ##################################################
        # Log kspace values.
        ##################################################
        cste_mod = self.dataloader_consistency.dataset.metadata[vol_id]["plot_cste"]
        cste_arg = np.pi / 180
        cste_real = self.dataloader_consistency.dataset.metadata[vol_id]["plot_cste"]
        cste_imag = cste_real
        
        # Plot modulus and argument.
        modulus_ilog = np.abs(volume_kspace_ilog)
        argument_ilog = np.angle(volume_kspace_ilog)
        
        # Plot real and imaginary parts.
        real_part_ilog = np.real(volume_kspace_ilog)
        imag_part_ilog = np.imag(volume_kspace_ilog)
        
        # Plot modulus and argument 
        modulus_log = np.abs(volume_kspace_log)
        argument_log = np.angle(volume_kspace_log)
        
        # Plot real and imaginary parts.
        real_part_log = np.real(volume_kspace_log)
        imag_part_log = np.imag(volume_kspace_log)
        
        ##################################################
        # Log image space values
        ##################################################
        volume_img_ilog = np.abs(volume_img_ilog)
        volume_img_log = np.abs(volume_img_log)

        for slice_id in range(shape[0]):
            self._plot_info(
                modulus_ilog[slice_id],
                argument_ilog[slice_id],
                cste_mod,
                cste_arg,
                "Modulus",
                "Argument",
                epoch_idx,
                f"prediction/slice_{slice_id}/kspace_ilogtransformed",
            )
            
            ## Do the same with the non transformed images
            self._plot_info(
                modulus_log[slice_id],
                argument_log[slice_id],
                cste_mod,
                cste_arg,
                "Modulus",
                "Argument",
                epoch_idx,
                f"prediction/slice_{slice_id}/kspace_transformed",
            )

            # Plot image.
            fig = plt.figure(figsize=(8, 8))
            plt.imshow(volume_img_ilog[slice_id])
            plt.axis('off')
            self.writer.add_figure(
                f"prediction/slice_{slice_id}/volume_img_ilogtransform", fig, global_step=epoch_idx
            )
            plt.close(fig)
            
            # Plot image.
            fig = plt.figure(figsize=(8, 8))
            plt.imshow(volume_img_log[slice_id])
            
            self.writer.add_figure(
                f"prediction/slice_{slice_id}/volume_img_logtransform", fig, global_step=epoch_idx
            )
            plt.close(fig)

        # Log evaluation metrics.
        ssim_val = ssim(self.ground_truth, volume_img_ilog)
        self.writer.add_scalar("eval/ssim", ssim_val, epoch_idx)

        psnr_val = psnr(self.ground_truth, volume_img_ilog)
        self.writer.add_scalar("eval/psnr", psnr_val, epoch_idx)

        nmse_val = nmse(self.ground_truth, volume_img_ilog)
        self.writer.add_scalar("eval/nmse", nmse_val, epoch_idx)

        # Update.
        self.last_nmse = nmse_val
        self.last_psnr = psnr_val
        self.last_ssim = ssim_val
        
        if self.best_ssim < ssim_val and self.best_psnr < psnr_val:
            self._save_checkpoint(epoch_idx)


    def _plot_info(
        self, data_1, data_2, cste_1, cste_2, title_1, title_2, epoch_idx, tag
    ):
        fig = plt.figure(figsize=(20, 20))
        plt.subplot(2, 2, 1)
        plt.imshow(data_1/cste_1)
        plt.colorbar()
        plt.axis('off')
        plt.title(f"{title_1} kspace")

        plt.subplot(2, 2, 2)
        plt.hist(data_1.flatten(), log=True, bins=100)
        # plt.legend()
        plt.title(f"{title_1} histogram")

        plt.subplot(2, 2, 3)
        plt.imshow(data_2/cste_2)
        plt.colorbar()
        plt.axis('off')
        plt.title(f"{title_2} kspace")

        plt.subplot(2, 2, 4)
        plt.hist(data_2.flatten(), log=True, bins=100)
        # plt.legend()
        plt.title(f"{title_2} histogram")

        self.writer.add_figure(tag, fig, global_step=epoch_idx)
        plt.close(fig)

    @torch.no_grad()
    def _log_weight_info(self, epoch_idx):
        """Log weight values and gradients."""
        for name, param in self.model.named_parameters():
            subplot_count = 1 if param.data is None else 2
            fig = plt.figure(figsize=(8 * subplot_count, 5))

            plt.subplot(1, subplot_count, 1)
            plt.hist(param.data.cpu().numpy().flatten(), bins=100, log=True)
            # plt.hist(param.data.cpu().numpy().flatten(), bins='auto', log=True)
            plt.title("Values")

            if param.grad is not None:
                plt.subplot(1, subplot_count, 2)
                # plt.hist(param.grad.cpu().numpy().flatten(), bins='auto', log=True)
                plt.hist(param.grad.cpu().numpy().flatten(), bins=100, log=True)
                plt.title("Gradients")

            tag = name.replace(".", "/")
            self.writer.add_figure(f"params/{tag}", fig, global_step=epoch_idx)
            plt.close(fig)

    @torch.no_grad()
    def _save_checkpoint(self, epoch_idx):
        """Save current state of the training process."""
        # Ensure the path exists.
        path = self.path_to_out / self.timestamp / "checkpoints"
        os.makedirs(path, exist_ok=True)

        path_to_file = path / f"epoch_{epoch_idx:04d}.pt"

        # Prepare state to save.
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }
        # Save trainer state.
        torch.save(save_dict, path_to_file)

    @torch.no_grad()
    def _log_information(self, loss):
        """Log 'scientific' and 'nuissance' hyperparameters."""

        # if hasattr(self.model, "activation"):
        #     self.hparam_info["hidden_activation"] = type(self.model.activation).__name__
        # elif type(self.model).__name__ == "Siren":
        #     self.hparam_info["hidden_activation"] = "Sine"
        # if hasattr(self.model, "out_activation"):
        #     self.hparam_info["output_activation"] = type(
        #         self.model.out_activation
        #     ).__name__
        # else:
        #     self.hparam_info["output_activation"] = "None"

        hparam_metrics = {"hparam/loss": loss}
        hparam_metrics["hparam/eval_metric/nmse"] = np.mean(self.last_nmse)
        hparam_metrics["hparam/eval_metric/psnr"] = np.mean(self.last_psnr)
        hparam_metrics["hparam/eval_metric/ssim"] = np.mean(self.last_ssim)
        
        self.writer.add_hparams(self.hparam_info, hparam_metrics)

        # Log model's architecture.
        inputs, _ = next(iter(self.dataloader_consistency))
        inputs = inputs.to(self.device)
        self.writer.add_graph(self.model, inputs)


###########################################################################
###########################################################################
##########################################################################