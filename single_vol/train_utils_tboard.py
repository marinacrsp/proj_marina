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
        self.hparam_info["resolution_levels"] = config["model"]["params"]["levels"]
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
            
            if (epoch_idx + 1) >= self.E_epoch:
                
                if self.lossadded == True:
                    
                    ## Evaluate only pisco loss every 20 samples
                    # if (epoch_idx + 1) % 20 == 0:
                    
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
            
            self.optimizer.step()
            
            avg_loss += batch_loss.item() * len(inputs)
            n_obs += len(inputs)        
        avg_loss = avg_loss / n_obs
        return avg_loss
    
    
    def _train_with_Lpisco (self):
        self.model.train()
        n_obs = 0
        avg_loss = 0.0
        avg_res1 = 0.0
        avg_res2 = 0.0
        vol_id = 0
        shape = self.dataloader_pisco.dataset.metadata[vol_id]["shape"]
        _, n_coils, _, _ = shape
        
        for inputs, _ in self.dataloader_pisco:
            # self.optimizer.zero_grad(set_to_none=True)
            
            #### Compute grid 
            t_coordinates, patch_coordinates, Nn = get_grappa_matrixes(inputs, shape)
            
            # Estimate the minibatch list of Ws together with the averaged residuals of the minibatch 
            Ws, batch_r1, batch_r2 = self.predict_ws(t_coordinates, patch_coordinates, n_coils, Nn)
            
            # Compute the pisco loss
            batch_Lp = L_pisco(Ws) * self.factor
            
            assert batch_Lp.requires_grad, "batch_Lp does not require gradients."
            
            # Update the model based on the Lpisco loss
            batch_Lp.backward()
            
            self.optimizer.step()

            # Add up the losses and residual averages to the batch sums
            avg_loss += batch_Lp.item() * len(inputs)
            avg_res1 += batch_r1
            avg_res2 += batch_r2
            
            n_obs += len(inputs)
        
        # From this set of points, the following residuals and losses are obtained
        avg_loss = avg_loss / n_obs
        avg_res1 = avg_res1/n_obs
        # avg_res2 = avg_res2/n_obs
        return avg_loss, avg_res1, avg_res2

    ###########################################################################
    ###########################################################################
    ###########################################################################
    
    def predict_ws (self, t_coordinates, patch_coordinates, n_coils, Nn):
        
        t_predicted = torch.zeros((t_coordinates.shape[0], n_coils), dtype=torch.complex64)
        patch_predicted = torch.zeros((patch_coordinates.shape[0], n_coils), dtype=torch.complex64)
        t_coordinates, patch_coordinates = t_coordinates.to(self.device), patch_coordinates.to(self.device) 

        for coil_id in range(n_coils):
            t_predicted[:,coil_id] = torch.view_as_complex(self.model(t_coordinates[:,coil_id,:]))
            patch_predicted[:,coil_id] = torch.view_as_complex(self.model(patch_coordinates[:,coil_id,:])).detach()
            # Reshape back the patch to its original shape : Nm x Nn x Nc
            
        patch_predicted = patch_predicted.view(t_coordinates.shape[0], Nn, n_coils)
        
        ##### Estimate the Ws for a random subset of values
        T_s, Ns = split_batch(t_predicted, self.minibatch)
        P_s, _ = split_batch(patch_predicted, self.minibatch)
        
        # Estimate the Weight matrixes
        Ws = []
        elem1 = 0
        elem2 = 0
        for i, t_s in enumerate(T_s):
            p_s = P_s[i]
            p_s = torch.flatten(p_s, start_dim=1)
            ws, res1, res2 = compute_Lsquares(p_s, t_s, self.alpha)
            Ws.append(ws)
            
            # Compute an average of the equation residuals
            elem1 += res1
            elem2 += res2
        return Ws, elem1/Ns, elem2/Ns
    
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

            coords[:, 0] = point_ids[:, 0]
            coords[:, 1] = point_ids[:, 1]
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
        volume_kspace[..., left_idx:right_idx,:] = center_vals #NOTE Shape of volume kspace is [n_coils, n_slices, height, width, 2] it's not in complex form
                
        volume_img = rss(inverse_fft2_shift(tensor_to_complex_np(volume_kspace)))
        
        self.model.train()
        return volume_img

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

        volume_img = self.predict(vol_id, shape, left_idx, right_idx, center_vals)
        
        volume_kspace = fft2_shift(volume_img)  # To get "single-coil" k-space.
        
        volume_kspace[..., left_idx:right_idx] = 0

        ##################################################
        # Log kspace values.
        ##################################################
        cste_mod = self.dataloader_consistency.dataset.metadata[vol_id]["plot_cste"]
        cste_arg = np.pi / 180
        cste_real = self.dataloader_consistency.dataset.metadata[vol_id]["plot_cste"]
        cste_imag = cste_real
        
        # Plot modulus and argument.
        modulus = np.abs(volume_kspace)
        argument = np.angle(volume_kspace)
        
        # Plot real and imaginary parts.
        real_part= np.real(volume_kspace)
        imag_part = np.imag(volume_kspace)

        ##################################################
        # Log image space values
        ##################################################
        volume_img= np.abs(volume_img)

        for slice_id in range(shape[0]):
            self._plot_info(
                modulus[slice_id],
                argument[slice_id],
                cste_mod,
                cste_arg,
                "Modulus",
                "Argument",
                epoch_idx,
                f"prediction/slice_{slice_id}/kspace_v1",
            )
            self._plot_info(
                real_part[slice_id],
                imag_part[slice_id],
                cste_real,
                cste_imag,
                "Real part",
                "Imaginary part",
                epoch_idx,
                f"prediction/slice_{slice_id}/kspace_v2",
            )

            # Plot image.
            fig = plt.figure(figsize=(8, 8))
            plt.imshow(volume_img[slice_id])
            self.writer.add_figure(
                f"prediction/slice_{slice_id}/volume_img", fig, global_step=epoch_idx
            )
            plt.close(fig)
            

        # Log evaluation metrics.
        ssim_val = ssim(self.ground_truth, volume_img)
        self.writer.add_scalar("eval/ssim", ssim_val, epoch_idx)

        psnr_val = psnr(self.ground_truth, volume_img)
        self.writer.add_scalar("eval/psnr", psnr_val, epoch_idx)

        nmse_val = nmse(self.ground_truth, volume_img)
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
        plt.imshow(data_1 / cste_1)
        plt.colorbar()
        plt.title(f"{title_1} kspace")

        plt.subplot(2, 2, 2)
        plt.hist(data_1.flatten(), log=True, bins=100)

        max_val = np.max(data_1)
        min_val = np.min(data_1)
        # ignoring zero data
        non_zero = data_1 > 0
        mean = np.mean(data_1[non_zero])
        median = np.median(data_1[non_zero])
        q05 = np.quantile(data_1[non_zero], 0.05)
        q95 = np.quantile(data_1[non_zero], 0.95)

        plt.axvline(
            mean, color="r", linestyle="dashed", linewidth=2, label=f"Mean: {mean:.2e}"
        )
        plt.axvline(
            median,
            color="g",
            linestyle="dashed",
            linewidth=2,
            label=f"Median: {median:.2e}",
        )
        plt.axvline(
            q05, color="b", linestyle="dotted", linewidth=2, label=f"Q05: {q05:.2e}"
        )
        plt.axvline(
            q95, color="b", linestyle="dotted", linewidth=2, label=f"Q95: {q95:.2e}"
        )
        plt.axvline(
            min_val,
            color="orange",
            linestyle="solid",
            linewidth=2,
            label=f"Min: {min_val:.2e}",
        )
        plt.axvline(
            max_val,
            color="purple",
            linestyle="solid",
            linewidth=2,
            label=f"Max: {max_val:.2e}",
        )
        plt.legend()
        plt.title(f"{title_1} histogram")

        plt.subplot(2, 2, 3)
        plt.imshow(data_2 / cste_2)
        plt.colorbar()
        plt.title(f"{title_2} kspace")

        plt.subplot(2, 2, 4)
        plt.hist(data_2.flatten(), log=True, bins=100)

        max_val = np.max(data_2)
        min_val = np.min(data_2)
        # ignoring zero data
        non_zero = data_2 > 0
        mean = np.mean(data_2[non_zero])
        median = np.median(data_2[non_zero])
        q05 = np.quantile(data_2[non_zero], 0.05)
        q95 = np.quantile(data_2[non_zero], 0.95)

        plt.axvline(
            mean, color="r", linestyle="dashed", linewidth=2, label=f"Mean: {mean:.2e}"
        )
        plt.axvline(
            median,
            color="g",
            linestyle="dashed",
            linewidth=2,
            label=f"Median: {median:.2e}",
        )
        plt.axvline(
            q05, color="b", linestyle="dotted", linewidth=2, label=f"Q05: {q05:.2e}"
        )
        plt.axvline(
            q95, color="b", linestyle="dotted", linewidth=2, label=f"Q95: {q95:.2e}"
        )
        plt.axvline(
            min_val,
            color="orange",
            linestyle="solid",
            linewidth=2,
            label=f"Min: {min_val:.2e}",
        )
        plt.axvline(
            max_val,
            color="purple",
            linestyle="solid",
            linewidth=2,
            label=f"Max: {max_val:.2e}",
        )
        plt.legend()
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