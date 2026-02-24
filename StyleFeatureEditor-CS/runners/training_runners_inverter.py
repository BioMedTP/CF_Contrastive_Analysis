import os
import sys
import json
import wandb
import datetime
import omegaconf

import torch
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
from tqdm.auto import tqdm
from io import BytesIO
from PIL import Image
from time import time
from pathlib import Path
from abc import abstractmethod

from runners.base_runner import BaseRunner
from utils.class_registry import ClassRegistry
from datasets.transforms import transforms_registry
from datasets.datasets import ImageDataset, ImageDatasetFlexible
from datasets.loaders import InfiniteLoader
from training.losses import disc_losses, LossBuilder
from training.optimizers import optimizers
from metrics.metrics import metrics_registry
from utils.common_utils import tensor2im, get_keys, visualize_batch_grid
from training.loggers import Timer, StreamingMeans, TrainigLogger
from utils.common_utils import tensor2im, get_keys
from models.methods import methods_registry

from models.psp.encoders.psp_encoders import ProgressiveStage
from utils.model_utils import toogle_grad
from configs.data_paths import DATASETS

training_runners = ClassRegistry()
        

@training_runners.add_to_registry(name="base_training_runner")
class BaseTrainingRunner(BaseRunner):
    def setup(self):
        self.start_step = self.config.train.start_step
        self._setup_device()
        self._setup_experiment_dir()

        self._setup_method()
        self._setup_logger()

        self._setup_metrics()
        self._setup_datasets()

        start_batch_size = (
            self.config.train.bs_used_before_adv_loss
            if self.config.train.train_dis
            else self.config.model.batch_size
        )

        self._setup_dataloaders(start_batch_size)

        self._setup_latent_editor()
        self._setup_optimizers()
        self._setup_loss()

    def _setup_logger(self):
        self.logger = TrainigLogger(self.config)

    def _setup_datasets(self):
        print("Loading dataset")
        transforms = transforms_registry[self.config.data.transform]().get_transforms()
        ds_name    = self.config.data.dataset
        cfg        = DATASETS.get(ds_name)
        if cfg is None:
            raise ValueError(f"Unknown dataset: {ds_name!r}")

        # now cfg looks like {"train_bg": "...", "train_t": "...", ...}
        self.train_dataset   = ImageDatasetFlexible(cfg["train"],  transforms["train"])
        self.test_dataset    = ImageDatasetFlexible(cfg["val"],    transforms["test"])
        self.special_dataset_bg = ImageDatasetFlexible(cfg["special_bg"],transforms["test"])
        self.special_dataset_t  = ImageDatasetFlexible(cfg["special_t"], transforms["test"])


    def _setup_dataloaders(self, batch_size):
        self.train_dataloader = InfiniteLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config.model.workers,
            drop_last=True,
            is_infinite=True
        )
        self.test_dataloader = InfiniteLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.model.workers,
            is_infinite=False
        )
        self.special_loader_bg = InfiniteLoader(
            self.special_dataset_bg,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config.model.workers,
            is_infinite=False
        )
        self.special_loader_t = InfiniteLoader(
            self.special_dataset_t,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config.model.workers,
            is_infinite=False
        )    
        self.special_batch_bg = next(self.special_loader_bg).to(self.device).float()
        self.special_batch_t = next(self.special_loader_t).to(self.device).float()

    def _setup_optimizers(self):
        params = list(self.method.inverter.parameters())

        optimizer_args = dict(
            self.config.optimizers[self.config.train.inverter_optimizer]
        )
        optimizer_args["params"] = params
        self.inverter_optimizer = optimizers[self.config.train.inverter_optimizer](
            **optimizer_args
        )

        if self.config.model.checkpoint_path != "":
            ckpt = torch.load(self.config.model.checkpoint_path, map_location="cpu")
            if "inverter_opt" in ckpt.keys():
                self.inverter_optimizer.load_state_dict(ckpt["inverter_opt"])
            else:
                print('WARNING, continuing training without loading inverter optimizer state!')

        if self.config.train.train_dis:
            params = list(self.method.discriminator.parameters())
            optimizer_args = dict(
                self.config.optimizers[self.config.train.disc_optimizer]
            )
            optimizer_args["params"] = params
            self.disc_optimizer = optimizers[self.config.train.disc_optimizer](
                **optimizer_args
            )

            if self.config.model.checkpoint_path != "":
                if "disc_opt" in ckpt.keys():
                    self.disc_optimizer.load_state_dict(ckpt["disc_opt"])
                else:
                    print('WARNING, continuing training without loading disc optimizer state!')

    def _setup_loss(self):
        enc_losses_dict = self.config.encoder_losses
        disc_losses_dict = self.config.disc_losses

        self.loss_builder = LossBuilder(
            enc_losses_dict, 
            disc_losses_dict, 
            self.device
        )

    def _setup_experiment_dir(self):
        base_root = Path(__file__).resolve().parent.parent
        num = 0
        exp_dir = self.config.exp.exp_dir
        exp_dir_name = "{}_{}".format(self.config.exp.name, str(num).zfill(3))

        exp_path = base_root / exp_dir / exp_dir_name
        while True:
            if exp_path.exists():
                num += 1
                exp_dir_name = "{}_{}".format(self.config.exp.name, str(num).zfill(3))
                print(exp_path, "already exists: move to", exp_dir_name)
            else:
                break
            exp_path = base_root / exp_dir / exp_dir_name
        self.experiment_dir = str(exp_path)
        os.makedirs(self.experiment_dir)
        print("Experiment directory: {self.experiment_dir}")

        with open(os.path.join(self.experiment_dir, "config.yaml"), "w") as f:
            omegaconf.OmegaConf.save(config=self.config, f=f.name)

        with open(os.path.join(self.experiment_dir, "run_command.sh"), "w") as f:
            f.write(" ".join(sys.argv))
            f.write("\n")

        self.metrics_dir = os.path.join(self.experiment_dir, "metrics")
        os.mkdir(self.metrics_dir)
        self.inference_results_dir = os.path.join(
            self.experiment_dir, "inference_results"
        )
        os.mkdir(self.inference_results_dir)
        # Directory where you want to save logs
        self.log_dir = os.path.join(self.experiment_dir, "val_logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
    def _setup_metrics(self):
        metrics_names = self.config.train.val_metrics

        self.metrics = []
        for metric_name in metrics_names:
            metric_args = {}
            if hasattr(self.config.metrics, metric_name):
                metric_args = getattr(self.config.metrics, metric_name)
            self.metrics.append(metrics_registry[metric_name](**metric_args))


    def to_train(self):
        self.method.train()

    def to_eval(self):
        self.method.eval()

    def run(self):
        iter_info = StreamingMeans()
        self.to_train()

        for self.global_step in range(self.start_step, self.config.train.steps + 1):
            with Timer(iter_info, "iter_train"):
                loss_dict = self.train_step()
                iter_info.update({f"iter_train/{k}": v for k, v in loss_dict.items()})

            if self.global_step % self.config.train.val_step == 0 :
                with Timer(iter_info, "iter_val"):
                    val_loss_dict = self.validate()

            if self.global_step % self.config.train.log_step == 0:
                self.inference_special()
                self.logger.save_train_logs(iter_info, self.global_step)
                iter_info.clear()
                # Save to file
                save_path = os.path.join(self.log_dir, "training_losses.txt")

                with open(save_path, "a") as f:  # Append mode so you keep all steps
                    line = f"Training at step {self.global_step} | "
                    line += ", ".join(f"{k}: {v:.6f}" for k, v in sorted(loss_dict.items()))
                    f.write(line + "\n")

            if self.global_step % self.config.train.checkpoint_step == 0:
                self.save_checkpoint()

    def train_step(self):
        x  = next(self.train_dataloader)
        x = x.to(self.device).float()
        output = self.forward(x)

        enc_loss, loss_dict = self.loss_builder.encoder_loss(output["encoder"])

        self.inverter_optimizer.zero_grad()
        enc_loss.backward()
        self.inverter_optimizer.step()
        loss_dict["enc_loss"] = float(enc_loss)

        if (
            self.config.train.train_dis
            and self.global_step >= self.config.train.dis_train_start_step
        ):
            if self.global_step == self.config.train.dis_train_start_step:
                print("Start training with discriminator")
            if self.train_dataloader.batch_size != self.config.model.batch_size:
                print(f"Changing batch size from {self.train_dataloader.batch_size} to {self.config.model.batch_size}")
                self.setup_dataloaders(self.config.model.batch_size)

            toogle_grad(self.method.discriminator, True)
            self.method.discriminator.train()

            disc_loss, disc_losses_dict = self.loss_builder.disc_loss(
                self.method.discriminator, 
                output["to_disc"]
                )
            loss_dict.update(disc_losses_dict)

            self.disc_optimizer.zero_grad()
            disc_loss.backward()
            self.disc_optimizer.step()

            toogle_grad(self.method.discriminator, False)
            self.method.discriminator.eval()

        self.method.latent_avg = self.method.latent_avg.detach()

        return loss_dict

    def save_checkpoint(self):
        save_name = f"iteration_{self.global_step}.pt"
        checkpoint_path = os.path.join(self.experiment_dir, save_name)
        save_dict = self.get_save_dict()
        print(f"Saving checkpoint to {checkpoint_path}")
        torch.save(save_dict, checkpoint_path)

        options_path = os.path.join(self.experiment_dir, "save_options.json")
        save_options = {"start_step": self.global_step + 1}

        if self.config.exp.wandb:
            save_options.update(self.logger.wandb_logger.wandb_args)

        with open(options_path, "w") as f:
            json.dump(save_options, f)

    def get_save_dict(self):
        save_dict = {
            "state_dict": self.method.state_dict(),
            "encoder_opt": self.inverter_optimizer.state_dict(),
            "latent_avg": self.method.latent_avg
        }

        if self.config.train.train_dis:
            save_dict["disc_opt"] = self.disc_optimizer.state_dict()
        return save_dict

    @torch.inference_mode()
    def inference_special(self):

        print("Start inference special")

        X = self.special_batch_bg
        Y = self.special_batch_t
        rec_X, rec_X_w = self.forward_reconstruction(X)
        rec_Y, rec_X_w = self.forward_reconstruction(Y)

        idx = self.config.data.special_idx

        row1 = torch.stack([X[idx], rec_X[idx], rec_X_w[idx]], dim=0)
        row2 = torch.stack([Y[idx], rec_Y[idx], rec_X_w[idx]], dim=0)
        # Assuming row1 and row2 are [5, C, H, W]
        columns = [torch.stack([row1[i], row2[i]], dim=0) for i in range(len(row1))]

        visualize_batch_grid(
            image_batches=columns,  # list of [2, C, H, W]
            titles=["Input", "Recon", "Recon_w"],
            save_path=f"{self.log_dir}/images/train_step_{self.global_step}.png"
        )


    @torch.inference_mode()
    def validate(self):
        """Run a quick validation pass over up to 11 batches and return average losses."""
        print("Start validating")

        dataloader = self.test_dataloader
        all_losses = defaultdict(float)
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            # Move to device and cast
            x = batch.to(self.device).float()

            # Forward pass
            output = self.forward(x)

            # Compute encoder loss and other loss terms
            enc_loss, loss_terms = self.loss_builder.encoder_loss(output["encoder"])

            # Convert torch tensors to floats and rename keys
            loss_terms = {f"{k}_x": float(v) for k, v in loss_terms.items()}
            loss_terms["enc_loss"] = float(enc_loss)

            # Accumulate
            for k, v in loss_terms.items():
                all_losses[k] += v

            num_batches += 1
            if batch_idx >= 10:  # stop after 11 batches
                break

        # Compute average loss per term
        avg_losses = {k: v / num_batches for k, v in all_losses.items()}

        # Log and return
        print(f"Validation results over {num_batches} batches:")
        for name, val in avg_losses.items():
            print(f"  {name}: {val:.4f}")

        return avg_losses

        # # Average all accumulated losses
        # averaged_losses = {k: v / num_batches for k, v in all_losses.items()}

        # # Save to file
        # save_path = os.path.join(self.log_dir, "validate_losses.txt")

        # with open(save_path, "a") as f:  # Append mode so you keep all steps
        #     line = f"Validation at step {self.global_step}  |  "
        #     line += ", ".join(f"{k}: {v:.6f}" for k, v in sorted(averaged_losses.items()))
        #     f.write(line + "\n")

        # self.to_train()
        # return averaged_losses

    @torch.inference_mode()
    def forward_reconstruction(self, x):
        y_hat_inv, w_inv, fused_feat, w_feat = self.method(
            x,
            return_latents=True,
            n_iter=self.global_step
        )
                                                
        y_hat_inv_w, _ = self.method.decoder(
            [w_inv],
            input_is_latent=True,
            is_stylespace=False,
            randomize_noise=False
        )

        return y_hat_inv, y_hat_inv_w

@training_runners.add_to_registry(name="fse_inverter")
class FSEInverterTrainingRunner(BaseTrainingRunner):
    def forward(self, x):
        y_hat_inv, w_inv, fused_feat, w_feat = self.method(
            x,
            return_latents=True,
            n_iter=self.global_step
        )
                                                
        y_hat_inv_w, _ = self.method.decoder(
            [w_inv],
            input_is_latent=True,
            is_stylespace=False,
            randomize_noise=False
        )

        y_hat = torch.cat([y_hat_inv, y_hat_inv_w], dim=0)
                                   
        output = {"encoder": {}, "to_disc": {}}
        use_adv_loss = (
            self.config.train.train_dis
            and self.global_step >= self.config.train.dis_train_start_step
        )
        output["encoder"]["use_adv_loss"] = use_adv_loss
        if use_adv_loss:
            output["encoder"]["fake_preds"] = self.method.discriminator(y_hat, None)
            output["to_disc"]["y_hat"] = y_hat
            output["to_disc"]["x"] = x
            output["to_disc"]["step"] = self.global_step
        
        y_hat = self.method.pool(y_hat)
        x = self.method.pool(x)
        x = torch.cat([x, x], dim=0)
        
        output["encoder"]["x"] = x
        output["encoder"]["y_hat"] = y_hat
        output["encoder"]["feat_recon"] = fused_feat
        output["encoder"]["feat_real"] = w_feat

        return output

    def _run_on_batch(self, inputs):
        result_batch = self.method(inputs)
        return result_batch


@training_runners.add_to_registry(name="fse_inverter_cs")
class FSEInverterCSTrainingRunner(BaseTrainingRunner):
    def forward(self, x):
        y_hat_inv, w_inv, fused_feat, w_feat = self.method(
            x,
            return_latents=True,
            n_iter=self.global_step
        )
                                                
        y_hat_inv_w, _ = self.method.decoder(
            [w_inv],
            input_is_latent=True,
            is_stylespace=False,
            randomize_noise=False
        )

        y_hat = torch.cat([y_hat_inv, y_hat_inv_w], dim=0)
                                   
        output = {"encoder": {}, "to_disc": {}}
        use_adv_loss = (
            self.config.train.train_dis
            and self.global_step >= self.config.train.dis_train_start_step
        )
        output["encoder"]["use_adv_loss"] = use_adv_loss
        if use_adv_loss:
            output["encoder"]["fake_preds"] = self.method.discriminator(y_hat, None)
            output["to_disc"]["y_hat"] = y_hat
            output["to_disc"]["x"] = x
            output["to_disc"]["step"] = self.global_step
        
        y_hat = self.method.pool(y_hat)
        x = self.method.pool(x)
        x = torch.cat([x, x], dim=0)
        
        output["encoder"]["x"] = x
        output["encoder"]["y_hat"] = y_hat
        output["encoder"]["feat_recon"] = fused_feat
        output["encoder"]["feat_real"] = w_feat

        return output

    def _run_on_batch(self, inputs):
        result_batch = self.method(inputs)
        return result_batch

