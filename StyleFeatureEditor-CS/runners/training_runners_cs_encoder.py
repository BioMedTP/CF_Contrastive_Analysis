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

from training.loggers import Timer, StreamingMeans, TrainigLogger
from utils.common_utils import tensor2im, get_keys, visualize_batch_grid
from models.methods import methods_registry

from models.psp.encoders.psp_encoders import ProgressiveStage
from utils.model_utils import toogle_grad
from configs.data_paths import DATASETS
import random
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
        self.train_dataset_bg   = ImageDatasetFlexible(cfg["train_bg"],  transforms["train"])
        self.train_dataset_t    = ImageDatasetFlexible(cfg["train_t"],   transforms["train"])
        self.test_dataset_bg    = ImageDatasetFlexible(cfg["val_bg"],    transforms["test"])
        self.test_dataset_t     = ImageDatasetFlexible(cfg["val_t"],     transforms["test"])
        self.special_dataset_bg = ImageDatasetFlexible(cfg["special_bg"],transforms["test"])
        self.special_dataset_t  = ImageDatasetFlexible(cfg["special_t"], transforms["test"])
        
    def _setup_dataloaders(self, batch_size):
        self.train_loader_bg = InfiniteLoader(
            self.train_dataset_bg,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config.model.workers,
            drop_last=True,
            is_infinite=True
        )
        self.train_loader_t = InfiniteLoader(
            self.train_dataset_t,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config.model.workers,
            drop_last=True,
            is_infinite=True
        )
        self.test_loader_bg = InfiniteLoader(
            self.test_dataset_bg,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.model.workers,
            is_infinite=False
        )
        self.test_loader_t = InfiniteLoader(
            self.test_dataset_t,
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
        params = list(self.method.encoder.parameters())

        optimizer_args = dict(
            self.config.optimizers[self.config.train.encoder_optimizer]
        )
        optimizer_args["params"] = params
        self.encoder_optimizer = optimizers[self.config.train.encoder_optimizer](
            **optimizer_args
        )

        if self.config.model.checkpoint_path != "":
            ckpt = torch.load(self.config.model.checkpoint_path, map_location="cpu")
            if "encoder_opt" in ckpt.keys():
                self.encoder_optimizer.load_state_dict(ckpt["encoder_opt"])
            else:
                print('WARNING, continuing training without loading encoder optimizer state!')

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
        print(f"Experiment directory: {self.experiment_dir}")

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
                if self.config.train.direction == "x_to_y":
                    loss_dict = self.train_step_dirX2Y()
                elif self.config.train.direction == "y_to_x":
                    loss_dict = self.train_step_dirY2X()
                elif self.config.train.direction == "two_directions":
                    loss_dict = self.train_step_two_directions()
                else:
                    raise ValueError(
                        f"Unknown training direction type: {self.config.train.direction}"
                    )
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

    def train_step_dirX2Y(self):
        x = next(self.train_loader_bg)
        y = next(self.train_loader_t)
        x, y = x.to(self.device).float(), y.to(self.device).float()
        
        output = self.forward(x, y)

        enc_loss, loss_dict = self.loss_builder.encoder_loss(output["encoder"])

        self.encoder_optimizer.zero_grad()
        enc_loss.backward()
        self.encoder_optimizer.step()
        loss_dict["enc_loss"] = float(enc_loss)

        if (
            self.config.train.train_dis
            and self.global_step >= self.config.train.dis_train_start_step
        ):
            if self.global_step == self.config.train.dis_train_start_step:
                print("Start training with discriminator")
            if self.train_loader_bg.batch_size != self.config.model.batch_size:
                print(f"Changing batch size from {self.train_loader_bg.batch_size} to {self.config.model.batch_size}")
                self._setup_dataloaders(self.config.model.batch_size)

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

    def train_step_dirY2X(self):
        x = next(self.train_loader_bg)
        y = next(self.train_loader_t)
        x, y = x.to(self.device).float(), y.to(self.device).float()
        
        output = self.forward(y, x)

        enc_loss, loss_dict = self.loss_builder.encoder_loss(output["encoder"])

        self.encoder_optimizer.zero_grad()
        enc_loss.backward()
        self.encoder_optimizer.step()
        loss_dict["enc_loss"] = float(enc_loss)

        if (
            self.config.train.train_dis
            and self.global_step >= self.config.train.dis_train_start_step
        ):
            if self.global_step == self.config.train.dis_train_start_step:
                print("Start training with discriminator")
            if self.train_loader_bg.batch_size != self.config.model.batch_size:
                print(f"Changing batch size from {self.train_loader_bg.batch_size} to {self.config.model.batch_size}")
                self._setup_dataloaders(self.config.model.batch_size)

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

    def train_step_two_directions(self):
        x = next(self.train_loader_bg)
        y = next(self.train_loader_t)
        x, y = x.to(self.device).float(), y.to(self.device).float()

        loss_dict = {}

        for direction, (x_in, y_in) in zip(["xy", "yx"], [(x, y), (y, x)]):
            output = self.forward(x_in, y_in)

            # Encoder loss and optimization
            enc_loss, enc_loss_dict = self.loss_builder.encoder_loss(output["encoder"])
            self.encoder_optimizer.zero_grad()
            enc_loss.backward()
            self.encoder_optimizer.step()

            # Record encoder losses
            loss_dict[f"enc_loss_{direction}"] = float(enc_loss)
            loss_dict.update({f"{k}_{direction}": float(v) for k, v in enc_loss_dict.items()})

            # Discriminator training
            if (
                self.config.train.train_dis
                and self.global_step >= self.config.train.dis_train_start_step
            ):
                if self.global_step == self.config.train.dis_train_start_step:
                    print("Start training with discriminator")
                if self.train_loader_bg.batch_size != self.config.model.batch_size:
                    print(f"Changing batch size from {self.train_loader_bg.batch_size} to {self.config.model.batch_size}")
                    self._setup_dataloaders(self.config.model.batch_size)

                toogle_grad(self.method.discriminator, True)
                self.method.discriminator.train()

                disc_loss, disc_losses_dict = self.loss_builder.disc_loss(self.method.discriminator, output["to_disc"])
                self.disc_optimizer.zero_grad()
                disc_loss.backward()
                self.disc_optimizer.step()

                # Record discriminator losses
                for k, v in disc_losses_dict.items():
                    loss_dict[f"{k}_{direction}"] = float(v)

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
            "encoder_opt": self.encoder_optimizer.state_dict(),
            "latent_avg": self.method.latent_avg
        }

        if self.config.train.train_dis:
            save_dict["disc_opt"] = self.disc_optimizer.state_dict()
        return save_dict
    
    @torch.inference_mode()
    def calculate_delta(self, x, x_cond):
        # sample X_E as training input and X'_E as training target
        if self.config.model.w_space_encoder == "pSp":
            _, w_x = self.method.pSp_encoder(x, return_latents=True)
            _, w_x_cond = self.method.pSp_encoder(x_cond, return_latents=True)
            c_x, s_x = self.method.pSp_cs_model(w_x)
            c_y, s_y = self.method.pSp_cs_model(w_x_cond)

        elif self.config.model.w_space_encoder == "e4e":
            w_x = self.method.e4e_encoder(x) + self.method.latent_avg
            w_x_cond = self.method.e4e_encoder(x_cond) + self.method.latent_avg
            c_x, s_x = self.method.e4e_cs_model(w_x)
            c_y, s_y = self.method.e4e_cs_model(w_x_cond)

        else:
            raise ValueError("Unsupported w_space_encoder") 

        encoder_recon, fx = self.method.decoder(
            [c_x + s_x],
            input_is_latent=True,
            randomize_noise=False,
            return_latents=False,
            return_features=True
        )

        encoder_swap, fy = self.method.decoder(
            [c_x + s_y], 
            is_stylespace=False,
            input_is_latent=True,
            randomize_noise=False,
            return_features=True
        )

        delta = fx[9] - fy[9]
        
        return delta, encoder_recon, encoder_swap

    @torch.inference_mode()
    def calculate_delta_s1s2(self, x, x_cond):
        # sample X_E as training input and X'_E as training target
        
        _, w_x = self.method.pSp_encoder(x, return_latents=True)
        _, w_x_cond = self.method.pSp_encoder(x_cond, return_latents=True)

        c_x, s1_x, s2_x = self.method.pSp_cs_model(w_x)
        c_y, s1_y, s2_y  = self.method.pSp_cs_model(w_x_cond)


        encoder_recon, fx = self.method.decoder(
            [c_x + s1_x + s2_x],
            input_is_latent=True,
            randomize_noise=False,
            return_latents=False,
            return_features=True
        )

        encoder_swap, fy = self.method.decoder(
            [c_x + s1_y + s2_y], 
            is_stylespace=False,
            input_is_latent=True,
            randomize_noise=False,
            return_features=True
        )

        delta = fx[9] - fy[9]
        
        return delta, encoder_recon, encoder_swap

    @torch.inference_mode()
    def swap_by_delta(self, x, x_cond, n_iter=1e5):
        x_resh = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
        x_cond_resh = F.interpolate(x_cond, size=(256, 256), mode="bilinear", align_corners=False)

        if self.config.train.train_runner=="fse_editor_cs1s2":
            delta, encoder_recon, encoder_swap = self.calculate_delta_s1s2(x_resh, x_cond_resh)
        else:
            delta, encoder_recon, encoder_swap = self.calculate_delta(x_resh, x_cond_resh)

        w_recon, predicted_feat = self.method.inverter.fs_backbone(x_resh)
        w_recon = w_recon + self.method.latent_avg
                
        _, w_feats = self.method.decoder(
            [w_recon],
            input_is_latent=True,
            return_features=True,
            is_stylespace=False,
            randomize_noise=False,
            early_stop=64
        )

        w_feat = w_feats[9]  # bs x 512 x 64 x 64 
        
        fused_feat = self.method.inverter.fuser(torch.cat([predicted_feat, w_feat], dim=1))
        if delta is None:
            delta = torch.zeros_like(fused_feat)  # inversion case
        
        edited_feat = self.method.encoder(torch.cat([fused_feat, delta], dim=1))
        feats = [None] * 9 + [edited_feat] + [None] * (17 - 9)

        images, _ = self.method.decoder(
            [w_recon],
            input_is_latent=True,
            return_features=True,
            new_features=feats,
            feature_scale=min(1.0, 0.0001 * n_iter),
            is_stylespace=False,
            randomize_noise=False
        )

        return images, encoder_recon, encoder_swap


    @torch.inference_mode()
    def inference_special(self):

        print("Start inference special")

        X = self.special_batch_bg
        Y = self.special_batch_t

        rec_X = self.method(X)
        rec_Y = self.method(Y)

        swap_X2Y, pSp_rec_X, pSp_swap_X2Y = self.swap_by_delta(X, Y)
        swap_Y2X, pSp_rec_Y, pSp_swap_Y2X = self.swap_by_delta(Y, X)

        # idx = self.config.data.special_idx
        if self.config.data.special_idx >= 0:
            idx = self.config.data.special_idx
        else:
            idx = random.randint(0, X.size(0) - 1)
        row1 = torch.stack([X[idx], pSp_rec_X[idx], pSp_swap_X2Y[idx], rec_X[idx], swap_X2Y[idx]], dim=0)
        row2 = torch.stack([Y[idx], pSp_rec_Y[idx], pSp_swap_Y2X[idx], rec_Y[idx], swap_Y2X[idx]], dim=0)
        # Assuming row1 and row2 are [5, C, H, W]
        columns = [torch.stack([row1[i], row2[i]], dim=0) for i in range(len(row1))]

        visualize_batch_grid(
            image_batches=columns,  # list of [2, C, H, W]
            titles=["Input", "pSp-cs Recon", "pSp-cs Swap", "Refined Recon", "Refined Swap"],
            save_path=f"{self.log_dir}/images/train_step_{self.global_step}.png"
        )


    @torch.inference_mode()
    def validate(self):

        print("Start validating")

        self.to_eval()

        dataloader_bg = self.test_loader_bg
        dataloader_t = self.test_loader_t

        all_losses = defaultdict(float)
        num_batches = 0

        for batch_idx, (batch_bg, batch_t) in enumerate(tqdm(zip(dataloader_bg, dataloader_t))):

            x = batch_bg.to(self.device).float()
            y = batch_t.to(self.device).float()

            # Forward model
            output_x = self.forward(x, y)
            output_y = self.forward(y, x)

            # Encoder loss only
            enc_loss_x, loss_dict_x = self.loss_builder.encoder_loss(output_x["encoder"])
            enc_loss_y, loss_dict_y = self.loss_builder.encoder_loss(output_y["encoder"])

            # Combine per-batch loss dict
            loss_dict = {}
            loss_dict.update({f"{k}_x": v for k, v in loss_dict_x.items()})
            loss_dict.update({f"{k}_y": v for k, v in loss_dict_y.items()})
            loss_dict["enc_loss_x"] = float(enc_loss_x)
            loss_dict["enc_loss_y"] = float(enc_loss_y)
            loss_dict["enc_loss"] = float((enc_loss_x + enc_loss_y) / 2)

            # Accumulate losses
            for k, v in loss_dict.items():
                all_losses[k] += float(v)

            num_batches += 1

            if batch_idx >= 10:
                break

        # Average all accumulated losses
        averaged_losses = {k: v / num_batches for k, v in all_losses.items()}

        # Save to file
        save_path = os.path.join(self.log_dir, "validate_losses.txt")

        with open(save_path, "a") as f:  # Append mode so you keep all steps
            line = f"Validation at step {self.global_step}  |  "
            line += ", ".join(f"{k}: {v:.6f}" for k, v in sorted(averaged_losses.items()))
            f.write(line + "\n")

        self.to_train()
        return averaged_losses



@training_runners.add_to_registry(name="fse_editor_cs")
class FSEEditorCSTrainingRunner(BaseTrainingRunner):
    def forward(self, x, x_cond):
        # get inversion batch
        y_hat_inv, w, fused_feat, w_feat = self.method(x, return_latents=True)

        # get editing batch
        with torch.no_grad():
            # sample X_E as training input and X'_E as training target
            x_resh = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
            x_cond_resh = F.interpolate(x_cond, size=(256, 256), mode="bilinear", align_corners=False)

            c_x, s_x, c_y, s_y = self.encoder_cs_forward(x_resh, x_cond_resh)

            x_E, fx = self.method.decoder(
                [c_x + s_x],
                input_is_latent=True,
                randomize_noise=False,
                return_latents=False,
                return_features=True
            )

            y_E, fy = self.method.decoder(
                [c_x + s_y], 
                is_stylespace=False,
                input_is_latent=True,
                randomize_noise=False,
                return_features=True
            )
            
            x_E_256 = F.interpolate(x_E, size=(256, 256), mode="bilinear", align_corners=False) # X_E

            delta = fx[9] - fy[9]

            ##### first way train: X_E ---> sfe (edit with delta) ---> X'_E    loss = X'_E - Y_E
            x_E_fused_feat, w_x_E_edited, x_E_is_stylespace = self.sfe_feat_forward(x_E_256)
        
        x_E_to_feature_editor = torch.cat([x_E_fused_feat, delta], dim=1)
        x_E_edited_feat = self.method.encoder(x_E_to_feature_editor)
        x_E_edited_feats = [None] * 9 + [x_E_edited_feat] + [None] * (17 - 9)

        y_hat_edit, _ = self.method.decoder(
            w_x_E_edited,
            input_is_latent=True,
            new_features=x_E_edited_feats,
            feature_scale=1.0,
            is_stylespace=x_E_is_stylespace,
            randomize_noise=False
        )

        #bs = x_resh.size(0)
        output = {"encoder": {}, "to_disc": {}}
        use_adv_loss = (
            self.config.train.train_dis
            and self.global_step >= self.config.train.dis_train_start_step
        )
        output["encoder"]["use_adv_loss"] = use_adv_loss

        if use_adv_loss:
            output["encoder"]["fake_preds"] = self.method.discriminator(y_hat_inv, None)
            output["to_disc"]["y_hat"] = y_hat_inv
            output["to_disc"]["x"] = x 
            output["to_disc"]["step"] = self.global_step

        x = torch.cat([x, y_E], dim=0)
        y_hat = torch.cat([y_hat_inv, y_hat_edit])
        
        y_hat = self.method.pool(y_hat)
        x = self.method.pool(x)
        output["encoder"]["x"] = x
        output["encoder"]["y_hat"] = y_hat

        return output

    def encoder_cs_forward(self, x, x_cond):

        if self.config.model.w_space_encoder == "pSp":
            _, w_x = self.method.pSp_encoder(x, return_latents=True)
            _, w_x_cond = self.method.pSp_encoder(x_cond, return_latents=True)
            c_x, s_x = self.method.pSp_cs_model(w_x)
            c_y, s_y = self.method.pSp_cs_model(w_x_cond)

        elif self.config.model.w_space_encoder == "e4e":
            w_x = self.method.e4e_encoder(x) + self.method.latent_avg
            w_x_cond = self.method.e4e_encoder(x_cond) + self.method.latent_avg
            c_x, s_x = self.method.e4e_cs_model(w_x)
            c_y, s_y = self.method.e4e_cs_model(w_x_cond)

        else:
            raise ValueError("Unsupported w_space_encoder") 
        return c_x, s_x, c_y, s_y

    def sfe_feat_forward(self, x_256):

        w, predicted_feats = self.method.inverter.fs_backbone(x_256)
        w = w + self.method.latent_avg

        w_edited = w  
        
        is_stylespace = isinstance(w_edited, tuple)

        if not is_stylespace:
            if isinstance(w_edited, (list, tuple)):
                w_edited = [torch.cat(w_edited, dim=0)]
            else:
                w_edited = [w_edited]

        _, w_feats = self.method.decoder(
            [w],
            input_is_latent=True,
            return_features=True,
            is_stylespace=False,
            randomize_noise=False,
            early_stop=64
        )
        w_feats = w_feats[9] 

        to_fuser = torch.cat([predicted_feats, w_feats], dim=1)
        fused_feat = self.method.inverter.fuser(to_fuser)

        return fused_feat, w_edited, is_stylespace

    def _run_on_batch(self, inputs):
        result_batch = self.method(inputs)
        return result_batch
    

@training_runners.add_to_registry(name="fse_editor_cs1s2")
class FSEEditorCS1S2TrainingRunner(BaseTrainingRunner):
    def forward(self, x, x_cond):
        # get inversion batch
        y_hat_inv, w, fused_feat, w_feat = self.method(x, return_latents=True)

        # get editing batch
        with torch.no_grad():
            # sample X_E as training input and X'_E as training target
            x_resh = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
            x_cond_resh = F.interpolate(x_cond, size=(256, 256), mode="bilinear", align_corners=False)

            orn_latent, edit_latent = self.encoder_cs_forward(x_resh, x_cond_resh)

            x_E, fx = self.method.decoder(
                [orn_latent],
                input_is_latent=True,
                randomize_noise=False,
                return_latents=False,
                return_features=True
            )

            y_E, fy = self.method.decoder(
                [edit_latent], 
                is_stylespace=False,
                input_is_latent=True,
                randomize_noise=False,
                return_features=True
            )
            
            x_E_256 = F.interpolate(x_E, size=(256, 256), mode="bilinear", align_corners=False) # X_E

            delta = fx[9] - fy[9]

            ##### first way train: X_E ---> sfe (edit with delta) ---> X'_E    loss = X'_E - Y_E
            x_E_fused_feat, w_x_E_edited, x_E_is_stylespace = self.sfe_feat_forward(x_E_256)
        
        x_E_to_feature_editor = torch.cat([x_E_fused_feat, delta], dim=1)
        x_E_edited_feat = self.method.encoder(x_E_to_feature_editor)
        x_E_edited_feats = [None] * 9 + [x_E_edited_feat] + [None] * (17 - 9)

        y_hat_edit, _ = self.method.decoder(
            w_x_E_edited,
            input_is_latent=True,
            new_features=x_E_edited_feats,
            feature_scale=1.0,
            is_stylespace=x_E_is_stylespace,
            randomize_noise=False
        )

        #bs = x_resh.size(0)
        output = {"encoder": {}, "to_disc": {}}
        use_adv_loss = (
            self.config.train.train_dis
            and self.global_step >= self.config.train.dis_train_start_step
        )
        output["encoder"]["use_adv_loss"] = use_adv_loss

        if use_adv_loss:
            output["encoder"]["fake_preds"] = self.method.discriminator(y_hat_inv, None)
            output["to_disc"]["y_hat"] = y_hat_inv
            output["to_disc"]["x"] = x 
            output["to_disc"]["step"] = self.global_step

        x = torch.cat([x, y_E], dim=0)
        y_hat = torch.cat([y_hat_inv, y_hat_edit])
        
        y_hat = self.method.pool(y_hat)
        x = self.method.pool(x)
        output["encoder"]["x"] = x
        output["encoder"]["y_hat"] = y_hat

        return output

    def encoder_cs_forward(self, x, x_cond):

        _, w_x = self.method.pSp_encoder(x, return_latents=True)
        _, w_x_cond = self.method.pSp_encoder(x_cond, return_latents=True)
        c_x, s1_x, s2_x = self.method.pSp_cs_model(w_x)
        c_y, s1_y, s2_y  = self.method.pSp_cs_model(w_x_cond)

        return c_x + s1_x + s2_x, c_x + s1_y + s2_y


    def sfe_feat_forward(self, x_256):

        w, predicted_feats = self.method.inverter.fs_backbone(x_256)
        w = w + self.method.latent_avg

        w_edited = w  
        
        is_stylespace = isinstance(w_edited, tuple)

        if not is_stylespace:
            if isinstance(w_edited, (list, tuple)):
                w_edited = [torch.cat(w_edited, dim=0)]
            else:
                w_edited = [w_edited]

        _, w_feats = self.method.decoder(
            [w],
            input_is_latent=True,
            return_features=True,
            is_stylespace=False,
            randomize_noise=False,
            early_stop=64
        )
        w_feats = w_feats[9] 

        to_fuser = torch.cat([predicted_feats, w_feats], dim=1)
        fused_feat = self.method.inverter.fuser(to_fuser)

        return fused_feat, w_edited, is_stylespace

    def _run_on_batch(self, inputs):
        result_batch = self.method(inputs)
        return result_batch


# @training_runners.add_to_registry(name="fse_editor_cs_double")
# class FSEEditorCSDoubleTrainingRunner(BaseTrainingRunner):
#     def forward(self, x, x_cond):
#         # get inversion batch
#         y_hat_inv, w, fused_feat, w_feat = self.method(x, return_latents=True)

#         # get editing batch
#         with torch.no_grad():
#             # sample X_E as training input and X'_E as training target
#             x_resh = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
#             x_cond_resh = F.interpolate(x_cond, size=(256, 256), mode="bilinear", align_corners=False)
            
#             c_x, s_x, c_y, s_y = self.encoder_cs_forward(x_resh, x_cond_resh)

#             x_E, fx = self.method.decoder(
#                 [c_x + s_x],
#                 input_is_latent=True,
#                 randomize_noise=False,
#                 return_latents=False,
#                 return_features=True
#             )

#             y_E, fy = self.method.decoder(
#                 [c_x + s_y], 
#                 is_stylespace=False,
#                 input_is_latent=True,
#                 randomize_noise=False,
#                 return_features=True
#             )
            
#             x_E_256 = F.interpolate(x_E, size=(256, 256), mode="bilinear", align_corners=False) # X_E
#             y_E_256 = F.interpolate(y_E, size=(256, 256), mode="bilinear", align_corners=False) # X'_E

#             delta = fx[9] - fy[9]
#             delta_inv = - delta

#             ##### first way train: X_E ---> sfe (edit with delta) ---> X'_E    loss = X'_E - Y_E
#             x_E_fused_feat, w_x_E_edited, x_E_is_stylespace = self.sfe_feat_forward(x_E_256)
#             y_E_fused_feat, w_y_E_edited, y_E_is_stylespace = self.sfe_feat_forward(y_E_256)
        
#         x_E_to_feature_editor = torch.cat([x_E_fused_feat, delta], dim=1)
#         x_E_edited_feat = self.method.encoder(x_E_to_feature_editor)
#         x_E_edited_feats = [None] * 9 + [x_E_edited_feat] + [None] * (17 - 9)

#         y_hat_edit, _ = self.method.decoder(
#             w_x_E_edited,
#             input_is_latent=True,
#             new_features=x_E_edited_feats,
#             feature_scale=1.0,
#             is_stylespace=x_E_is_stylespace,
#             randomize_noise=False
#         )

#         y_E_to_feature_editor = torch.cat([y_E_fused_feat, delta_inv], dim=1)
#         y_E_edited_feat = self.method.encoder(y_E_to_feature_editor)
#         y_E_edited_feats = [None] * 9 + [y_E_edited_feat] + [None] * (17 - 9)        

#         x_hat_edit, _ = self.method.decoder(
#             w_y_E_edited,
#             input_is_latent=True,
#             new_features=y_E_edited_feats,
#             feature_scale=1.0,
#             is_stylespace=y_E_is_stylespace,
#             randomize_noise=False
#         )

#         #bs = x_resh.size(0)
#         output = {"encoder": {}, "to_disc": {}}
#         use_adv_loss = (
#             self.config.train.train_dis
#             and self.global_step >= self.config.train.dis_train_start_step
#         )
#         output["encoder"]["use_adv_loss"] = use_adv_loss

#         if use_adv_loss:
#             output["encoder"]["fake_preds"] = self.method.discriminator(y_hat_inv, None)
#             output["to_disc"]["y_hat"] = y_hat_inv
#             output["to_disc"]["x"] = x 
#             output["to_disc"]["step"] = self.global_step

#         x = torch.cat([x, y_E, x_E], dim=0)
#         y_hat = torch.cat([y_hat_inv, y_hat_edit, x_hat_edit])
        
#         y_hat = self.method.pool(y_hat)
#         x = self.method.pool(x)
#         output["encoder"]["x"] = x
#         output["encoder"]["y_hat"] = y_hat

#         return output

#     def encoder_cs_forward(self, x, x_cond):
#         if self.config.model.w_space_encoder == "pSp":
#             _, w_x = self.method.pSp_encoder(x, return_latents=True)
#             _, w_x_cond = self.method.pSp_encoder(x_cond, return_latents=True)
#             c_x, s_x = self.method.pSp_cs_model(w_x)
#             c_y, s_y = self.method.pSp_cs_model(w_x_cond)

#         elif self.config.model.w_space_encoder == "e4e":
#             w_x = self.method.e4e_encoder(x) + self.method.latent_avg
#             w_x_cond = self.method.e4e_encoder(x_cond) + self.method.latent_avg
#             c_x, s_x = self.method.e4e_cs_model(w_x)
#             c_y, s_y = self.method.e4e_cs_model(w_x_cond)

#         else:
#             raise ValueError("Unsupported w_space_encoder") 
#         return c_x, s_x, c_y, s_y


#     def sfe_feat_forward(self, x_256):

#         w, predicted_feats = self.method.inverter.fs_backbone(x_256)
#         w = w + self.method.latent_avg

#         w_edited = w  
        
#         is_stylespace = isinstance(w_edited, tuple)

#         if not is_stylespace:
#             if isinstance(w_edited, (list, tuple)):
#                 w_edited = [torch.cat(w_edited, dim=0)]
#             else:
#                 w_edited = [w_edited]

#         _, w_feats = self.method.decoder(
#             [w],
#             input_is_latent=True,
#             return_features=True,
#             is_stylespace=False,
#             randomize_noise=False,
#             early_stop=64
#         )
#         w_feats = w_feats[9] 

#         to_fuser = torch.cat([predicted_feats, w_feats], dim=1)
#         fused_feat = self.method.inverter.fuser(to_fuser)

#         return fused_feat, w_edited, is_stylespace

#     def _run_on_batch(self, inputs):
#         result_batch = self.method(inputs)
#         return result_batch



# @training_runners.add_to_registry(name="fse_editor_cs_cycle")
# class FSEEditorCSCycleTrainingRunner(BaseTrainingRunner):
#     def forward(self, x, x_cond):
#         # get inversion batch
#         y_hat_inv, w, fused_feat, w_feat = self.method(x, return_latents=True)

#         # get editing batch
#         with torch.no_grad():
#             # sample X_E as training input and X'_E as training target
#             x_resh = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
#             x_cond_resh = F.interpolate(x_cond, size=(256, 256), mode="bilinear", align_corners=False)

#             c_x, s_x, c_y, s_y = self.encoder_cs_forward(x_resh, x_cond_resh)

#             x_E, fx = self.method.decoder(
#                 [c_x + s_x],
#                 input_is_latent=True,
#                 randomize_noise=False,
#                 return_latents=False,
#                 return_features=True
#             )

#             y_E, fy = self.method.decoder(
#                 [c_x + s_y], 
#                 is_stylespace=False,
#                 input_is_latent=True,
#                 randomize_noise=False,
#                 return_features=True
#             )
            
#             x_E_256 = F.interpolate(x_E, size=(256, 256), mode="bilinear", align_corners=False) # X_E
#             # y_E_256 = F.interpolate(y_E, size=(256, 256), mode="bilinear", align_corners=False) # X'_E

#             delta = fx[9] - fy[9]
#             delta_inv = - delta

#             ##### first way train: X_E ---> sfe (edit with delta) ---> X'_E    loss = X'_E - Y_E
#             x_E_fused_feat, w_x_E_edited, x_E_is_stylespace = self.sfe_feat_forward(x_E_256)
#             #y_E_fused_feat, w_y_E_edited, y_E_is_stylespace = self.sfe_feat_forward(y_E_256)
        
#         x_E_to_feature_editor = torch.cat([x_E_fused_feat, delta], dim=1)
#         x_E_edited_feat = self.method.encoder(x_E_to_feature_editor)
#         x_E_edited_feats = [None] * 9 + [x_E_edited_feat] + [None] * (17 - 9)

#         y_hat_edit, _ = self.method.decoder(
#             w_x_E_edited,
#             input_is_latent=True,
#             new_features=x_E_edited_feats,
#             feature_scale=1.0,
#             is_stylespace=x_E_is_stylespace,
#             randomize_noise=False
#         )

#         inv_x_edit = self.cycle_forward(x, delta)

#         #bs = x_resh.size(0)
#         output = {"encoder": {}, "to_disc": {}}
#         use_adv_loss = (
#             self.config.train.train_dis
#             and self.global_step >= self.config.train.dis_train_start_step
#         )
#         output["encoder"]["use_adv_loss"] = use_adv_loss

#         if use_adv_loss:
#             output["encoder"]["fake_preds"] = self.method.discriminator(y_hat_inv, None)
#             output["to_disc"]["y_hat"] = y_hat_inv
#             output["to_disc"]["x"] = x 
#             output["to_disc"]["step"] = self.global_step

#         x = torch.cat([x, y_E, x], dim=0)
#         y_hat = torch.cat([y_hat_inv, y_hat_edit, inv_x_edit])
        
#         y_hat = self.method.pool(y_hat)
#         x = self.method.pool(x)
#         output["encoder"]["x"] = x
#         output["encoder"]["y_hat"] = y_hat

#         return output
    
#     def encoder_cs_forward(self, x, x_cond):
#         if self.config.model.w_space_encoder == "pSp":
#             _, w_x = self.method.pSp_encoder(x, return_latents=True)
#             _, w_x_cond = self.method.pSp_encoder(x_cond, return_latents=True)
#             c_x, s_x = self.method.pSp_cs_model(w_x)
#             c_y, s_y = self.method.pSp_cs_model(w_x_cond)

#         elif self.config.model.w_space_encoder == "e4e":
#             w_x = self.method.e4e_encoder(x) + self.method.latent_avg
#             w_x_cond = self.method.e4e_encoder(x_cond) + self.method.latent_avg
#             c_x, s_x = self.method.e4e_cs_model(w_x)
#             c_y, s_y = self.method.e4e_cs_model(w_x_cond)

#         else:
#             raise ValueError("Unsupported w_space_encoder") 
#         return c_x, s_x, c_y, s_y
        
#     def cycle_forward(self, x, delta):
#         with torch.no_grad():
#             x_resh = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
#             fused_feat, w_edited, is_stylespace = self.sfe_feat_forward(x_resh)

#             to_feature_editor = torch.cat([fused_feat, delta], dim=1)
#             edited_feat = self.method.encoder(to_feature_editor)
#             edited_feats = [None] * 9 + [edited_feat] + [None] * (17 - 9)

#             x_edit, _ = self.method.decoder(
#                 w_edited,
#                 input_is_latent=True,
#                 new_features=edited_feats,
#                 feature_scale=1.0,
#                 is_stylespace=is_stylespace,
#                 randomize_noise=False
#             )

#             x_edit_resh = F.interpolate(x_edit, size=(256, 256), mode="bilinear", align_corners=False)
#             inv_fused_feat, w_inv_edited, inv_is_stylespace = self.sfe_feat_forward(x_edit_resh)

#         inv_to_feature_editor = torch.cat([inv_fused_feat, -delta], dim=1)
#         inv_edited_feat = self.method.encoder(inv_to_feature_editor)
#         inv_edited_feats = [None] * 9 + [inv_edited_feat] + [None] * (17 - 9)

#         inv_x_edit, _ = self.method.decoder(
#             w_inv_edited,
#             input_is_latent=True,
#             new_features=inv_edited_feats,
#             feature_scale=1.0,
#             is_stylespace=inv_is_stylespace,
#             randomize_noise=False
#         )

#         return inv_x_edit
    
#     def sfe_feat_forward(self, x_256):

#         w, predicted_feats = self.method.inverter.fs_backbone(x_256)
#         w = w + self.method.latent_avg

#         w_edited = w  
        
#         is_stylespace = isinstance(w_edited, tuple)

#         if not is_stylespace:
#             if isinstance(w_edited, (list, tuple)):
#                 w_edited = [torch.cat(w_edited, dim=0)]
#             else:
#                 w_edited = [w_edited]

#         _, w_feats = self.method.decoder(
#             [w],
#             input_is_latent=True,
#             return_features=True,
#             is_stylespace=False,
#             randomize_noise=False,
#             early_stop=64
#         )
#         w_feats = w_feats[9] 

#         to_fuser = torch.cat([predicted_feats, w_feats], dim=1)
#         fused_feat = self.method.inverter.fuser(to_fuser)

#         return fused_feat, w_edited, is_stylespace

#     def _run_on_batch(self, inputs):
#         result_batch = self.method(inputs)
#         return result_batch
    
