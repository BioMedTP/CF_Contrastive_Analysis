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
from datasets.datasets import ImageDataset
from datasets.loaders import InfiniteLoader
from training.losses import disc_losses, LossBuilder
from training.optimizers import optimizers
from metrics.metrics import metrics_registry

from training.loggers import Timer, StreamingMeans, TrainigLogger
from utils.common_utils import tensor2im, get_keys, visualize_batch_grid
from models.methods import methods_registry

from models.psp.encoders.psp_encoders import ProgressiveStage
from utils.model_utils import toogle_grad


training_runners = ClassRegistry()
        

FACE_DIRECTIONS = {
    "age": [-7, -5, 5, 7, 10],
    "fs_makeup": [5, 8, 12],
    "afro": [0.03, 0.07],
    "angry": [0.06, 0.1],
    "purple_hair": [0.07, 0.1, 0.12],
    "glasses": [-10, -7],
    "face_roundness": [-13, -7, 7, 13], 
    "rotation": [-5.0, -3.0, -1.0, 1.0, 3.0, 5.0],
    "bobcut": [0.07, 0.12, 0.18],
    "bowlcut": [0.07, 0.14],
    "mohawk": [0.07, 0.10],
    "blond hair": [-8, -4, 4, 8],
    "fs_smiling": [-6, -3, 3, 6, 9]
}


def get_random_edit():
    direction = np.random.choice(list(FACE_DIRECTIONS.keys()))
    strenght = np.random.choice(FACE_DIRECTIONS[direction])
    return direction, strenght
        

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
        transform_dict = transforms_registry[self.config.data.transform]().get_transforms()

        self.train_dataset_bg = ImageDataset(self.config.data.input_train_bg_dir, transform_dict["train"])
        self.train_dataset_t = ImageDataset(self.config.data.input_train_t_dir, transform_dict["train"])
        self.test_dataset_bg = ImageDataset(self.config.data.input_val_bg_dir, transform_dict["test"])
        self.test_dataset_t = ImageDataset(self.config.data.input_val_t_dir, transform_dict["test"])   

        self.special_dataset_bg = ImageDataset(self.config.data.special_bg_dir, transform_dict["test"])
        self.special_dataset_t = ImageDataset(self.config.data.special_t_dir, transform_dict["test"])
        

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
            shuffle=False,
            num_workers=self.config.model.workers,
            is_infinite=False
        )
        self.special_loader_t = InfiniteLoader(
            self.special_dataset_t,
            batch_size=batch_size,
            shuffle=False,
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
        x_resh = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
        x_cond_resh = F.interpolate(x_cond, size=(256, 256), mode="bilinear", align_corners=False)

        w_e4e_x = self.method.e4e_encoder(x_resh) + self.method.latent_avg
        w_e4e_x_cond = self.method.e4e_encoder(x_cond_resh) + self.method.latent_avg

        e4e_c_x, e4e_s_x = self.method.cs_model(w_e4e_x)
        e4e_c_y, e4e_s_y = self.method.cs_model(w_e4e_x_cond)

        x_E, fx_e4e = self.method.decoder(
            [e4e_c_x + e4e_s_x],
            input_is_latent=True,
            randomize_noise=False,
            return_latents=False,
            return_features=True
        )

        y_E, fy_e4e = self.method.decoder(
            [e4e_c_x + e4e_s_y], 
            is_stylespace=False,
            input_is_latent=True,
            randomize_noise=False,
            return_features=True
        )

        delta = fx_e4e[9] - fy_e4e[9]   
        shift = e4e_s_y - e4e_c_x

        return delta, shift  

    @torch.inference_mode()
    def swap_by_delta(self, x, x_cond, n_iter=1e5):
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)

        delta, shift = self.calculate_delta(x, x_cond)

        w_recon, predicted_feat = self.method.inverter.fs_backbone(x)
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

        return images


    @torch.inference_mode()
    def inference_special(self):

        print("Start inference special")

        X = self.special_batch_bg
        Y = self.special_batch_t

        result_batch_bg = self._run_on_batch(X)
        result_batch_t = self._run_on_batch(Y)

        swap_X2Y = self.swap_by_delta(X, Y)
        swap_Y2X = self.swap_by_delta(Y, X)

        idx = self.config.data.special_idx
        visualize_batch_grid(
                image_batches = [
                torch.stack([X[idx], Y[idx]], dim=0), 
                torch.stack([result_batch_bg[idx], result_batch_t[idx]], dim=0),
                torch.stack([swap_X2Y[idx], swap_Y2X[idx]], dim=0)], 
                titles=["Input", "Recon", "Swap"],
                save_path=f"{self.log_dir}/images/train_step_{self.global_step}.png")


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

            if batch_idx >= 4:
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
class FSEEditorTrainingRunner(BaseTrainingRunner):
    def forward(self, x, x_cond):
        # get inversion batch
        y_hat_inv, w, fused_feat, w_feat = self.method(x, return_latents=True)

        # get editing batch
        with torch.no_grad():
            # sample X_E as training input and X'_E as training target
            x_resh = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
            x_cond_resh = F.interpolate(x_cond, size=(256, 256), mode="bilinear", align_corners=False)

            w_e4e_x = self.method.e4e_encoder(x_resh) + self.method.latent_avg
            w_e4e_x_cond = self.method.e4e_encoder(x_cond_resh) + self.method.latent_avg

            e4e_c_x, e4e_s_x = self.method.cs_model(w_e4e_x)
            _, e4e_s_y = self.method.cs_model(w_e4e_x_cond)

            x_E, fx_e4e = self.method.decoder(
                [e4e_c_x + e4e_s_x],
                input_is_latent=True,
                randomize_noise=False,
                return_latents=False,
                return_features=True
            )

            y_E, fy_e4e = self.method.decoder(
                [e4e_c_x + e4e_s_y], 
                is_stylespace=False,
                input_is_latent=True,
                randomize_noise=False,
                return_features=True
            )

            y_E_256 = F.interpolate(y_E, size=(256, 256), mode="bilinear", align_corners=False) # X'_E
            x_E_256 = F.interpolate(x_E, size=(256, 256), mode="bilinear", align_corners=False) # X_E
            delta = fx_e4e[9] - fy_e4e[9]

            x_E_fused_feat, w_x_E_edited, x_E_is_stylespace = self.sfe_feat_forward(x_E_256)
        

        to_feature_editor = torch.cat([x_E_fused_feat, delta], dim=1)
        x_E_edited_feat = self.method.encoder(to_feature_editor)
        x_E_edited_feats = [None] * 9 + [x_E_edited_feat] + [None] * (17 - 9)

        y_hat_edit, _ = self.method.decoder(
            w_x_E_edited,
            input_is_latent=True,
            new_features=x_E_edited_feats,
            feature_scale=1.0,
            is_stylespace=x_E_is_stylespace,
            randomize_noise=False
        )
        
        bs = x_resh.size(0)
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
            # if self.config.train.train_domain_disc:
            #     output["encoder"]["fake_preds"] = self.method.discriminator(y_hat_inv, None)
            #     output["to_disc"]["y_hat"] = y_hat_inv
            #     output["to_disc"]["x"] = x 
            #     output["to_disc"]["step"] = self.global_step                


        x = torch.cat([x, y_E], dim=0)
        y_hat = torch.cat([y_hat_inv, y_hat_edit])
        
        y_hat = self.method.pool(y_hat)
        x = self.method.pool(x)
        output["encoder"]["x"] = x
        output["encoder"]["y_hat"] = y_hat

        return output
    
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

@training_runners.add_to_registry(name="fse_editor_cs_double")
class FSEEditorTrainingRunner(BaseTrainingRunner):
    def forward(self, x, x_cond):
        # get inversion batch
        y_hat_inv, w, fused_feat, w_feat = self.method(x, return_latents=True)

        # get editing batch
        with torch.no_grad():
            # sample X_E as training input and X'_E as training target
            x_resh = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
            x_cond_resh = F.interpolate(x_cond, size=(256, 256), mode="bilinear", align_corners=False)

            w_e4e_x = self.method.e4e_encoder(x_resh) + self.method.latent_avg
            w_e4e_x_cond = self.method.e4e_encoder(x_cond_resh) + self.method.latent_avg

            e4e_c_x, e4e_s_x = self.method.w_cs_model(w_e4e_x)
            _, e4e_s_y = self.method.w_cs_model(w_e4e_x_cond)
    
            x_E, fx_e4e = self.method.decoder(
                [e4e_c_x + e4e_s_x],
                input_is_latent=True,
                randomize_noise=False,
                return_latents=False,
                return_features=True
            )

            y_E, fy_e4e = self.method.decoder(
                [e4e_c_x + e4e_s_y], 
                is_stylespace=False,
                input_is_latent=True,
                randomize_noise=False,
                return_features=True
            )

            y_E_256 = F.interpolate(y_E, size=(256, 256), mode="bilinear", align_corners=False) # X'_E
            x_E_256 = F.interpolate(x_E, size=(256, 256), mode="bilinear", align_corners=False) # X_E
            x_E_delta = fx_e4e[9] - fy_e4e[9]
            y_E_delta = - x_E_delta

            ##### first way train: X_E ---> sfe (edit with delta) ---> X'_E    loss = X'_E - Y_E
            x_E_fused_feat, w_x_E_edited, x_E_is_stylespace = self.sfe_feat_forward(x_E_256)
            y_E_fused_feat, w_y_E_edited, y_E_is_stylespace = self.sfe_feat_forward(y_E_256)
        
        x_E_to_feature_editor = torch.cat([x_E_fused_feat, x_E_delta], dim=1)
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

        y_E_to_feature_editor = torch.cat([y_E_fused_feat, y_E_delta], dim=1)
        y_E_edited_feat = self.method.encoder(y_E_to_feature_editor)
        y_E_edited_feats = [None] * 9 + [y_E_edited_feat] + [None] * (17 - 9)        

        x_hat_edit, _ = self.method.decoder(
            w_y_E_edited,
            input_is_latent=True,
            new_features=y_E_edited_feats,
            feature_scale=1.0,
            is_stylespace=y_E_is_stylespace,
            randomize_noise=False
        )

        bs = x_resh.size(0)
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

        x = torch.cat([x, y_E, x_E], dim=0)
        y_hat = torch.cat([y_hat_inv, y_hat_edit, x_hat_edit])
        
        y_hat = self.method.pool(y_hat)
        x = self.method.pool(x)
        output["encoder"]["x"] = x
        output["encoder"]["y_hat"] = y_hat

        return output

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



@training_runners.add_to_registry(name="fse_editor_cs_cycle")
class FSEEditorTrainingRunner(BaseTrainingRunner):
    def forward(self, x, x_cond):
        # get inversion batch
        y_hat_inv, w, fused_feat, w_feat = self.method(x, return_latents=True)

        # get editing batch
        with torch.no_grad():
            # sample X_E as training input and X'_E as training target
            x_resh = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
            x_cond_resh = F.interpolate(x_cond, size=(256, 256), mode="bilinear", align_corners=False)

            w_e4e_x = self.method.e4e_encoder(x_resh) + self.method.latent_avg
            w_e4e_x_cond = self.method.e4e_encoder(x_cond_resh) + self.method.latent_avg

            e4e_c_x, e4e_s_x = self.method.cs_model(w_e4e_x)
            _, e4e_s_y = self.method.cs_model(w_e4e_x_cond)
    
            x_E, fx_e4e = self.method.decoder(
                [e4e_c_x + e4e_s_x],
                input_is_latent=True,
                randomize_noise=False,
                return_latents=False,
                return_features=True
            )

            y_E, fy_e4e = self.method.decoder(
                [e4e_c_x + e4e_s_y], 
                is_stylespace=False,
                input_is_latent=True,
                randomize_noise=False,
                return_features=True
            )

            # y_E_256 = F.interpolate(y_E, size=(256, 256), mode="bilinear", align_corners=False) # X'_E
            x_E_256 = F.interpolate(x_E, size=(256, 256), mode="bilinear", align_corners=False) # X_E
            x_E_delta = fx_e4e[9] - fy_e4e[9]
            y_E_delta = - x_E_delta

            ##### first way train: X_E ---> sfe (edit with delta) ---> X'_E    loss = X'_E - Y_E
            x_E_fused_feat, w_x_E_edited, x_E_is_stylespace = self.sfe_feat_forward(x_E_256)
            
        
        x_E_to_feature_editor = torch.cat([x_E_fused_feat, x_E_delta], dim=1)
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
        with torch.no_grad():
            y_E_256 = F.interpolate(y_hat_edit, size=(256, 256), mode="bilinear", align_corners=False) # X'_E
            y_E_fused_feat, w_y_E_edited, y_E_is_stylespace = self.sfe_feat_forward(y_E_256)

        y_E_to_feature_editor = torch.cat([y_E_fused_feat, y_E_delta], dim=1)
        y_E_edited_feat = self.method.encoder(y_E_to_feature_editor)
        y_E_edited_feats = [None] * 9 + [y_E_edited_feat] + [None] * (17 - 9)        

        x_hat_edit, _ = self.method.decoder(
            w_y_E_edited,
            input_is_latent=True,
            new_features=y_E_edited_feats,
            feature_scale=1.0,
            is_stylespace=y_E_is_stylespace,
            randomize_noise=False
        )

        bs = x_resh.size(0)
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

        x = torch.cat([x, y_E, x_E], dim=0)
        y_hat = torch.cat([y_hat_inv, y_hat_edit, x_hat_edit])
        
        y_hat = self.method.pool(y_hat)
        x = self.method.pool(x)
        output["encoder"]["x"] = x
        output["encoder"]["y_hat"] = y_hat

        return output

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
    

@training_runners.add_to_registry(name="fse_editor_pSp_cs")
class FSEEditorTrainingRunner(BaseTrainingRunner):
    def forward(self, x, x_cond):
        # get inversion batch
        y_hat_inv, w, fused_feat, w_feat = self.method(x, return_latents=True)

        # get editing batch
        with torch.no_grad():
            # sample X_E as training input and X'_E as training target
            x_resh = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
            x_cond_resh = F.interpolate(x_cond, size=(256, 256), mode="bilinear", align_corners=False)

            _, w_pSp_x = self.method.pSp_encoder.forward(x_resh, return_latents=True)
            _, w_pSp_x_cond = self.method.pSp_encoder.forward(x_cond_resh, return_latents=True)

            e4e_c_x, e4e_s_x = self.method.cs_model(w_pSp_x)
            _, e4e_s_y = self.method.cs_model(w_pSp_x_cond)
    
            x_E, fx_e4e = self.method.decoder(
                [e4e_c_x + e4e_s_x],
                input_is_latent=True,
                randomize_noise=False,
                return_latents=False,
                return_features=True
            )

            y_E, fy_e4e = self.method.decoder(
                [e4e_c_x + e4e_s_y], 
                is_stylespace=False,
                input_is_latent=True,
                randomize_noise=False,
                return_features=True
            )

            y_E_256 = F.interpolate(y_E, size=(256, 256), mode="bilinear", align_corners=False) # X'_E
            x_E_256 = F.interpolate(x_E, size=(256, 256), mode="bilinear", align_corners=False) # X_E
            x_E_delta = fx_e4e[9] - fy_e4e[9]
            y_E_delta = - x_E_delta

            ##### first way train: X_E ---> sfe (edit with delta) ---> X'_E    loss = X'_E - Y_E
            x_E_fused_feat, w_x_E_edited, x_E_is_stylespace = self.sfe_feat_forward(x_E_256)
            y_E_fused_feat, w_y_E_edited, y_E_is_stylespace = self.sfe_feat_forward(y_E_256)
        
        x_E_to_feature_editor = torch.cat([x_E_fused_feat, x_E_delta], dim=1)
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

        y_E_to_feature_editor = torch.cat([y_E_fused_feat, y_E_delta], dim=1)
        y_E_edited_feat = self.method.encoder(y_E_to_feature_editor)
        y_E_edited_feats = [None] * 9 + [y_E_edited_feat] + [None] * (17 - 9)        

        x_hat_edit, _ = self.method.decoder(
            w_y_E_edited,
            input_is_latent=True,
            new_features=y_E_edited_feats,
            feature_scale=1.0,
            is_stylespace=y_E_is_stylespace,
            randomize_noise=False
        )

        bs = x_resh.size(0)
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

        x = torch.cat([x, y_E, x_E], dim=0)
        y_hat = torch.cat([y_hat_inv, y_hat_edit, x_hat_edit])
        
        y_hat = self.method.pool(y_hat)
        x = self.method.pool(x)
        output["encoder"]["x"] = x
        output["encoder"]["y_hat"] = y_hat

        return output

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