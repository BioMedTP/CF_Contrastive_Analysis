import os
import torch
import torch.nn.functional as F

def write_log_to_txt(log_msg, results_dir, filename):
    with open(f"{results_dir}/logs/{filename}", 'a') as f:
        f.write(log_msg)

def train_common_disc(model, pSp_net, cs_mlp_net,
                      train_bg_dataloader, train_t_dataloader,
                      test_bg_dataloader, test_t_dataloader,
                      criterion, optimizer, opts, device, global_step):
    model.train()
    cs_mlp_net.eval()

    device_type = "cuda" if torch.cuda.is_available() else "cpu"


    log_dir = f"{opts.exp_dir}/logs"
    ckpt_dir = f"{opts.exp_dir}/checkpoints"

    scaler = torch.cuda.amp.GradScaler()
    gradient_accumulation_steps = 8
    print(f"training discriminator at steps {global_step}")

    for epoch in range(opts.max_disc_epochs):
        total_loss, correct, total = 0.0, 0, 0
        optimizer.zero_grad()

        for step, (batch_bg, batch_t) in enumerate(zip(train_bg_dataloader, train_t_dataloader)):
            x_bg, _ = batch_bg
            x_t, _ = batch_t

            x_bg = x_bg.to(device).float()
            x_t = x_t.to(device).float()

            with torch.no_grad():
                _, w_bg_pSp = pSp_net.forward(x_bg, return_latents=True)
                _, w_t_pSp = pSp_net.forward(x_t, return_latents=True)

                latent_bg_c, _ = cs_mlp_net(w_bg_pSp)
                latent_t_c, _ = cs_mlp_net(w_t_pSp)

            labels_bg = torch.zeros(latent_bg_c.size(0), 1, device=device)
            labels_t = torch.ones(latent_t_c.size(0), 1, device=device)

            with torch.cuda.amp.autocast():
                logits_bg = model(latent_bg_c)
                logits_t = model(latent_t_c)

                loss_bg = criterion(logits_bg, labels_bg)
                loss_t = criterion(logits_t, labels_t)
                loss = loss_bg + loss_t

            scaler.scale(loss).backward()

            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_bg_dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # Accuracy & Loss
            all_logits = torch.cat([logits_bg, logits_t], dim=0)
            all_labels = torch.cat([labels_bg, labels_t], dim=0)

            preds = (all_logits > 0.5).float()
            correct += (preds == all_labels).sum().item()
            total += all_labels.size(0)
            total_loss += loss.item() / gradient_accumulation_steps  # Normalize per accumulation

        train_loss = total_loss / len(train_bg_dataloader)
        train_acc = correct / total

        # Validation
        if (epoch + 1) % opts.disc_val_interval == 0 or (epoch + 1) == opts.max_disc_epochs:
            val_loss, val_acc = validate(model, cs_mlp_net, pSp_net, test_bg_dataloader, test_t_dataloader, criterion, device)
            log_msg = (f"Epoch [{epoch+1}/{opts.max_disc_epochs}] | "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}\n")
        else:
            log_msg = (f"Epoch [{epoch+1}/{opts.max_disc_epochs}] | "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}\n")

        write_log_to_txt(log_msg, log_dir, "disc_train_val_loss.txt")

        # if (epoch + 1) % opts.disc_save_interval == 0 or (epoch + 1) == opts.max_disc_epochs:
        #     checkpoint_path = f"{ckpt_dir}/disc_epoch_{epoch+1}.pth"
        #     torch.save(model.state_dict(), checkpoint_path)
        #     print(f"Checkpoint saved: {checkpoint_path}")
    print(f"finish training discriminator at steps {global_step}")
    model.eval()
    cs_mlp_net.train()

@torch.no_grad()
def validate(model, cs_mlp_net, pSp_net, test_bg_dataloader, test_t_dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    for (batch_bg, batch_t) in zip(test_bg_dataloader, test_t_dataloader):
        x_bg, _ = batch_bg
        x_t, _ = batch_t

        x_bg = x_bg.to(device).float()
        x_t = x_t.to(device).float()

        _, w_bg_pSp = pSp_net.forward(x_bg, return_latents=True)
        _, w_t_pSp = pSp_net.forward(x_t, return_latents=True)

        latent_bg_c, _ = cs_mlp_net(w_bg_pSp)
        latent_t_c, _ = cs_mlp_net(w_t_pSp)

        labels_bg = torch.zeros(latent_bg_c.size(0), 1, device=device)
        labels_t = torch.ones(latent_t_c.size(0), 1, device=device)

        with torch.cuda.amp.autocast():
            logits_bg = model(latent_bg_c)
            logits_t = model(latent_t_c)

            loss_bg = criterion(logits_bg, labels_bg)
            loss_t = criterion(logits_t, labels_t)
            loss = loss_bg + loss_t

        all_logits = torch.cat([logits_bg, logits_t], dim=0)
        all_labels = torch.cat([labels_bg, labels_t], dim=0)

        preds = (all_logits > 0.5).float()
        correct += (preds == all_labels).sum().item()
        total += all_labels.size(0)
        total_loss += loss.item()

    val_loss = total_loss / len(test_bg_dataloader)
    val_acc = correct / total
    return val_loss, val_acc


def calc_adv_loss(
    disc_net,
    latent_bg_c,
    latent_t_c,
    opts,
    step=None,
    # epoch=None
):
    """
    Computes adversarial loss and conditionally logs it to a file based on opts.log_interval.

    Args:
        disc_net: Discriminator network
        latent_bg_c: Latent code from background input
        latent_t_c: Latent code from target input
        opts: Namespace or object with attributes:
              - adv_loss_type: 'JS_confusion', 'non_satu', or 'confusion_mse'
              - adv_lambda
              - exp_dir (path for logs)
              - log_interval (optional)
        step: Current training step (optional, for controlling logging)
        epoch: Current epoch (optional, for logging)

    Returns:
        final_loss: weighted adversarial loss (tensor)
        loss_dict: dictionary of individual loss terms
    """
    loss_dict = {}
    adv_loss_type = opts.adv_loss_type
    adv_lambda = opts.adv_lambda
    latent_bg_c = latent_bg_c.view(latent_bg_c.size(0), -1)  # [B, 16*512 = 8192]
    latent_t_c  = latent_t_c.view(latent_t_c.size(0), -1)    # Same

    if adv_loss_type == "JS_confusion":
        logits_bg = disc_net(latent_bg_c)
        logits_t = disc_net(latent_t_c)

        probs_bg = torch.sigmoid(logits_bg)
        probs_t = torch.sigmoid(logits_t)

        eps = 1e-6
        loss_adv_bg = - (torch.log(probs_bg + eps) + torch.log(1 - probs_bg + eps))
        loss_adv_t = - (torch.log(probs_t + eps) + torch.log(1 - probs_t + eps))

        loss_adv = (loss_adv_bg.mean() + loss_adv_t.mean()) / 2
        loss_dict['loss_adv_bg'] = float(loss_adv_bg.mean())
        loss_dict['loss_adv_t'] = float(loss_adv_t.mean())

    elif adv_loss_type == "non_satu":
        logits_bg = disc_net(latent_bg_c)
        target_label = torch.ones_like(logits_bg)
        loss_adv = F.binary_cross_entropy_with_logits(logits_bg, target_label)
        loss_dict['loss_adv'] = float(loss_adv)

    elif adv_loss_type == "confusion_mse":
        logits_bg = disc_net(latent_bg_c)
        logits_t = disc_net(latent_t_c)

        probs_bg = torch.sigmoid(logits_bg)
        probs_t = torch.sigmoid(logits_t)

        target = torch.full_like(probs_bg, 0.5)
        loss_adv_bg = F.mse_loss(probs_bg, target)
        loss_adv_t = F.mse_loss(probs_t, target)
        loss_adv = (loss_adv_bg + loss_adv_t) / 2

        loss_dict['loss_adv_probs_bg'] = float(loss_adv_bg)
        loss_dict['loss_adv_probs_t'] = float(loss_adv_t)

    else:
        raise ValueError(f"Unknown adv_loss_type: {adv_loss_type}")

    # Final scaled loss
    loss_dict['loss_adv'] = float(loss_adv)
    final_loss = loss_adv * adv_lambda
    loss_dict['loss'] = float(final_loss)

    log_msg = ""
    if step is not None:
        log_msg += f"Step {step}: "
    log_msg += ", ".join([f"{k}={v:.6f}" for k, v in loss_dict.items()]) + "\n"

    return final_loss, log_msg


def write_log_to_txt(log_msg, results_dir, filename):
    os.makedirs(os.path.join(results_dir, "logs"), exist_ok=True)
    with open(os.path.join(results_dir, "logs", filename), 'a') as f:
        f.write(log_msg)


