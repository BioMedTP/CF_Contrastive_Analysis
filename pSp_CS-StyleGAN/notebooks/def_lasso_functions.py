import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from models.mlp3D import MappingNetwork_cs_sparsity, EqualizedLinear
import matplotlib.pyplot as plt


def L1_lasso_output_regularization(output, reg_type='row'):
    if reg_type == 'row':
        return torch.sum(torch.abs(output).sum(dim=2))
    elif reg_type == 'column':
        return torch.sum(torch.abs(output).sum(dim=1))
    elif reg_type == 'element':
        return torch.sum(torch.abs(output))
    else:
        raise ValueError(f"Invalid regularization type: '{reg_type}'.")

def L1_lasso_weight_regularization(model, network='net_s', reg_type='all'):
    """
    Apply L1 regularization on the specified layers of either `net_s` or `net_c` in the model.

    Args:
        model (MappingNetwork_cs_sparsity): The model with the network to regularize.
        network (str): The network to apply Lasso regularization ('net_s' or 'net_c').
        reg_type (str): Type of weight regularization ('all' or 'last').
            - 'all': Applies Lasso to all EqualizedLinear layers in the specified network.
            - 'last': Applies Lasso only to the last EqualizedLinear layer in the specified network.

    Returns:
        torch.Tensor: Computed L1 regularization penalty for the selected layers.
    """
    lasso_weight_penalty = 0.0
    
    # Select the target network based on the network argument
    target_network = getattr(model, network, None)
    if target_network is None:
        raise ValueError(f"Invalid network '{network}'. Use 'net_s' or 'net_c'.")

    # Get layers of the selected network
    layers = list(target_network.children())

    if reg_type == 'all':
        # Apply Lasso to all EqualizedLinear layers in the target network
        for layer in layers:
            if isinstance(layer[0], EqualizedLinear):
                # Apply L1 regularization across all style dimensions in the 3D weight tensor
                lasso_weight_penalty += torch.sum(torch.abs(layer[0].weight.weight))  # Access the 3D weight tensor
    elif reg_type == 'last':
        # Apply Lasso only to the last EqualizedLinear layer in the target network
        last_layer = layers[-1][0]
        if isinstance(last_layer, EqualizedLinear):
            # Apply L1 regularization across all style dimensions in the 3D weight tensor
            lasso_weight_penalty += torch.sum(torch.abs(last_layer.weight.weight))  # Access the 3D weight tensor
    else:
        raise ValueError(f"Invalid reg_type '{reg_type}'. Use 'all' or 'last'.")

    return lasso_weight_penalty


def calculate_output_sparsity(output, threshold=0.0):
    """
    Calculate sparsity of output activations.

    Args:
        output (torch.Tensor): The output tensor for which sparsity is calculated.
        threshold (float): Threshold below which values are considered zero.

    Returns:
        sparsity (float): The fraction of elements that are effectively zero.
        zero_count (int): Number of zero-valued elements.
        total_count (int): Total number of elements.
    """
    zero_count = (output.abs() <= threshold).sum().item()   #If threshold=0.0, only values that are exactly zero will be counted as sparse. For any other threshold > 0, values with absolute magnitude less than or equal to the threshold will be considered sparse
    total_count = output.numel()
    sparsity = zero_count / total_count if total_count > 0 else 0
    return sparsity, zero_count, total_count

def calculate_weight_sparsity(model, network='net_s', reg_type='all', threshold=0.0):
    zero_count, total_count = 0, 0
    target_network = getattr(model, network, None)
    if target_network is None:
        raise ValueError(f"Invalid network '{network}'. Use 'net_s' or 'net_c'.")

    layers = list(target_network.children())
    
    if reg_type == 'all':
        for layer in layers:
            if isinstance(layer[0], EqualizedLinear):
                weights = layer[0].weight.weight
                zero_count += (weights.abs() <= threshold).sum().item()
                total_count += weights.numel()
    elif reg_type == 'last':
        last_layer = layers[-1][0]
        if isinstance(last_layer, EqualizedLinear):
            weights = last_layer.weight.weight
            zero_count += (weights.abs() <= threshold).sum().item()
            total_count += weights.numel()
    else:
        raise ValueError(f"Invalid reg_type '{reg_type}'. Use 'all' or 'last'.")
    sparsity = zero_count / total_count if total_count > 0 else 0
    return sparsity, zero_count, total_count

def calculate_latent_loss(latent_bg_s, latent_t_s, latent_bg, latent_t, latent_bg_target, latent_t_target, 
                          lasso_sbg_lambda=0.01, lasso_st_lambda=0.01, lasso_weight_lambda=0.01, model=None, 
                          lasso_output_type='row', lasso_weight_type='all', network='net_s'):
    """
    Calculate total loss, including L1 regularization for output and weights.

    Args:
        latent_bg_s: Background latent tensor.
        latent_t_s: Target latent tensor.
        latent_bg: Background tensor.
        latent_t: Target tensor.
        latent_bg_target: Target tensor for background.
        latent_t_target: Target tensor for output.
        lasso_sbg_lambda: Weight for sbg L1 regularization.
        lasso_st_lambda: Weight for st L1 regularization.
        lasso_weight_lambda: Weight for weight L1 regularization.
        model: The model on which to apply regularization.
        lasso_output_type: Type of L1 regularization for output ('row', 'column', 'element').
        lasso_weight_type: Type of L1 regularization for weights ('all', 'last').
        network: Specifies 'net_s' or 'net_c' for weight regularization.

    Returns:
        total_loss: Computed total loss.
        loss_dict: Dictionary with individual loss components.
    """
    loss_silent_bg = F.mse_loss(latent_bg_s, torch.zeros_like(latent_bg_s))
    loss_distance_bg = F.mse_loss(latent_bg, latent_bg_target)
    loss_distance_t = F.mse_loss(latent_t, latent_t_target)
    
    lasso_sbg_loss = lasso_sbg_lambda * L1_lasso_output_regularization(latent_bg_s, reg_type=lasso_output_type) 
    lasso_st_loss = lasso_st_lambda * L1_lasso_output_regularization(latent_t_s, reg_type=lasso_output_type) 
    lasso_weight_loss = (
        lasso_weight_lambda * L1_lasso_weight_regularization(model, network=network, reg_type=lasso_weight_type) 
        if model else 0.0
    )
    
    total_loss = loss_silent_bg + loss_distance_bg + loss_distance_t + lasso_sbg_loss + lasso_st_loss + lasso_weight_loss
    
    # Ensure .item() is only called on tensors
    loss_dict = {
        'loss_silent_bg': loss_silent_bg.item() if isinstance(loss_silent_bg, torch.Tensor) else loss_silent_bg,
        'loss_distance_bg': loss_distance_bg.item() if isinstance(loss_distance_bg, torch.Tensor) else loss_distance_bg,
        'loss_distance_t': loss_distance_t.item() if isinstance(loss_distance_t, torch.Tensor) else loss_distance_t,
        'lasso_sbg_loss': lasso_sbg_loss.item() if isinstance(lasso_sbg_loss, torch.Tensor) else lasso_sbg_loss,
        'lasso_st_loss': lasso_st_loss.item() if isinstance(lasso_st_loss, torch.Tensor) else lasso_st_loss,
        'lasso_weight_loss': lasso_weight_loss.item() if isinstance(lasso_weight_loss, torch.Tensor) else lasso_weight_loss,
        'total_loss': total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
    }

    return total_loss, loss_dict

def plot_results(weight_sparsity_tracker, sbg_sparsity_tracker, st_sparsity_tracker, 
                 lasso_sbg_tracker, lasso_st_tracker, lasso_weight_tracker, latent_MSE_tracker):
    
    plt.figure(figsize=(10, 6))
    plt.plot(weight_sparsity_tracker, label="weight")
    plt.title("Weight Sparsity Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Weight Sparsity")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(sbg_sparsity_tracker, label="Silent bg")
    plt.title("S-bg Sparsity Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Silent background Sparsity")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(st_sparsity_tracker, label="Silent t")
    plt.title("S-t Sparsity Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Silent target Sparsity")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(lasso_sbg_tracker, label="lasso sbg loss")
    plt.title("Lasso Sbg Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(lasso_st_tracker, label="lasso st loss")
    plt.title("Lasso St Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(lasso_weight_tracker, label="weight loss")
    plt.title("Lasso Weight Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(latent_MSE_tracker, label="loss")
    plt.title("MSE Reconstruction Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def show_latent_map(latent_matrix, img_idx=0, show_height=18, show_weight=512):
    if latent_matrix.dim() == 3:
        show_matrix = latent_matrix[img_idx, :show_height, :show_weight]
    else:
        show_matrix = latent_matrix[:show_height, :show_weight]

    show_matrix = show_matrix.cpu().numpy() if show_matrix.is_cuda else show_matrix.numpy()

    plt.figure(figsize=(12, 4))
    plt.imshow(show_matrix, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.show()


def visualize_weight_matrix(model):
    """
    Visualize the weight matrices (512x512) in each `EqualizedLinear` layer of the model.
    """
    # Loop over each layer to identify and visualize EqualizedLinear layers
    for name, layer in model.named_modules():
        if isinstance(layer, EqualizedLinear):
            # Extract weight matrix (assuming it’s shaped 512x512)
            weight_matrix = layer.weight().detach().cpu().numpy()
            
            # Plot the weight matrix as a heatmap
            plt.figure(figsize=(6, 6))
            plt.imshow(weight_matrix, cmap='viridis', aspect='auto')
            plt.colorbar()
            plt.title(f"Weight Matrix in Layer: {name}")
            plt.xlabel("Input Dimension (512)")
            plt.ylabel("Output Dimension (512)")
            plt.show()

def get_model_weights_info(model):
    """
    Collects information about all weights in the model, including their name, shape, 
    and total number of parameters.
    
    Args:
        model (nn.Module): The trained model (e.g., MappingNetwork_cs_sparsity).
        
    Returns:
        weights_info (dict): A dictionary with detailed information about each weight tensor.
    """
    weights_info = {}

    for name, param in model.named_parameters():
        if param.requires_grad:  # Only include learnable parameters
            weight_shape = param.shape
            num_params = param.numel()  # Total number of parameters in the tensor
            
            weights_info[name] = {
                "shape": weight_shape,
                "num_params": num_params
            }
    
    return weights_info


def validate(model, latent_bg_target=None, latent_t_target=None, track_weight_threshold=0.0, track_output_threshold=0.0):
    model.eval()
    
    # Determine the device of the model
    device = next(model.parameters()).device
    
    # Create latent targets on the same device as the model
    if latent_bg_target is None:
        latent_bg_target = torch.randn(model.opts.batch_size, model.opts.style_dim, model.opts.latent_dim, device=device)
    if latent_t_target is None:
        latent_t_target = torch.randn(model.opts.batch_size, model.opts.style_dim, model.opts.latent_dim, device=device)

    with torch.no_grad():
        # Pass inputs through the model
        latent_bg_c, latent_bg_s = model(latent_bg_target, zero_out_silent=model.opts.zero_out_bg)
        latent_t_c, latent_t_s = model(latent_t_target, zero_out_silent=model.opts.zero_out_t)
        #print(model.opts.zero_out_threshold)
        # Calculate weight and output sparsity
        weight_sparsity, weight_zero_count, weight_total_count = calculate_weight_sparsity(model, threshold=track_weight_threshold)
        sbg_sparsity, sbg_zero_count, sbg_total_count = calculate_output_sparsity(latent_bg_s, threshold=track_output_threshold)
        st_sparsity, st_zero_count, st_total_count = calculate_output_sparsity(latent_t_s, threshold=track_output_threshold)

        print("Testing results:")
        print(f"Weight Sparsity: {weight_sparsity:.4f} ({weight_zero_count} of {weight_total_count}), "
              f"Sbg Sparsity: {sbg_sparsity:.4f} ({sbg_zero_count} of {sbg_total_count}), "
              f"St Sparsity: {st_sparsity:.4f} ({st_zero_count} of {st_total_count})\n")
        return latent_bg_s, latent_t_s

class Options:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
def train_model_function(batch_size=4, num_epochs=100, learning_rate=0.01, style_dim=18, latent_dim=512, n_layers_mlp=8, mlp_norm_type='no_dim', 
                         zero_out_bg=False, zero_out_t=False, zero_out_threshold=0.0, zero_out_type='hard', 
                         lasso_sbg_lambda=0.01, lasso_st_lambda=0.01, lasso_output_type='element', 
                         lasso_weight_lambda=0.0, lasso_weight_type='all', 
                         track_output_threshold=0.0, track_weight_threshold=1e-4,
                         plot_curves=False, device='cpu'):
    
    # Gather all function parameters, including 'device'
    opts_params = locals()

    # Initialize options
    opts = Options(**opts_params)

    print(f"Using device: {opts.device}")

    
    model = MappingNetwork_cs_sparsity(opts).to(opts.device)
    
    # Initialize random latent tensors and move to device
    latent_bg_target = torch.randn(opts.batch_size, opts.style_dim, opts.latent_dim).to(opts.device)
    latent_t_target = torch.randn(opts.batch_size, opts.style_dim, opts.latent_dim).to(opts.device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opts.learning_rate)

    # Trackers for results
    weight_sparsity_tracker = []
    sbg_sparsity_tracker = []
    st_sparsity_tracker = []
    lasso_sbg_tracker = []
    lasso_st_tracker = []
    lasso_weight_tracker = []
    latent_MSE_tracker = []

    print(f"\nTraining with zero_out_threshold = {opts.zero_out_threshold}, lasso_sbg_lambda = {opts.lasso_sbg_lambda}, lasso_st_lambda = {opts.lasso_st_lambda}, lasso_weight_lambda = {opts.lasso_weight_lambda}")
    
    for epoch in range(opts.num_epochs):
        model.train()

        # latent_bg_input = torch.randn(opts.batch_size, opts.style_dim, opts.latent_dim).to(device)
        # latent_t_input = torch.randn(opts.batch_size, opts.style_dim, opts.latent_dim).to(device)
            
        # Forward pass
        latent_bg_c, latent_bg_s = model(latent_bg_target, zero_out_silent=opts.zero_out_bg)
        latent_t_c, latent_t_s = model(latent_t_target, zero_out_silent=opts.zero_out_t)

        latent_bg = latent_bg_c
        latent_t = latent_t_c + latent_t_s

        # Loss computation
        loss, loss_dict = calculate_latent_loss(
            latent_bg_s, latent_t_s, latent_bg, latent_t, latent_bg_target, latent_t_target, 
            lasso_sbg_lambda=opts.lasso_sbg_lambda, lasso_st_lambda=opts.lasso_st_lambda, lasso_weight_lambda=opts.lasso_weight_lambda, 
            model=model, lasso_output_type=opts.lasso_output_type, lasso_weight_type=opts.lasso_weight_type
        )
        
        # Backward pass and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track weight sparsity
        weight_sparsity, weight_zero_count, weight_total_count = calculate_weight_sparsity(model, threshold=opts.track_weight_threshold)
        weight_sparsity_tracker.append(weight_sparsity)

        # Track output sparsity for latent_t_s
        sbg_sparsity, sbg_zero_count, sbg_total_count = calculate_output_sparsity(latent_bg_s, threshold=opts.track_output_threshold)
        st_sparsity, st_zero_count, st_total_count = calculate_output_sparsity(latent_t_s, threshold=opts.track_output_threshold)
        
        sbg_sparsity_tracker.append(sbg_sparsity)
        st_sparsity_tracker.append(st_sparsity)

        # Track total loss
        lasso_sbg_tracker.append(loss_dict['lasso_sbg_loss'])
        lasso_st_tracker.append(loss_dict['lasso_st_loss'])
        lasso_weight_tracker.append(loss_dict['lasso_weight_loss'])
        mse_loss = loss_dict['loss_silent_bg'] + loss_dict['loss_distance_bg'] + loss_dict['loss_distance_t']
        latent_MSE_tracker.append(mse_loss)

        if epoch % (opts.num_epochs // 10) == 0:
            print(f"Epoch [{epoch + 1}/{opts.num_epochs}], MSE loss: {mse_loss:.4f}, " 
                  f"Lasso sbg loss: {loss_dict['lasso_sbg_loss']:.4f}, "
                  f"Lasso st loss: {loss_dict['lasso_st_loss']:.4f}, " 
                  f"Lasso weight loss: {loss_dict['lasso_weight_loss']:.4f}, "
                  f"Weight Sparsity: {weight_sparsity:.4f} ({weight_zero_count} of {weight_total_count}), "
                  f"Sbg Sparsity: {sbg_sparsity:.4f} ({sbg_zero_count} of {sbg_total_count}), "
                  f"St Sparsity: {st_sparsity:.4f} ({st_zero_count} of {st_total_count})")
            
    if plot_curves:
      # Plot results 
      plot_results(weight_sparsity_tracker, sbg_sparsity_tracker, st_sparsity_tracker, 
                 lasso_sbg_tracker, lasso_st_tracker, lasso_weight_tracker, latent_MSE_tracker)
      
    trackers = {'weight_sparsity_tracker': weight_sparsity_tracker,
                'sbg_sparsity_tracker': sbg_sparsity_tracker,
                'st_sparsity_tracker': st_sparsity_tracker,
                'lasso_sbg_tracker': lasso_sbg_tracker,
                'lasso_st_tracker': lasso_st_tracker,
                'lasso_weight_tracker': lasso_weight_tracker,
                'latent_MSE_tracker': latent_MSE_tracker
                }


    return model, trackers, latent_bg_target, latent_t_target

def plot_comparison_curves(track_results, title='param', label_title='param'):
    plt.figure(figsize=(10, 6))
    for label, tracker in track_results.items():
        plt.plot(tracker, label=label_title + '=' + str(label))
    plt.title(title + " Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel(title)
    plt.legend()
    plt.grid(True)
    plt.show()


