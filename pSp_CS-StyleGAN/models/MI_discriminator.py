import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, opts):
        """
        Discriminator that processes (batch, 18, 1024) → (batch, 1)
        Args:
            opts: Configuration object with attributes:
                  - opts.latent_dim: Latent space dimension (e.g., 512)
                  - opts.n_layer_disc: Number of layers in the MLP
        """
        super(Discriminator, self).__init__()
        self.opts = opts
        input_dim = self.opts.latent_dim * 2  # 1024 if latent_dim=512
        hidden_dim = 512
        num_layers = self.opts.n_layer_disc

        layers = []
        prev_dim = input_dim

        # Dynamically add hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())  # Activation function
            prev_dim = hidden_dim  # Update previous dimension

        # Final layer (hidden_dim -> 1) WITHOUT Sigmoid
        layers.append(nn.Linear(hidden_dim, 1))

        # Define MLP
        self.mlp = nn.Sequential(*layers)

    def forward(self, c, s):
        """
        c: Tensor of shape (batch, 18, 512)
        s: Tensor of shape (batch, 18, 512)
        """
        x = torch.cat([c, s], dim=-1)  # Shape: (batch, 18, 1024)
        x = self.mlp(x)  # Shape: (batch, 18, 1)
        x = x.mean(dim=1)  # Aggregate across 18 features → (batch, 1)
        return x  # Return raw logits (no sigmoid)


class DiscriminatorGlobal(nn.Module):
    def __init__(self, opts):
        """
        DiscriminatorGlobal processes flattened (batch, 18*1024) → (batch, 1)
        Args:
            opts: Configuration object with attributes:
                  - opts.style_dim: Style space dimension (e.g., 18)
                  - opts.latent_dim: Latent space dimension (e.g., 512)
                  - opts.n_layer_disc: Number of layers in the MLP
        """
        super(DiscriminatorGlobal, self).__init__()
        self.opts = opts

        input_dim = self.opts.style_dim * self.opts.latent_dim * 2  # 18 * 512 * 2 = 18432
        hidden_dim = 512
        num_layers = self.opts.n_mut_layers

        layers = []
        prev_dim = input_dim

        # Dynamically add hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())  # Activation function
            prev_dim = hidden_dim  # Update previous dimension

        # Final layer (hidden_dim -> 1) WITHOUT Sigmoid
        layers.append(nn.Linear(hidden_dim, 1))

        # Define MLP
        self.mlp = nn.Sequential(*layers)

    def forward(self, c, s):
        """
        c: Tensor of shape (batch, 18, 512)
        s: Tensor of shape (batch, 18, 512)
        """
        x = torch.cat([c, s], dim=-1)  # Shape: (batch, 18, 1024)
        x = x.view(x.shape[0], -1)  # Flatten to (batch, 18432)
        x = self.mlp(x)  # Shape: (batch, 1)
        return x  # Return raw logits (no sigmoid)


class MI_Discriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.num_layers = 18
        self.latent_dim = args.latent_dim  # 512

        input_dim = self.num_layers * self.latent_dim * 2
        hidden_dim = 512
        num_layers = args.n_layers

        layers = []
        prev_dim = input_dim

        for _ in range(max(0, num_layers - 1)):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, c, s):
        """
        c, s: [B, 18, 512]
        """
        B = c.shape[0]

        # flatten as a whole
        c = c.view(B, -1)   # [B, 18*512]
        s = s.view(B, -1)   # [B, 18*512]

        x = torch.cat([c, s], dim=1)  # [B, 2*18*512]
        logits = self.mlp(x)          # [B, 1]
        return logits.squeeze(1)      # [B]
