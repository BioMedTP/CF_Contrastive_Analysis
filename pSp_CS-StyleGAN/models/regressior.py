import torch
import torch.nn as nn



class MI_Discriminator(nn.Module):
    def __init__(
        self,
        style_dim: int = 18,
        latent_dim: int = 512,
        num_layers: int = 3,
        hidden_dim: int = 1024,
    ):
        super().__init__()

        input_dim = 2 * style_dim * latent_dim  # C || S

        layers = []
        dim = input_dim
        for _ in range(num_layers - 1):
            layers += [
                nn.Linear(dim, hidden_dim),
                nn.ReLU(inplace=True),
            ]
            dim = hidden_dim

        layers.append(nn.Linear(dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, c, s):
        """
        c, s: [B, style_dim, latent_dim]
        """
        B = c.shape[0]

        # flatten whole W+
        c = c.view(B, -1)
        s = s.view(B, -1)

        x = torch.cat([c, s], dim=1)  # [B, 2 * style_dim * latent_dim]
        logits = self.mlp(x)

        return logits.squeeze(1)


class ReconR(nn.Module):
    def __init__(
        self,
        latent_dim: int = 512,
        hidden_dim: int = 512,
        n_layers: int = 3,
    ):
        super().__init__()
        assert n_layers >= 2, "n_layers must be >= 2"

        layers = []
        dim = latent_dim
        for _ in range(n_layers - 1):
            layers += [
                nn.Linear(dim, hidden_dim),
                nn.ReLU(inplace=True),
            ]
            dim = hidden_dim

        layers.append(nn.Linear(dim, latent_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, c):
        """
        c: [B, style_dim, latent_dim]
        """
        B, L, D = c.shape

        c_flat = c.reshape(-1, D).contiguous()  # ✅ safe
        s_hat = self.mlp(c_flat)
        s_hat = s_hat.view(B, L, D)

        return s_hat
