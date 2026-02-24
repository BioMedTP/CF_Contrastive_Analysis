import torch
import torch.nn as nn

from criteria.lpips.networks import get_network, LinLayers
from criteria.lpips.utils import get_state_dict


# class LPIPS(nn.Module):
#     r"""Creates a criterion that measures
#     Learned Perceptual Image Patch Similarity (LPIPS).
#     Arguments:
#         net_type (str): the network type to compare the features:
#                         'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
#         version (str): the version of LPIPS. Default: 0.1.
#     """
#     def __init__(self, net_type: str = 'alex', version: str = '0.1'):

#         assert version in ['0.1'], 'v0.1 is only supported now'

#         super(LPIPS, self).__init__()

#         # pretrained network
#         self.net = get_network(net_type).to("cuda")

#         # linear layers
#         self.lin = LinLayers(self.net.n_channels_list).to("cuda")
#         self.lin.load_state_dict(get_state_dict(net_type, version))

#     def forward(self, x: torch.Tensor, y: torch.Tensor):
#         feat_x, feat_y = self.net(x), self.net(y)

#         diff = [(fx - fy) ** 2 for fx, fy in zip(feat_x, feat_y)]
#         res = [l(d).mean((2, 3), True) for d, l in zip(diff, self.lin)]

#         return torch.sum(torch.cat(res, 0)) / x.shape[0]



class LPIPS(nn.Module):
    """LPIPS that supports both RGB and grayscale images."""
    
    def __init__(self, net_type: str = 'alex', version: str = '0.1'):
        super().__init__()

        self.net = get_network(net_type).to("cuda")
        self.lin = LinLayers(self.net.n_channels_list).to("cuda")
        self.lin.load_state_dict(get_state_dict(net_type, version))

    def _to_rgb(self, x):
        """Convert grayscale [B,1,H,W] → [B,3,H,W] but leave RGB untouched."""
        if x.shape[1] == 1:          # grayscale
            return x.repeat(1, 3, 1, 1)
        elif x.shape[1] == 3:        # already RGB
            return x
        else:
            raise ValueError(f"LPIPS expects 1 or 3 channels, got {x.shape[1]}")
    
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x = self._to_rgb(x)
        y = self._to_rgb(y)

        feat_x, feat_y = self.net(x), self.net(y)
        diffs = [(fx - fy)**2 for fx, fy in zip(feat_x, feat_y)]
        res = [lin(d).mean((2,3), True) for d, lin in zip(diffs, self.lin)]
        return torch.sum(torch.cat(res, 0)) / x.shape[0]

