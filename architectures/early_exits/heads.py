from typing import Union, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class ViTHead(nn.Module):

    def __init__(self):
        super().__init__()


def vit_standard_head_class(num_layers, hidden_dim=None):
    assert num_layers > 0

    class ViTStandardHead(ViTHead):
        def __init__(self, in_size: int, out_size: int):
            super().__init__()
            nonlocal hidden_dim
            self._hidden_dim = in_size if hidden_dim is None else hidden_dim
            self._fcs = torch.nn.ModuleList()
            for i in range(num_layers - 1):
                self._fcs.append(torch.nn.Linear(in_size, self._hidden_dim))
                in_size = self._hidden_dim
            self._fcs.append(torch.nn.Linear(in_size, out_size))

        def forward(self, x: torch.Tensor):
            # take the hidden state corresponding to the first (class) token
            x = x[:, 0]
            for i in range(num_layers - 1):
                x = F.gelu(self._fcs[i](x))
            x = self._fcs[-1](x)
            return x

    return ViTStandardHead


def vit_cascading_head_class(num_layers, hidden_dim=None):
    assert num_layers > 0

    class ViTCascadingHead(ViTHead):
        def __init__(self, in_size: int, out_size: int, cascading: bool = True, layer_norm: bool = True,
                     detach: bool = True):
            super().__init__()
            nonlocal hidden_dim
            self._hidden_dim = in_size if hidden_dim is None else hidden_dim
            self._out_size = out_size
            self._cascading = cascading
            self._detach = detach
            self._cascading_norm = nn.LayerNorm(out_size) if layer_norm else nn.Identity()
            self._fcs = torch.nn.ModuleList()
            for i in range(num_layers - 1):
                self._fcs.append(torch.nn.Linear(in_size, self._hidden_dim))
                in_size = self._hidden_dim
            if self._cascading:
                self._fcs.append(torch.nn.Linear(in_size + out_size, out_size))
            else:
                self._fcs.append(torch.nn.Linear(in_size, out_size))

        def forward(self, x: torch.Tensor, cascading_input: torch.Tensor = None):
            # take the hidden state corresponding to the first (class) token
            x = x[:, 0]
            for i in range(num_layers - 1):
                x = F.gelu(self._fcs[i](x))
            if self._cascading:
                assert isinstance(cascading_input, torch.Tensor)
                if self._detach:
                    cascading_input = cascading_input.detach()
                # apply layer norm to previous logits (from ZTW paper's appendix)
                cascading_input = self._cascading_norm(cascading_input)
                x = torch.cat((x, cascading_input), dim=-1)
            x = self._fcs[-1](x)
            return x

    return ViTCascadingHead



HEAD_TYPES = {
    'vit_standard_head': vit_standard_head_class(1),
    'vit_2l_head': vit_standard_head_class(2, 1024),
    'vit_2l_2048_head': vit_standard_head_class(2, 2048),
    'vit_2l_4096_head': vit_standard_head_class(2, 4096),
    'vit_2l_8192_head': vit_standard_head_class(2, 8192),
    'vit_3l_head': vit_standard_head_class(3, 8192),
    'vit_4l_head': vit_standard_head_class(4, 8192),
    'vit_5l_head': vit_standard_head_class(5, 8192),
    'vit_cascading_head': vit_cascading_head_class(1),
    'vit_cascading_2l_head': vit_cascading_head_class(2, 1024),
    'vit_cascading_2l_2048_head': vit_cascading_head_class(2, 2048),
    'vit_cascading_2l_4096_head': vit_cascading_head_class(2, 4096),
    'vit_cascading_2l_8192_head': vit_cascading_head_class(2, 8192),
    'vit_cascading_3l_head': vit_cascading_head_class(3, 8192),
    'vit_cascading_4l_head': vit_cascading_head_class(4, 8192),
    'vit_cascading_5l_head': vit_cascading_head_class(5, 8192),
}
