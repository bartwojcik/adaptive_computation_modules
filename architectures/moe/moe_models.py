from functools import partial

import torch

from architectures.moe.moe_layers import MoELayer
from architectures.vit import VisionTransformer

MOE_MAP = {
}


def moe_vit_block_forward(self, input: torch.Tensor):
    torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
    assert not torch.any(torch.isnan(input)), f'{input=}'
    x = self.ln_1(input)
    x, _ = self.self_attention(query=x, key=x, value=x, need_weights=False)
    x = self.dropout(x)
    x = x + input
    y = self.ln_2(x)
    # not all FFN layers have to be replaced with a MoE layer
    if isinstance(self.mlp, MoELayer):
        y, gating_data = self.mlp(y)
        return x + y, gating_data
    else:
        y = self.mlp(y)
        return x + y, None


def moe_vit_encoder_forward(self, input: torch.Tensor):
    torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
    encoder_block_x = self.dropout(input + self.pos_embedding)
    gating_data_list = []
    for layer in self.layers:
        encoder_block_x, encoder_block_gating_data = layer(encoder_block_x)
        if encoder_block_gating_data is not None:
            gating_data_list.append(encoder_block_gating_data)
    return self.ln(encoder_block_x), gating_data_list


def moe_vit_main_forward(self, x: torch.Tensor, return_gating_data: bool = False):
    # Reshape and permute the input tensor
    x = self._process_input(x)
    n = x.shape[0]
    # Expand the class token to the full batch
    batch_class_token = self.class_token.expand(n, -1, -1)
    x = torch.cat([batch_class_token, x], dim=1)
    x, gating_data_list = self.encoder(x)
    # Classifier "token" as used by standard language architectures
    x = x[:, 0]
    x = self.heads(x)
    if return_gating_data is True:
        return x, gating_data_list
    else:
        return x


def MoEViT(vit_kwargs, place_at, moe_type, moe_kwargs, initialization_factor=1.0):
    vit = VisionTransformer(**vit_kwargs)
    # replace the selected FFN layers with MoEs
    vit.moe_type = moe_type
    vit.place_at = place_at
    vit.initialization_factor = initialization_factor
    if 'k' in moe_kwargs:
        vit.k = moe_kwargs['k']
    moe_type = MOE_MAP[moe_type]
    for i in place_at:
        # replace layer
        vit.encoder.layers[i].mlp = moe_type(vit.hidden_dim, layer_id=i, **moe_kwargs)
    for i in range(len(vit.encoder.layers)):
        vit.encoder.layers[i].forward = partial(moe_vit_block_forward, vit.encoder.layers[i])
    vit.encoder.forward = partial(moe_vit_encoder_forward, vit.encoder)
    vit.forward = partial(moe_vit_main_forward, vit)
    return vit
