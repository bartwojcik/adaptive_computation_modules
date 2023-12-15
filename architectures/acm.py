import logging
from copy import deepcopy
from functools import partial
from typing import Union

import torch
from torch import nn

from architectures.custom import create_attention_projection_filter_condition
from architectures.vit import ffn_filter_condition
from libs.aimle.aimle import aimle
from libs.aimle.target import TargetDistribution, AdaptiveTargetDistribution
from utils import find_module_names, add_save_activations_hook, remove_hooks, get_module_by_name, set_module_by_name


class ACMGatingNetwork(nn.Module):
    pass


class SoftGatingNetwork(ACMGatingNetwork):
    def __init__(self, name, hidden_dim, num_choices):
        super().__init__()
        self.name = name
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(self.hidden_dim, num_choices)

    def forward(self, x):
        x = self.fc(x)
        x = nn.functional.softmax(x, dim=-1)
        return x


class GumbelGatingNetwork(ACMGatingNetwork):
    def __init__(self, name, hidden_dim, num_choices, gs_tau, gs_hard=True, min_choice=None):
        super().__init__()
        self.name = name
        self.hidden_dim = hidden_dim
        if gs_tau == 'trainable':
            self.gs_tau = nn.Parameter(torch.ones(1))
        else:
            self.gs_tau = gs_tau
        self.gs_hard = gs_hard
        self.min_choice = 0
        if min_choice is not None:
            for k, v in min_choice.items():
                if k in name:
                    self.min_choice = v
                    self.real_choices = num_choices - v
                    self.fc = nn.Linear(self.hidden_dim, self.real_choices)
                    break
            else:
                self.fc = nn.Linear(self.hidden_dim, num_choices)
        else:
            self.fc = nn.Linear(self.hidden_dim, num_choices)

    def forward(self, x):
        input_type = x.dtype
        # x = self.net(x)
        x = self.fc(x)
        if self.training:
            x = nn.functional.gumbel_softmax(x, tau=self.gs_tau, hard=self.gs_hard)
        else:
            num_classes = x.size(-1)
            x = x.argmax(dim=-1)
            x = nn.functional.one_hot(x, num_classes).to(input_type)
        # "minimum choice correction"
        if self.min_choice > 0:
            batch_size, seq_len, _ = x.size()
            x = x.flatten(0, 1)
            x = torch.cat([torch.zeros(x.size(0), self.min_choice, device=x.device), x], dim=-1)
            x = x.view(batch_size, seq_len, x.size(-1))
        return x


class DeepGumbelGatingNetwork(ACMGatingNetwork):
    def __init__(self, name, hidden_dim, num_choices, gs_tau, gs_hard=True, min_choice=None, depth=2,
                 gating_hidden_dim=128):
        super().__init__()
        assert depth >= 2
        self.name = name
        self.hidden_dim = hidden_dim
        if gs_tau == 'trainable':
            self.gs_tau = nn.Parameter(torch.ones(1))
        else:
            self.gs_tau = gs_tau
        self.gs_hard = gs_hard
        self.min_choice = 0
        modules = []
        modules.append(nn.Linear(self.hidden_dim, gating_hidden_dim))
        for _ in range(depth - 2):
            modules.append(nn.Linear(gating_hidden_dim, gating_hidden_dim))
        if min_choice is not None:
            for k, v in min_choice.items():
                if k in name:
                    self.min_choice = v
                    self.real_choices = num_choices - v
                    modules.append(nn.Linear(gating_hidden_dim, self.real_choices))
                    break
            else:
                modules.append(nn.Linear(gating_hidden_dim, num_choices))
        else:
            modules.append(nn.Linear(gating_hidden_dim, num_choices))
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        input_type = x.dtype
        x = self.net(x)
        if self.training:
            x = nn.functional.gumbel_softmax(x, tau=self.gs_tau, hard=self.gs_hard)
        else:
            num_classes = x.size(-1)
            x = x.argmax(dim=-1)
            x = nn.functional.one_hot(x, num_classes).to(input_type)
        # "minimum choice correction"
        if self.min_choice > 0:
            batch_size, seq_len, _ = x.size()
            x = x.flatten(0, 1)
            x = torch.cat([torch.zeros(x.size(0), self.min_choice, device=x.device), x], dim=-1)
            x = x.view(batch_size, seq_len, x.size(-1))
        return x


def onehot_argmax_for_imle(x):
    with torch.inference_mode():
        input_type = x.dtype
        num_classes = x.size(-1)
        x = x.argmax(dim=-1)
    with torch.no_grad():
        x = nn.functional.one_hot(x, num_classes).to(input_type)
    return x


def top1_for_imle(x):
    with torch.inference_mode():
        x = x.detach()
        scores, indices = torch.topk(x, 1, sorted=True)
        mask = torch.zeros_like(x, device=x.device).scatter_(-1, indices, 1.0)
    return mask.clone()


class AIMLEGatingNetwork(ACMGatingNetwork):
    def __init__(self, name, hidden_dim, num_choices, alpha=1.0, beta=1.0, g_scaling=False):
        super().__init__()
        self.name = name
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(self.hidden_dim, num_choices)
        # self.differentiable_op = aimle(onehot_argmax_for_imle,
        if alpha is None:
            self.differentiable_op = aimle(top1_for_imle,
                                           target_distribution=AdaptiveTargetDistribution(initial_beta=beta))
        else:
            self.differentiable_op = aimle(top1_for_imle,
                                           target_distribution=TargetDistribution(alpha=alpha,
                                                                                  beta=beta,
                                                                                  do_gradient_scaling=g_scaling))

    def forward(self, x):
        x = self.fc(x)
        x.requires_grad_(True)
        # x.register_hook(create_print_grad_hook(self.name))
        if self.training:
            x = self.differentiable_op(x)
        else:
            x = onehot_argmax_for_imle(x)
        return x


class AdaptiveComputationMLP(nn.Module):
    def __init__(self, name: str, hidden_dim: int, block_dim: int, num_blocks: int, dropout: float = 0.0,
                 activation: str = 'gelu', bias: bool = False):
        from common import ACTIVATION_NAME_MAP
        super().__init__()
        self.name = name
        self.hidden_dim = hidden_dim
        self.block_dim = block_dim
        self.num_blocks = num_blocks
        self.num_choices = num_blocks + 1
        self.bias = bias
        self.w1 = nn.Parameter(torch.empty(hidden_dim, num_blocks * block_dim))
        self.b1 = nn.Parameter(torch.empty(num_blocks * block_dim)) if bias is True or bias == 'hidden_only' else None
        self.w2 = nn.Parameter(torch.empty(num_blocks * block_dim, hidden_dim))
        self.b2 = nn.Parameter(
            torch.empty(self.num_choices * hidden_dim)) if bias is True or bias == 'out_only' else None
        self.activation = ACTIVATION_NAME_MAP[activation]()
        self.dropout = dropout
        self.gating_network = None
        self._forward_mode = 'gated'
        self._checkpoint = False
        self._detach_mode = 'detach'
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.w1, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.w2, a=5 ** 0.5)
        if self.bias is True or self.bias == 'hidden_only':
            # b1
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w1[:, :self.block_dim])
            bound = 1 / fan_in ** 0.5 if fan_in > 0 else 0
            nn.init.uniform_(self.b1, -bound, bound)
        if self.bias is True or self.bias == 'out_only':
            # b2
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w2[:self.block_dim])
            bound = 1 / fan_in ** 0.5 if fan_in > 0 else 0
            nn.init.uniform_(self.b2, -bound, bound)

    def forward_all(self, x):
        zero_result = torch.zeros_like(x).unsqueeze(-2)
        x = x @ self.w1
        if self.bias is True or self.bias == 'hidden_only':
            x = x + self.b1
        x = self.activation(x)
        x = nn.functional.dropout(x, self.dropout, self.training)
        x_chunked = x.view(x.size(0), self.num_blocks, self.block_dim)
        x_chunked = torch.bmm(x_chunked.permute(1, 0, 2),
                              self.w2.view(self.num_blocks, self.block_dim, self.hidden_dim)).permute(1, 0, 2)
        x = torch.cat([zero_result, x_chunked], dim=-2)
        if self.bias is True or self.bias == 'out_only':
            x = x + self.b2.view(1, self.num_choices, self.hidden_dim)
        if self._detach_mode == 'detach':
            x = x + x.cumsum(dim=-2).detach() - x.detach()
        elif self._detach_mode == 'no_detach':
            x = x.cumsum(dim=-2)
        x = nn.functional.dropout(x, self.dropout, self.training)
        return x

    def forward_gated(self, x, gating_indices):
        # TODO polish this?
        # for now, minimizing FLOPs should be more important than performance on GPU
        highest_choice = gating_indices.max()
        current_x = torch.zeros_like(x)
        if self.bias is True or self.bias == 'out_only':
            current_x = current_x + self.b2.view(1, self.num_choices, self.hidden_dim)[:, 0]
        for i in range(1, highest_choice):
            mask = gating_indices >= i
            start_index = (i - 1) * self.block_dim
            stop_index = i * self.block_dim
            block_x = x[mask] @ self.w1[:, start_index:stop_index]
            if self.bias is True or self.bias == 'hidden_only':
                block_x = block_x + self.b1[start_index:stop_index]
            block_x = self.activation(block_x)
            block_x = nn.functional.dropout(block_x, self.dropout, self.training)
            block_x = block_x @ self.w2[start_index:stop_index]
            if self.bias is True or self.bias == 'out_only':
                block_x = block_x + self.b2.view(1, self.num_choices, self.hidden_dim)[:, i]
            current_x[mask] += block_x
        current_x = nn.functional.dropout(current_x, self.dropout, self.training)
        assert current_x.size() == x.size(), f'{current_x.size()=} {x.size()=}'
        return current_x

    @property
    def checkpoint(self):
        return self._checkpoint

    @checkpoint.setter
    def checkpoint(self, value):
        assert isinstance(value, bool)
        self._checkpoint = value

    @property
    def detach_mode(self):
        return self._detach_mode

    @detach_mode.setter
    def detach_mode(self, value):
        assert isinstance(value, str) and value in ['detach', 'no_detach']
        self._detach_mode = value

    @property
    def forward_mode(self):
        return self._forward_mode

    @forward_mode.setter
    def forward_mode(self, value):
        if isinstance(value, int):
            assert 0 <= value < self.num_choices
        elif isinstance(value, float):
            assert 0.0 <= value <= 1.0
            value = round(value * (self.num_choices - 1))
        elif isinstance(value, str):
            assert value in ['gated', 'all', 'random']
        self._forward_mode = value

    def plain_forward(self, x):
        # assumes the entire module is in a residual block and zero output is allowed
        orig_size = x.size()
        if self._forward_mode == 'gated':
            gating_network_outputs = self.gating_network(x)
            gating_indices = gating_network_outputs.argmax(dim=-1)
        else:
            gating_network_outputs = None
        if self._forward_mode == 'all' or isinstance(self._forward_mode, int):
            outputs = self.forward_all(x.flatten(0, 1))
            outputs = outputs.view(*orig_size[:2], self.num_choices, self.hidden_dim)
            if isinstance(self._forward_mode, int):
                outputs = outputs[:, :, self._forward_mode]
        elif self._forward_mode in ['gated']:
            # returns already-reduced output
            outputs = self.forward_gated(x.flatten(0, 1), gating_indices.flatten(0, 1))
            outputs = outputs.view(*orig_size)
        return outputs, {self.name: gating_network_outputs}

    def forward(self, x):
        if self.checkpoint == True:
            return torch.utils.checkpoint.checkpoint(self.plain_forward, x, use_reentrant=False)
        else:
            return self.plain_forward(x)


def acmize_vit(original_model: nn.Module, example_input: torch.Tensor,
               ffn: bool, o_proj: bool, q_proj: bool, k_proj: bool, v_proj: bool,
               block_flops_factor: float, num_blocks: int,
               dropout: Union[float, str], attention_dropout: Union[float, str],
               bias: bool = None, attention_bias: bool = None, activation: str = 'gelu',
               attention_activation: str = 'identity'):
    original_model.eval()
    d = original_model.hidden_dim
    attention_linear_block_dim = round(block_flops_factor * (d ** 2 + d) / (2 * d + 1))
    ffn_block_dim = round(block_flops_factor * original_model.mlp_dim)
    logging.info(f'Determined attention_linear_block_dim for {num_blocks} blocks: {attention_linear_block_dim}')
    logging.info(f'Determined ffn_block_dim for {num_blocks} blocks: {ffn_block_dim}')
    if dropout == 'same':
        dropout = original_model.dropout
    if attention_dropout == 'same':
        attention_dropout = original_model.attention_dropout
    model = deepcopy(original_model)
    ffn_modules_to_replace = []
    attention_linear_modules_to_replace = []
    if ffn:
        ffn_modules_to_replace += find_module_names(original_model, ffn_filter_condition)
    if o_proj:
        attention_linear_modules_to_replace += find_module_names(original_model,
                                                                 create_attention_projection_filter_condition(
                                                                     'o_proj'))
    if q_proj:
        attention_linear_modules_to_replace += find_module_names(original_model,
                                                                 create_attention_projection_filter_condition('q_proj'))
    if k_proj:
        attention_linear_modules_to_replace += find_module_names(original_model,
                                                                 create_attention_projection_filter_condition('k_proj'))
    if v_proj:
        attention_linear_modules_to_replace += find_module_names(original_model,
                                                                 create_attention_projection_filter_condition('v_proj'))
    # use hooks to get an example input
    modules_to_replace = ffn_modules_to_replace + attention_linear_modules_to_replace
    module_inputs, _, handles = add_save_activations_hook(original_model, modules_to_replace)
    original_model(example_input)
    remove_hooks(handles)
    # calculate size and replace the selected layers with ACMs
    for name in ffn_modules_to_replace:
        original_module = get_module_by_name(model, name)
        bias = bias if bias is not None else original_module[0].bias is not None
        replacement = AdaptiveComputationMLP(name, model.hidden_dim, ffn_block_dim, num_blocks, dropout,
                                             activation=activation, bias=bias)
        set_module_by_name(model, name, replacement)
        logging.info(f'Replacing {name}')
    for name in attention_linear_modules_to_replace:
        original_module = get_module_by_name(model, name)
        attention_bias = attention_bias if attention_bias is not None else original_module[0].bias is not None
        replacement = AdaptiveComputationMLP(name, model.hidden_dim, attention_linear_block_dim, num_blocks,
                                             attention_dropout, activation=attention_activation, bias=attention_bias)
        set_module_by_name(model, name, replacement)
        logging.info(f'Replacing {name}')

    # replace forwards, so that gating choices can be returned directly
    # and potential budget can be passed into the gating networks?

    def attention_forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, _key_padding_mask=None,
                          need_weights=False, attn_mask=None):
        assert query.size() == key.size() == value.size()
        batch_size, seq_length, embed_dim = query.size()
        gating_data = {}
        if isinstance(self.q_proj, AdaptiveComputationMLP):
            query, q_gating_data = self.q_proj(query)
            gating_data.update(q_gating_data)
        else:
            query = self.q_proj(query)
        if isinstance(self.k_proj, AdaptiveComputationMLP):
            key, k_gating_data = self.k_proj(key)
            gating_data.update(k_gating_data)
        else:
            key = self.k_proj(key)
        if isinstance(self.v_proj, AdaptiveComputationMLP):
            value, v_gating_data = self.v_proj(value)
            gating_data.update(v_gating_data)
        else:
            value = self.v_proj(value)
        # separate the head dimension, and permute dimensions into [Batch, Head, SeqLen, Dims]
        query = query.reshape(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        key = key.reshape(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        value = value.reshape(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # Determine value outputs
        values, attention = self.scaled_dot_product(query, key, value, mask=attn_mask,
                                                    dropout_p=self.dropout_p if self.training else 0.0)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        if isinstance(self.o_proj, AdaptiveComputationMLP):
            o, o_routing_data = self.o_proj(values)
            gating_data.update(o_routing_data)
        else:
            o = self.o_proj(values)
        if need_weights:
            return o, attention, gating_data
        else:
            return o, None, gating_data

    def block_forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        assert not torch.any(torch.isnan(input)), f'{input=} {self=}'
        x = self.ln_1(input)
        x, _, gating_data = self.self_attention(query=x, key=x, value=x, need_weights=False)
        x = self.dropout(x)
        x = x + input
        y = self.ln_2(x)
        # not all FFN layers have to be replaced with an ACM layer
        if isinstance(self.mlp, AdaptiveComputationMLP):
            y, mlp_gating_data = self.mlp(y)
            gating_data.update(mlp_gating_data)
        else:
            y = self.mlp(y)
        return x + y, gating_data

    def encoder_forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        encoder_block_x = self.dropout(input + self.pos_embedding)
        gating_data = {}
        for layer in self.layers:
            encoder_block_x, encoder_block_gating_data = layer(encoder_block_x)
            gating_data.update(encoder_block_gating_data)
        return self.ln(encoder_block_x), gating_data

    def main_forward(self, x: torch.Tensor, return_gating_data: bool = False):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]
        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x, gating_data = self.encoder(x)
        # Classifier "token" as used by standard language architectures
        x = x[:, 0]
        x = self.heads(x)
        if return_gating_data is True:
            return x, gating_data
        else:
            return x

    for i in range(len(model.encoder.layers)):
        model.encoder.layers[i].self_attention.forward = partial(attention_forward,
                                                                 model.encoder.layers[i].self_attention)
        model.encoder.layers[i].forward = partial(block_forward, model.encoder.layers[i])
    model.encoder.forward = partial(encoder_forward, model.encoder)
    model.forward = partial(main_forward, model)
    return model, modules_to_replace


GATING_NETWORK_MAP = {
    'acm_soft': SoftGatingNetwork,
    'acm_gumbel': GumbelGatingNetwork,
    'acm_deep_gumbel': DeepGumbelGatingNetwork,
    'acm_aimle4': AIMLEGatingNetwork,
}


def add_gating_networks(acmized_model, gating_network_type, gating_network_args):
    acm_module_names = find_module_names(acmized_model, lambda _, m: isinstance(m, AdaptiveComputationMLP))
    # create gating networks
    for acm_module_name in acm_module_names:
        acm_module = get_module_by_name(acmized_model, acm_module_name)
        acm_module.gating_network = GATING_NETWORK_MAP[gating_network_type](acm_module_name,
                                                                            acm_module.hidden_dim,
                                                                            acm_module.num_choices,
                                                                            **gating_network_args)
