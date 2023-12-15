import logging
from copy import deepcopy
from functools import partial
from typing import List

import torch
from k_means_constrained import KMeansConstrained
from torch import nn

from architectures.moe.moe_layers import MoELayer, MOE_IMPL_MAP, ModuleBatchedExperts, ExecuteAllExperts
from architectures.moe.moe_models import moe_vit_block_forward, moe_vit_encoder_forward, moe_vit_main_forward
from architectures.vit import ffn_filter_condition
from common import ACTIVATION_NAME_MAP
from utils import find_module_names, get_module_by_name, set_module_by_name


class MoeficationMoE(MoELayer):
    # https://arxiv.org/pdf/2110.01786.pdf
    def __init__(self, hidden_dim, layer_id, num_experts, bias, expert_dim, activation, experts_class='module'):
        super().__init__(num_experts, layer_id)
        self.hidden_dim = hidden_dim
        self.expert_dim = expert_dim
        self.num_experts = num_experts
        self.k = num_experts
        self.bias = bias
        # instantiate experts
        self.experts = MOE_IMPL_MAP[experts_class](dim=hidden_dim,
                                                   num_experts=num_experts,
                                                   depth=2,
                                                   expert_dim=self.expert_dim,
                                                   bias=False if not bias else 'without_last',
                                                   activation=activation)
        if self.bias:
            self.last_bias = nn.Parameter(torch.zeros(1, hidden_dim))
        self.forward_mode = 'all'
        self.router = None
        self.k = None

    def gate(self, x):
        # x is of size (batch_size, sequence_length, dim)
        if self.forward_mode == 'all':
            # sanity check - route to all tensors when router is not present
            routing_tensor = torch.ones(x.size(0), x.size(1), self.num_experts, dtype=x.dtype, device=x.device)
        elif self.forward_mode == 'topk':
            routing_tensor = self.router(x)
            top = torch.topk(routing_tensor, k=self.k, dim=-1)[1]
            routing_tensor = torch.zeros_like(routing_tensor)
            routing_tensor.scatter_(2, top, 1.0)
        else:
            raise ValueError(f'Unsupported forward_mode: {self.forward_mode}')
        return routing_tensor

    def forward(self, x):
        # x is of size (batch_size, sequence_length, dim)
        routing_tensor = self.gate(x)
        orig_size = x.size()
        x = x.view(-1, x.size(-1))
        out = self.experts(x, routing_tensor.view(-1, routing_tensor.size(-1)))
        if self.bias:
            out = out + self.last_bias
        out = out.view(orig_size)
        return out, (routing_tensor,)


def replace_layer_with_moe(ffn, num_experts, layer_id, experts_class):
    # ffn is a nn.Sequential
    # with nn.Linear layers at indices 0 and 3
    w1 = ffn[0]
    hidden_dim = w1.in_features
    d_ff = w1.out_features
    assert d_ff % num_experts == 0, f'd_ff has to be divisible by the number of experts'
    expert_size = d_ff // num_experts
    activation = type(ffn[1])
    moe_layer = MoeficationMoE(hidden_dim, layer_id, num_experts, w1.bias is not None, expert_size, activation,
                               experts_class)
    return moe_layer


def replace_with_moes(original_model: nn.Module, num_experts: int, experts_class='module'):
    original_model.eval()
    model = deepcopy(original_model)
    ffn_modules_to_replace = find_module_names(original_model, ffn_filter_condition)
    # calculate size and replace the selected layers with MoE layers
    for i, name in enumerate(ffn_modules_to_replace):
        original_module = get_module_by_name(model, name)
        logging.info(f'Replacing {name} (hidden size {original_module[0].out_features}) with {num_experts} experts')
        replacement = replace_layer_with_moe(original_module, num_experts, i, experts_class)
        set_module_by_name(model, name, replacement)
    for i in range(len(model.encoder.layers)):
        model.encoder.layers[i].forward = partial(moe_vit_block_forward, model.encoder.layers[i])
    model.encoder.forward = partial(moe_vit_encoder_forward, model.encoder)
    model.forward = partial(moe_vit_main_forward, model)
    return model, ffn_modules_to_replace


def param_clustering_split(ffn, moe_layer):
    num_experts = moe_layer.num_experts
    # ffn is a nn.Sequential
    # with nn.Linear layers at indices 0 and 3
    w1 = ffn[0]
    d_ff = w1.out_features
    expert_size = d_ff // num_experts
    w1_normalized = nn.functional.normalize(w1.weight, p=2.0, dim=1)
    labels = KMeansConstrained(n_clusters=num_experts, size_min=expert_size, size_max=expert_size) \
        .fit_predict(w1_normalized.detach().cpu().numpy())
    # split weights into experts by labels
    if isinstance(moe_layer.experts, ModuleBatchedExperts):
        assert moe_layer.experts.depth == 2
        # experts is a nn.ModuleList
        # each expert is a nn.Sequential module
        # with nn.Linear layers at indices 0 and 2
        with torch.no_grad():
            filled_neuron_counts = [0 for _ in range(num_experts)]
            for neuron_index, expert_index in enumerate(labels):
                expert_neuron_index = filled_neuron_counts[expert_index]
                moe_layer.experts.e[expert_index][0].weight[expert_neuron_index].copy_(ffn[0].weight[neuron_index])
                if moe_layer.bias:
                    moe_layer.experts.e[expert_index][0].bias[expert_neuron_index].copy_(ffn[0].bias[neuron_index])
                moe_layer.experts.e[expert_index][2].weight[:, expert_neuron_index].copy_(
                    ffn[3].weight[:, neuron_index])
                filled_neuron_counts[expert_index] += 1
            # copy the last layer bias
            if moe_layer.bias:
                moe_layer.last_bias.copy_(ffn[3].bias)
    elif isinstance(moe_layer.experts, ExecuteAllExperts):
        assert moe_layer.experts.depth == 2
        with torch.no_grad():
            filled_neuron_counts = [0 for _ in range(num_experts)]
            for neuron_index, expert_index in enumerate(labels):
                expert_neuron_index = filled_neuron_counts[expert_index]
                moe_layer.experts.layers[0].w[expert_index, :, expert_neuron_index].copy_(ffn[0].weight[neuron_index])
                if moe_layer.bias:
                    moe_layer.experts.layers[0].b[expert_index, :, expert_neuron_index].copy_(ffn[0].bias[neuron_index])
                moe_layer.experts.layers[1].w[expert_index, expert_neuron_index].copy_(ffn[3].weight[:, neuron_index])
                filled_neuron_counts[expert_index] += 1
            # copy the last layer bias
            if moe_layer.bias:
                moe_layer.last_bias.copy_(ffn[3].bias)
    else:
        # TODO
        raise NotImplementedError('Other variants not handled yet')


def split_original_parameters(original_model: nn.Module, moe_model: nn.Module, replaced_module_names: List[str]):
    original_model.eval()
    assert len(replaced_module_names) > 0
    # calculate size and replace the selected layers with MoE layers
    for i, name in enumerate(replaced_module_names):
        original_module = get_module_by_name(original_model, name)
        moe_module = get_module_by_name(moe_model, name)
        num_experts = moe_module.num_experts
        logging.info(f'Clustering parameters from {name} into {num_experts} experts')
        param_clustering_split(original_module, moe_module)


class MoeficationRouter(nn.Sequential):
    def __init__(self, hidden_dim, num_experts, width=128, depth=2, bias=False, activation='tanh',
                 output_activation='identity'):
        layers = []
        if depth == 1:
            layers.append(nn.Linear(hidden_dim, num_experts, bias=bias))
        else:
            layers.append(nn.Linear(hidden_dim, width, bias=bias))
            layers.append(ACTIVATION_NAME_MAP[activation]())
            for i in range(depth - 2):
                layers.append(nn.Linear(width, width, bias=bias))
                layers.append(ACTIVATION_NAME_MAP[activation]())
            layers.append(nn.Linear(width, num_experts, bias=bias))
        layers.append(ACTIVATION_NAME_MAP[output_activation]())
        super().__init__(*layers)


def add_routers(model, router_args):
    moe_module_names = find_module_names(model, lambda _, m: isinstance(m, MoeficationMoE))
    moe_module_dict = {}
    # create gating networks
    for moe_module_name in moe_module_names:
        moe_module = get_module_by_name(model, moe_module_name)
        logging.info(f'Adding router to {moe_module_name}')
        moe_module.router = MoeficationRouter(moe_module.hidden_dim,
                                              moe_module.num_experts,
                                              **router_args)
        moe_module_dict[moe_module_name] = moe_module
    assert len(moe_module_dict) > 0
    return moe_module_dict
