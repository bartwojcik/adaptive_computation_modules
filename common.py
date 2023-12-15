import math

import omegaconf
import torch.nn
from torch import nn, optim
from torch.nn.init import calculate_gain

from architectures.acm import AdaptiveComputationMLP
from architectures.avit import AvitWrapper
from architectures.early_exits.sdn import SDN
from architectures.early_exits.ztw import ZTWCascading, ZTWEnsembling
from architectures.moe.moe_layers import MoELayer, ExpertsLayer
from architectures.moe.moe_models import MoEViT

from architectures.pretrained import get_vit_b_16
from architectures.vit import VisionTransformer, MLP
from utils import BCEWithLogitsLossWrapper


def default_init(model):
    for m in model.modules():
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()


def acm_uniform(model):
    # stupid initialization as baseline
    for m in model.modules():
        if isinstance(m, AdaptiveComputationMLP):
            nn.init.uniform_(m.w1, -1.0, 1.0)
            nn.init.uniform_(m.w2, -1.0, 1.0)
            if hasattr(m, 'b1') and m.b1 is not None:
                nn.init.constant_(m.b1, 0.0)
            if hasattr(m, 'b2') and m.b2 is not None:
                nn.init.constant_(m.b2, 0.0)


def acm_orthogonal(model):
    gain = calculate_gain('relu')
    for m in model.modules():
        if isinstance(m, AdaptiveComputationMLP):
            for i in range(m.num_blocks):
                start_index = i * m.block_dim
                stop_index = (i + 1) * m.block_dim
                nn.init.orthogonal_(m.w1[:, start_index:stop_index], gain=gain)
                nn.init.orthogonal_(m.w2[start_index:stop_index], gain=gain)
            if hasattr(m, 'b1') and m.b1 is not None:
                nn.init.constant_(m.b1, 0.0)
            if hasattr(m, 'b2') and m.b2 is not None:
                nn.init.constant_(m.b2, 0.0)


def acm_he_normal(model):
    for m in model.modules():
        if isinstance(m, AdaptiveComputationMLP):
            for i in range(m.num_blocks):
                start_index = i * m.block_dim
                stop_index = (i + 1) * m.block_dim
                nn.init.kaiming_normal_(m.w1[:, start_index:stop_index])
                nn.init.kaiming_normal_(m.w2[start_index:stop_index])
            if hasattr(m, 'b1') and m.b1 is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.w1[:, 0:m.block_dim])
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(m.b1, -bound, bound)
            if hasattr(m, 'b2') and m.b2 is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.w2[0:m.block_dim])
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(m.b2, -bound, bound)


def get_default_args():
    default_args = omegaconf.OmegaConf.create()

    default_args.exp_id = 0  # experiment id
    default_args.runs_dir = "runs"  # directory to save the results to
    default_args.model_class = None  # class of the model to train
    default_args.model_args = None  # arguments to be passed to the model init function
    default_args.dataset = None  # dataset to train on
    default_args.dataset_args = None  # customization arguments for the dataset
    default_args.mixup_alpha = None  # alpha parameter for mixup's beta distribution
    default_args.cutmix_alpha = None  # alpha parameter for cutmix's beta distribution
    default_args.mixup_mode = None  # how to apply mixup/cutmix ('batch', 'pair' or 'elem')
    default_args.mixup_smoothing = None  # label smoothing when using mixup
    default_args.init_fun = None  # parameters init function to be used
    default_args.batch_size = None  # batch size for training
    default_args.loss_type = None  # loss function to be used for training
    default_args.loss_args = None  # arguments to be passed to the loss init function
    default_args.optimizer_class = None  # class of the optimizer to use for training
    default_args.optimizer_args = None  # arguments to be passed to the optimizer init function
    default_args.scheduler_class = None  # class of the scheduler to use for training
    default_args.scheduler_args = None  # arguments to be passed to the scheduler init function
    default_args.clip_grad_norm = None  # gradient clipping norm
    default_args.epochs = None  # number of epochs to train for
    default_args.mixed_precision = None  # whether to use accelerate's mixed precision
    default_args.eval_points = 100  # number of short evaluations on the validation/test data while training
    default_args.eval_batches = 10  # number of batches to evaluate on each time while training
    default_args.save_every = 10  # save model every N minutes
    default_args.use_wandb = False  # use weights and biases

    # use only None for method specific args, and fill them with default values in the code!
    # unless they are not used in generate_run_name(), changing the defaults changes run names for unrelated runs!
    # method specific args
    default_args.base_on = None  # unique experiment name to use the model from

    # Early Exit specific
    default_args.with_backbone = None  # whether to train the backbone network along with the heads
    default_args.auxiliary_loss_type = None  # type of the auxiliary loss for early-exit heads
    default_args.auxiliary_loss_weight = None  # weight of the auxiliary loss for early-exit heads
    default_args.eval_thresholds = None  # number of early exit thresholds to evaluate

    # MoEfication specific
    default_args.k_to_eval = None  # list of ks to evaluate moefication model with
    default_args.router_loss_type = None  # loss function to be used for training the routers
    default_args.router_loss_args = None  # arguments to be passed to the router loss init function

    # ACM specific
    default_args.checkpoint_acm = None  # use checkpoint to limit memory requirements
    default_args.acm_detach_mode = None  # train mode to use for ACMs (e.g. with detach or without)
    default_args.acm_freeze_mode = None  # freeze learners sequentially during training
    default_args.acm_distill_loss_type = None  # loss type for representation distillation stage
    default_args.gating_error_factor = None  # factor for selecting labels for layerwise gating network training
    default_args.routing_loss_type = None  # routing loss type in the end-to-end training stage
    default_args.routing_loss_weight = None  # routing loss weight in the end-to-end training stage
    default_args.routing_loss_target = None  # if not None, then model will be penalized for deviating from this budget
    default_args.entropy_loss_weight = None  # entropy loss weight in the end-to-end training stage
    default_args.compute_diversity_loss_weight = None  # compute diversity loss weight in the end-to-end training stage
    default_args.acm_finetune_mode = None  # how to finetune the model with ACM present

    # A-ViT specific
    default_args.ponder_loss_weight = None  # weight of the pondering loss
    default_args.distr_target_depth = None  # mean depth of the distributional prior
    default_args.distr_prior_loss_weight = None  # weight of the distributional prior regularization

    return default_args


INIT_NAME_MAP = {
    'default': default_init,
    'acm_uniform': acm_uniform,
    'acm_ortho': acm_orthogonal,
    'acm_he_normal': acm_he_normal,
    None: None,
}

ACTIVATION_NAME_MAP = {
    'relu': torch.nn.ReLU,
    'gelu': torch.nn.GELU,
    'sigmoid': torch.nn.Sigmoid,
    'tanh': torch.nn.Tanh,
    'leaky_relu': torch.nn.LeakyReLU,
    'softplus': torch.nn.Softplus,
    'silu': torch.nn.SiLU,
    'identity': torch.nn.Identity,
}

LOSS_NAME_MAP = {
    'ce': nn.CrossEntropyLoss,
    'bcewl_c': BCEWithLogitsLossWrapper,
    'bcewl': nn.BCEWithLogitsLoss,
    'bce': nn.BCELoss,
    'nll': nn.NLLLoss,
    'mse': nn.MSELoss,
    'mae': nn.L1Loss,
    'huber': nn.HuberLoss,
}

OPTIMIZER_NAME_MAP = {
    'sgd': optim.SGD,
    'adam': optim.AdamW,
    'adagrad': optim.Adagrad,
}

SCHEDULER_NAME_MAP = {
    'reduce_on_plateau': optim.lr_scheduler.ReduceLROnPlateau,
    'cosine': optim.lr_scheduler.CosineAnnealingLR,
    'cosine_warm_restarts': optim.lr_scheduler.CosineAnnealingWarmRestarts,
}

MODEL_NAME_MAP = {
    'vit': VisionTransformer,
    'avit': AvitWrapper,
    'sdn': SDN,
    'ztw_cascading': ZTWCascading,
    'ztw_ensembling': ZTWEnsembling,
    'tv_vit_b_16': get_vit_b_16,
}
