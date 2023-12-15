import logging
from pathlib import Path

import torch
from accelerate import Accelerator
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

from architectures.acm import AdaptiveComputationMLP
from datasets import DATASETS_NAME_MAP
from eval import get_preds_acm
from utils import get_loader, find_module_names, get_module_by_name, load_model
from visualize.cost_vs_plot import FONT_SIZE


def get_default_args():
    default_args = OmegaConf.create()
    default_args.runs_dir = Path.cwd() / 'runs'  # Root dir where experiment data was saved.
    default_args.exp_names = []  # Unique experiment names to visualize the results for (excluding exp_id).
    default_args.exp_ids = [0]  # Experiment ids.
    default_args.display_names = None  # Pretty display names that will be used when generating the plot.
    default_args.output_dir = Path.cwd() / 'figures'  # Target directory.
    default_args.batches = 0  # Number of batches to use.
    default_args.batch_size = None  # Batch size.
    default_args.dataset = None  # Dataset to evaluate on. If "None", will use the dataset used by the run.
    default_args.dataset_args = None  # Dataset arguments. If "None", will use same args as in the training run.
    default_args.use_wandb = False  # Use W&B. Will save and load the models from the W&B cloud.
    return default_args


def is_gated_acm(model):
    has_acms = False
    all_acms_have_gating_networks = True
    acm_module_names = find_module_names(model, lambda _, m: True if isinstance(m, AdaptiveComputationMLP) else False)
    for module_name in acm_module_names:
        has_acms = True
        acm_module = get_module_by_name(model, module_name)
        if acm_module.gating_network is None:
            all_acms_have_gating_networks = False
    return has_acms and all_acms_have_gating_networks


def set_for_eval(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, AdaptiveComputationMLP):
            m.forward_mode = 'gated'


def get_gating_data(args, accelerator, acm_model, run_args, batches=0):
    dataset = args.dataset if args.dataset is not None else run_args.dataset
    dataset_args = args.dataset_args if args.dataset_args is not None else run_args.dataset_args
    _, _, data = DATASETS_NAME_MAP[dataset](**dataset_args)
    dataloader = get_loader(data, run_args.batch_size if args.batch_size is None else args.batch_size,
                            accelerator, shuffle=True)
    _, _, gating_data = get_preds_acm(accelerator, acm_model, dataloader, batches=batches)
    return gating_data


def gating_counts(gating_data):
    return gating_data.flatten(0, 1).sum(dim=0)


def sample_learners_used(gating_data):
    learners_used = gating_data * torch.arange(0, gating_data.size(-1), device=gating_data.device) \
        .unsqueeze(0).unsqueeze(0)
    learners_used = learners_used.sum(dim=-1).mean(dim=1)
    return learners_used


def token_learners_used(gating_data):
    learners_used = gating_data * torch.arange(0, gating_data.size(-1), device=gating_data.device) \
        .unsqueeze(0).unsqueeze(0)
    learners_used = learners_used.sum(dim=-1).flatten()
    return learners_used


def gating_histogram_figure(gating_counts):
    saved_args = locals()
    xs = torch.arange(0, gating_counts.size(0)).numpy()
    gating_counts = gating_counts.cpu().numpy()
    fig, ax = plt.subplots(figsize=(16, 16), facecolor='w', edgecolor='k')
    ax.bar(xs, gating_counts)
    ax.set_xlabel('Learners used', fontsize=FONT_SIZE)
    for i, tick in enumerate(ax.xaxis.get_major_ticks()):
        tick.label.set_fontsize(FONT_SIZE)
    ax.set_ylabel('Number of tokens', fontsize=FONT_SIZE)
    for i, tick in enumerate(ax.yaxis.get_major_ticks()):
        tick.label.set_fontsize(FONT_SIZE)
    ax.autoscale()
    fig.set_tight_layout(True)
    return fig, saved_args


def sample_learners_used_figure(learners_used, num_choices, bins=100):
    saved_args = locals()
    learners_used = learners_used.cpu().numpy()
    fig, ax = plt.subplots(figsize=(16, 16), facecolor='w', edgecolor='k')
    ax.hist(learners_used, bins=bins, range=(0.0, num_choices))
    ax.set_xlim(left=0.0, right=num_choices - 1)
    ax.set_xlabel('Average number of learners used', fontsize=FONT_SIZE)
    for i, tick in enumerate(ax.xaxis.get_major_ticks()):
        tick.label.set_fontsize(FONT_SIZE)
    ax.set_ylabel('Number of samples', fontsize=FONT_SIZE)
    for i, tick in enumerate(ax.yaxis.get_major_ticks()):
        tick.label.set_fontsize(FONT_SIZE)
    fig.set_tight_layout(True)
    return fig, saved_args


def token_learners_used_figure(learners_used, num_choices, bins=100):
    saved_args = locals()
    learners_used = learners_used.cpu().numpy()
    fig, ax = plt.subplots(figsize=(16, 16), facecolor='w', edgecolor='k')
    ax.hist(learners_used, bins=bins, range=(0.0, num_choices))
    ax.set_xlim(left=0.0, right=num_choices - 1)
    ax.set_xlabel('Average number of learners used', fontsize=FONT_SIZE)
    for i, tick in enumerate(ax.xaxis.get_major_ticks()):
        tick.label.set_fontsize(FONT_SIZE)
    ax.set_ylabel('Number of tokens', fontsize=FONT_SIZE)
    for i, tick in enumerate(ax.yaxis.get_major_ticks()):
        tick.label.set_fontsize(FONT_SIZE)
    fig.set_tight_layout(True)
    return fig, saved_args


def main(args):
    logging.basicConfig(
        format=(
            '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] ' '%(message)s'
        ),
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
        force=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    display_names = args.display_names if args.display_names is not None else args.exp_names
    accelerator = Accelerator(split_batches=True)
    for exp_name, display_name in zip(args.exp_names, display_names):
        for exp_id in args.exp_ids:
            acm_model, run_args, state = load_model(args, exp_name, exp_id)
            if not is_gated_acm(acm_model):
                logging.info(f'{exp_name}_{exp_id} is not a gated ACM model - skipping.')
            else:
                acm_model = accelerator.prepare(acm_model)
                gating_data = get_gating_data(args, accelerator, acm_model, run_args, batches=args.batches)
                aggregated_gating_data = sum(v for k, v in gating_data.items())
                num_choices = aggregated_gating_data.size(-1)
                # aggregated for all layers
                counts = gating_counts(aggregated_gating_data)
                fig, saved_args = gating_histogram_figure(counts)
                save_path = args.output_dir / f'counts_{display_name}.png'
                args_save_path = args.output_dir / f'counts_{display_name}.pt'
                fig.savefig(save_path)
                plt.close(fig)
                torch.save(saved_args, args_save_path)
                logging.info(f'Figure saved in {str(save_path)}, args saved in {str(args_save_path)}')
                # aggregated for images
                learners_used = sample_learners_used(aggregated_gating_data) / len(gating_data)
                fig, saved_args = sample_learners_used_figure(learners_used, num_choices)
                save_path = args.output_dir / f'sample_counts_{display_name}.png'
                args_save_path = args.output_dir / f'sample_counts_{display_name}.pt'
                fig.savefig(save_path)
                plt.close(fig)
                torch.save(saved_args, args_save_path)
                logging.info(f'Figure saved in {str(save_path)}, args saved in {str(args_save_path)}')
                # aggregated for tokens
                learners_used = token_learners_used(aggregated_gating_data) / len(gating_data)
                fig, saved_args = token_learners_used_figure(learners_used, num_choices)
                save_path = args.output_dir / f'token_counts_{display_name}.png'
                args_save_path = args.output_dir / f'token_counts_{display_name}.pt'
                fig.savefig(save_path)
                plt.close(fig)
                torch.save(saved_args, args_save_path)
                logging.info(f'Figure saved in {str(save_path)}, args saved in {str(args_save_path)}')
                # per-layer
                for k, v in gating_data.items():
                    layer_gating_data = v
                    counts = gating_counts(layer_gating_data)
                    fig, saved_args = gating_histogram_figure(counts)
                    save_path = args.output_dir / f'layer_counts_{display_name}_{k}.png'
                    args_save_path = args.output_dir / f'layer_counts_{display_name}_{k}.pt'
                    fig.savefig(save_path)
                    plt.close(fig)
                    torch.save(saved_args, args_save_path)
                    logging.info(f'Figure saved in {str(save_path)}, args saved in {str(args_save_path)}')


if __name__ == "__main__":
    args = OmegaConf.merge(get_default_args(), OmegaConf.from_cli())
    main(args)
