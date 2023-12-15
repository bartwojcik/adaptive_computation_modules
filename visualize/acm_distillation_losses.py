import logging
from pathlib import Path

import torch
from accelerate import Accelerator
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

from architectures.acm import AdaptiveComputationMLP
from architectures.custom import simplify_mha
from datasets import DATASETS_NAME_MAP
from utils import load_model, get_loader, find_module_names, add_save_activations_hook, get_module_by_name
from visualize.cost_vs_plot import FONT_SIZE


def get_default_args():
    default_args = OmegaConf.create()
    default_args.runs_dir = Path.cwd() / 'runs'  # Root dir where experiment data was saved.
    default_args.exp_names = []  # Unique experiment names to visualize the results for (excluding exp_id).
    default_args.exp_ids = [0]  # Experiment ids.
    default_args.display_names = None  # Pretty display names that will be used when generating the plot.
    default_args.output_dir = Path.cwd() / 'figures'  # Target directory.
    default_args.batches = 0  # Number of batches to use.
    default_args.dataset = None  # Dataset to evaluate on. If "None", will use the dataset used by the run.
    default_args.dataset_args = None  # Dataset arguments. If "None", will use same args as in the training run.
    default_args.batch_size = None  # Batch size.
    default_args.use_wandb = False  # Use W&B. Will save and load the models from the W&B cloud.
    return default_args


def get_acm_errors(accelerator, base_model, acmed_model, data_loader, batches: int = 0):
    base_model.eval()
    acmed_model.eval()
    # setup acm model and representation saving hooks
    replaced_module_names = find_module_names(acmed_model, lambda _, m: isinstance(m, AdaptiveComputationMLP))
    assert len(replaced_module_names) > 0, f'{replaced_module_names=}'
    base_modules_inputs, base_modules_outputs, base_model_hook_handles = \
        add_save_activations_hook(base_model, replaced_module_names)
    acm_modules = {name: get_module_by_name(acmed_model, name) for name in replaced_module_names}
    for m in acm_modules.values():
        m.forward_mode = 'all'
    #
    batch_average_acm_errors = []
    with torch.no_grad():
        for batch, (X, _) in enumerate(data_loader):
            _ = base_model(X)
            # (batch_size, sequence_len, depth_dim, hidden_dim)
            base_output = torch.stack([base_modules_outputs[module_name] for module_name in replaced_module_names],
                                      dim=-2)
            acm_outputs = {}
            for module_name, acm_module in acm_modules.items():
                acm_outputs[module_name], _ = acm_module(base_modules_inputs[module_name][0].detach())
            # (batch_size, sequence_len, depth_dim, acm_blocks_dim, hidden_dim)
            acm_output = torch.stack([acm_outputs[module_name] for module_name in replaced_module_names],
                                     dim=-3)
            logging.info(f'{acm_output.size()=}')
            # MAE
            # average_acm_errors = torch.abs(acm_output - base_output.unsqueeze(-2)).mean(dim=-1)
            # MSE
            average_acm_errors = ((acm_output - base_output.unsqueeze(-2)) ** 2).mean(dim=-1)
            #
            average_acm_errors = accelerator.gather_for_metrics(average_acm_errors)
            batch_average_acm_errors.append(average_acm_errors.detach().cpu())
            if batch >= batches > 0:
                break
    batch_average_acm_errors = torch.cat(batch_average_acm_errors)
    return replaced_module_names, batch_average_acm_errors


def get_errors(args, accelerator, original_model, acmed_model, run_args, batches=0):
    dataset = args.dataset if args.dataset is not None else run_args.dataset
    dataset_args = args.dataset_args if args.dataset_args is not None else run_args.dataset_args
    _, _, data = DATASETS_NAME_MAP[dataset](**dataset_args)
    dataloader = get_loader(data, run_args.batch_size if args.batch_size is None else args.batch_size,
                            accelerator, shuffle=False)
    replaced_module_names, average_acm_errors = get_acm_errors(accelerator, original_model, acmed_model, dataloader,
                                                               batches)
    return replaced_module_names, average_acm_errors


def process_distillation_loss_matrix(average_acm_errors):
    # average_acm_errors is (batch_size * sequence_len, acm_blocks_dim)
    with torch.no_grad():
        # drop the first column
        average_acm_errors = average_acm_errors[:, 1:]
        # normalize each row with the maximum for each ACM block
        # max_average_acm_errors, _ = average_acm_errors.max(dim=1)
        # average_acm_errors = average_acm_errors / max_average_acm_errors.unsqueeze(1)
        # normalize each entry by improvement over its predecessor
        # average_acm_errors = average_acm_errors[:, :-1] / average_acm_errors[:, 1:]
        # log scale
        # average_acm_errors = average_acm_errors.log()
        # sort samples by median ACM error
        aggregate_for_sort, _ = average_acm_errors.median(dim=-1)
        sort_indices = torch.argsort(aggregate_for_sort, descending=True, stable=False, dim=0)
        result = torch.index_select(average_acm_errors, 0, sort_indices)
        # normalize each column with the maximum for each sample
        # result_max, _ = result.max(dim=0)
        # result = result / result_max.unsqueeze(0)
        # drop the first column
        # result = result[:, 1:]
    return result


def rep_distillation_error_figure(matrix):
    saved_args = locals()
    fig, ax = plt.subplots(figsize=(16, 16), facecolor='w', edgecolor='k')
    # ax.imshow(matrix, cmap='viridis', aspect='auto')
    # im = ax.imshow(matrix, cmap='hsv', aspect='auto')
    im = ax.imshow(matrix.cpu().numpy(), cmap='gist_rainbow_r', aspect='auto')
    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=FONT_SIZE)
    ax.set_xlabel('Learners used', fontsize=FONT_SIZE)
    ax.autoscale()
    num_learners = matrix.size(1)
    ticks = [i for i in range(num_learners)]
    labels = [int(i + 1) for i in range(num_learners)]
    ax.set_xticks(ticks, labels)
    for i, tick in enumerate(ax.xaxis.get_major_ticks()):
        tick.label.set_fontsize(FONT_SIZE)
    # ax.set_ylabel('sample', fontsize=FONT_SIZE)
    ax.set_yticks([])
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
            if run_args.model_class != 'acm':
                logging.info(f'Skipping {exp_name}')
                continue
            original_model, _, _ = load_model(args, run_args.base_on, exp_id)
            simplify_mha(original_model)
            acm_model = accelerator.prepare(acm_model)
            original_model = accelerator.prepare(original_model)
            logging.info(f'Checking representation distillation losses for: {exp_name}_{exp_id}')
            replaced_module_names, average_acm_errors = \
                get_errors(args, accelerator, original_model, acm_model, run_args, batches=args.batches)
            for i, module_name in enumerate(replaced_module_names):
                loss_matrix = process_distillation_loss_matrix(average_acm_errors.flatten(0, 1)[:, i])
                fig, saved_args = rep_distillation_error_figure(loss_matrix)
                save_path = args.output_dir / f'rep_dist_{display_name}_{module_name}.png'
                args_save_path = args.output_dir / f'rep_dist_{display_name}_{module_name}.pt'
                fig.savefig(save_path)
                plt.close(fig)
                torch.save(saved_args, args_save_path)
                logging.info(f'Figure saved in {str(save_path)}, args saved in {str(args_save_path)}')



if __name__ == "__main__":
    args = OmegaConf.merge(get_default_args(), OmegaConf.from_cli())
    main(args)
