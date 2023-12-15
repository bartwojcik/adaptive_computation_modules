import logging
from pathlib import Path

import torch
from accelerate import Accelerator
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.utils import draw_segmentation_masks

from architectures.acm import AdaptiveComputationMLP
from datasets import DATASETS_NAME_MAP
from eval import benchmark_acm
from utils import load_model, get_loader
from visualize.acm_gating_distribution import is_gated_acm


def get_default_args():
    default_args = OmegaConf.create()
    default_args.runs_dir = Path.cwd() / 'runs'  # Root dir where experiment data was saved.
    default_args.exp_names = []  # Unique experiment names to visualize the results for (excluding exp_id).
    default_args.exp_ids = [0]  # Experiment ids.
    default_args.display_names = None  # Pretty display names that will be used when generating the plot.
    default_args.output_dir = Path.cwd() / 'figures'  # Target directory.
    default_args.data_indices = 0  # Indices of the data samples to display the patches for.
    default_args.num_cheapest = None  # Number of data samples that consumed the least amount of compute.
    default_args.num_costliest = None  # Number of data samples that consumed the most amount of compute.
    default_args.dataset = None  # Dataset to evaluate on. If "None", will use the dataset used by the run.
    default_args.dataset_args = None  # Dataset arguments. If "None", will use same args as in the training run.
    default_args.batch_size = None  # Batch size.
    default_args.use_wandb = False  # Use W&B. Will save and load the models from the W&B cloud.
    return default_args


def get_normalization_transform(data):
    if isinstance(data.transforms.transform, transforms.Compose):
        transforms_set = data.transforms.transform.transforms
    else:
        transforms_set = [data.transforms.transform]
    for transform in transforms_set:
        if isinstance(transform, transforms.Normalize):
            return transform


def set_for_eval(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, AdaptiveComputationMLP):
            m.forward_mode = 'gated'


def get_acm_costs_tensor(accelerator, model, loader):
    unwrapped_model = accelerator.unwrap_model(model)
    _, block_costs, _, _ = benchmark_acm(unwrapped_model, loader)
    x, _ = next(iter(loader))
    x = x[:1]
    set_for_eval(model)
    sample, sample_gating_data = model(x, return_gating_data=True)
    acm_costs = []
    for k, block_cost in block_costs.items():
        num_choices = sample_gating_data[k].size(-1)
        learner_costs = torch.tensor(block_cost, device=sample.device).unsqueeze(0) * \
                        torch.arange(0, num_choices, device=sample.device)
        acm_costs.append(learner_costs)
    acm_costs_tensor = torch.stack(acm_costs, 0)
    return acm_costs_tensor


def compute_spatial_load(accelerator, model, loader, gating_tensor):
    acm_costs_tensor = get_acm_costs_tensor(accelerator, model, loader).cpu()
    max_token_cost = acm_costs_tensor[:, -1].sum()
    spatial_load = torch.einsum('nsag,ag->ns', gating_tensor, acm_costs_tensor) / max_token_cost
    return spatial_load


def merge_by_costs(tensor_dict, costs, x, y, y_pred, gating_tensor, number_of_samples, descending):
    if 'costs' not in tensor_dict:
        tensor_dict['costs'] = costs.cpu()
        tensor_dict['x'] = x.cpu()
        tensor_dict['y'] = y.cpu()
        tensor_dict['y_pred'] = y_pred.cpu()
        tensor_dict['gating_tensor'] = gating_tensor.cpu()
    else:
        # concatenate everything
        tensor_dict['costs'] = torch.cat([tensor_dict['costs'], costs.cpu()], dim=0)
        tensor_dict['x'] = torch.cat([tensor_dict['x'], x.cpu()], dim=0)
        tensor_dict['y'] = torch.cat([tensor_dict['y'], y.cpu()], dim=0)
        tensor_dict['y_pred'] = torch.cat([tensor_dict['y_pred'], y_pred.cpu()], dim=0)
        tensor_dict['gating_tensor'] = torch.cat([tensor_dict['gating_tensor'], gating_tensor.cpu()], dim=0)
    # sort all tensors by compute spent on the sample, and then discard samples
    tensor_dict['costs'], indices = torch.sort(tensor_dict['costs'], 0, descending=descending)
    tensor_dict['costs'] = tensor_dict['costs'][:number_of_samples]
    tensor_dict['x'] = tensor_dict['x'][indices][:number_of_samples]
    tensor_dict['y'] = tensor_dict['y'][indices][:number_of_samples]
    tensor_dict['y_pred'] = tensor_dict['y_pred'][indices][:number_of_samples]
    tensor_dict['gating_tensor'] = tensor_dict['gating_tensor'][indices][:number_of_samples]


def merge_selected_samples(tensor_dict, costs, x, y, y_pred, gating_tensor, indices):
    if 'costs' not in tensor_dict:
        tensor_dict['costs'] = costs[indices].cpu()
        tensor_dict['x'] = x[indices].cpu()
        tensor_dict['y'] = y[indices].cpu()
        tensor_dict['y_pred'] = y_pred[indices].cpu()
        tensor_dict['gating_tensor'] = gating_tensor[indices].cpu()
    else:
        # concatenate everything
        tensor_dict['costs'] = torch.cat([tensor_dict['costs'], costs[indices].cpu()], dim=0)
        tensor_dict['x'] = torch.cat([tensor_dict['x'], x[indices].cpu()], dim=0)
        tensor_dict['y'] = torch.cat([tensor_dict['y'], y[indices].cpu()], dim=0)
        tensor_dict['y_pred'] = torch.cat([tensor_dict['y_pred'], y_pred[indices].cpu()], dim=0)
        tensor_dict['gating_tensor'] = torch.cat([tensor_dict['gating_tensor'], gating_tensor[indices].cpu()], dim=0)


def process_dataset(accelerator, model, data_loader, number_with_least_compute, number_with_most_compute, data_indices):
    data_indices = set(data_indices)
    least_compute = {}
    most_compute = {}
    selected_samples = {}
    acm_costs_tensor = get_acm_costs_tensor(accelerator, model, data_loader)
    set_for_eval(model)
    current_index = 0
    with torch.no_grad():
        for x, y in data_loader:
            y_pred, gating_data = model(x, return_gating_data=True)
            gating_tensor = torch.stack([v for v in gating_data.values()], dim=2)
            # calculate computational cost for each sample
            sample_costs = torch.einsum('nsag,ag->n', gating_tensor, acm_costs_tensor)
            # merge current batch to find samples with least and most compute
            merge_by_costs(least_compute, sample_costs, x, y, y_pred, gating_tensor, number_with_least_compute, False)
            merge_by_costs(most_compute, sample_costs, x, y, y_pred, gating_tensor, number_with_most_compute, True)
            # accumulate "selected" images
            indices_current_batch = set(range(current_index, current_index + x.size(0)))
            current_index += x.size(0)
            indices_to_select = data_indices & indices_current_batch
            indices_to_select = [i % x.size(0) for i in indices_to_select]
            if len(indices_to_select) > 0:
                indices_to_select = torch.tensor(indices_to_select, device=x.device)
                merge_selected_samples(selected_samples, sample_costs, x, y, y_pred, gating_tensor, indices_to_select)
            if number_with_most_compute == 0 and \
                    number_with_least_compute == 0 and \
                    selected_samples['x'].size(0) >= len(data_indices):
                break
    return selected_samples, least_compute, most_compute


def setup_and_process(args, acm_model, run_args, accelerator):
    dataset = args.dataset if args.dataset is not None else run_args.dataset
    dataset_args = args.dataset_args if args.dataset_args is not None else run_args.dataset_args
    _, _, data = DATASETS_NAME_MAP[dataset](**dataset_args)
    logging.info(f'Testset size: {len(data)}')
    dataloader = get_loader(data, run_args.batch_size if args.batch_size is None else args.batch_size, accelerator,
                            shuffle=False)
    normalization_transform = get_normalization_transform(data)
    selected_samples, least_compute, most_compute = process_dataset(accelerator,
                                                                    acm_model,
                                                                    dataloader,
                                                                    args.num_cheapest,
                                                                    args.num_costliest,
                                                                    args.data_indices)
    patch_size = acm_model.patch_size
    return selected_samples, least_compute, most_compute, patch_size, normalization_transform, dataloader


def denormalize_image(image, normalization_transform):
    mean = normalization_transform.mean
    std = normalization_transform.std
    de_mean = [-m / s for m, s in zip(mean, std)]
    de_std = [1.0 / s for s in std]
    denormalized_image = transforms.Normalize(de_mean, de_std)(image)
    return denormalized_image


def prepare_image_with_patch_selection(image, g_x, patch_size, normalization_transform):
    # g_x should be of size (seq_len) by now
    assert g_x.dim() == 1, f'{g_x.size()=}'
    assert image.dim() == 3
    assert image.size(0) == 3
    # denormalize the image data
    if normalization_transform is not None:
        image = denormalize_image(image, normalization_transform)
    image = (image.clone() * 255).to(torch.uint8)
    patches_in_row = image.size(-1) // patch_size
    for token_index, token_weight in enumerate(g_x):
        # class token is the first in the sequence
        if token_index > 0:
            token_index -= 1
            patch_x, patch_y = divmod(token_index, patches_in_row)
            from_x, to_x = patch_x * patch_size, (patch_x + 1) * patch_size
            from_y, to_y = patch_y * patch_size, (patch_y + 1) * patch_size
            mask = torch.zeros(1, image.size(-2), image.size(-1), device=image.device, dtype=torch.bool)
            token_value = token_weight.item()
            mask[0, from_x:to_x, from_y:to_y] = True if token_value > 0.0 else False
            image = draw_segmentation_masks(image, mask, alpha=token_value, colors='red')
    return image.permute(1, 2, 0).numpy()


def prepare_patch_selection_heatmap(image, g_x, patch_size):
    # g_x should be of size (seq_len) by now
    assert g_x.dim() == 1, f'{g_x.size()=}'
    assert image.dim() == 3
    assert image.size(0) == 3
    patches_in_row = image.size(-1) // patch_size
    heatmap = torch.empty(patches_in_row, patches_in_row, device=image.device)
    for token_index, token_weight in enumerate(g_x):
        # class token is the first in the sequence
        if token_index > 0:
            token_index -= 1
            patch_x, patch_y = divmod(token_index, patches_in_row)
            heatmap[patch_x, patch_y] = token_weight
    return heatmap.numpy()


def prepare_image(image, normalization_transform):
    image = denormalize_image(image, normalization_transform)
    image = image.permute(1, 2, 0).numpy()
    return image


def generate_spatial_load_figure(x, pred, label, spatial_load, patch_size, normalization_transform, mode):
    # spatial_load should be of size (seq_len)
    assert spatial_load.dim() == 1, f'{spatial_load.size()=}'
    if mode == 'mask':
        image = prepare_image_with_patch_selection(x, spatial_load, patch_size, normalization_transform)
        fig, ax = plt.subplots(figsize=(16, 16))
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.axis('off')
        # fig.suptitle(f'prediction: {pred.argmax()} label: {label}')
        fig.set_tight_layout(True)
        return fig
    elif mode == 'separate':
        image = prepare_image(x, normalization_transform)
        selection_heatmap = prepare_patch_selection_heatmap(x, spatial_load, patch_size)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(32, 16))
        ax1.imshow(image)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_yticklabels([])
        ax1.set_xticklabels([])
        ax1.axis('off')
        ax2.imshow(selection_heatmap, cmap='inferno', interpolation='nearest', vmin=0.0, vmax=1.0)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_yticklabels([])
        ax2.set_xticklabels([])
        ax2.axis('off')
        # fig.suptitle(f'prediction: {pred.argmax()} label: {label}')
        fig.set_tight_layout(True)
        return fig
    elif mode == 'image_only':
        image = prepare_image(x, normalization_transform)
        fig, ax = plt.subplots(figsize=(16, 16))
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.axis('off')
        # fig.suptitle(f'prediction: {pred.argmax()} label: {label}')
        fig.set_tight_layout(True)
        return fig


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
                logging.info(f'Generating patch selection plots for: {exp_name}_{exp_id}')
                selected_samples, least_compute, most_compute, patch_size, normalization_transform, loader = \
                    setup_and_process(args, acm_model, run_args, accelerator)
                assert selected_samples['x'].size(0) == len(args.data_indices), f'{selected_samples}'
                named_sets = [('selected', selected_samples), ('cheapest', least_compute), ('costliest', most_compute)]
                for name, tensor_set in named_sets:
                    images = tensor_set['x']
                    preds = tensor_set['y_pred']
                    labels = tensor_set['y']
                    gating_tensor = tensor_set['gating_tensor']
                    spatial_load = compute_spatial_load(accelerator, acm_model, loader, gating_tensor)
                    for j in range(images.size(0)):
                        for mode in ['image_only', 'mask', 'separate']:
                        # for mode in ['image_only', 'mask']:
                            fig = generate_spatial_load_figure(images[j],
                                                               preds[j],
                                                               labels[j],
                                                               spatial_load[j],
                                                               patch_size,
                                                               normalization_transform,
                                                               mode)
                            save_path = args.output_dir / f'{display_name}_spatial_load_{name}_{j}_{mode}.png'
                            fig.savefig(save_path)
                            logging.info(f'Figure saved in {str(save_path)}')
                            plt.close(fig)


if __name__ == "__main__":
    args = OmegaConf.merge(get_default_args(), OmegaConf.from_cli())
    main(args)
