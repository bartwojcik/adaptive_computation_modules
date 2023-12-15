import logging
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Type

import torch
from omegaconf import OmegaConf
from torch import nn

from architectures.acm import acmize_vit, AdaptiveComputationMLP
from architectures.custom import simplify_mha
from common import get_default_args, INIT_NAME_MAP, LOSS_NAME_MAP
from eval import test_classification, benchmark
from train import TrainingContext, setup_accelerator, setup_data, setup_optimization, setup_files_and_logging, \
    setup_state
from utils import load_model, get_module_by_name, add_save_activations_hook, save_state, get_lrs, remove_hooks, \
    save_final, Mixup


class AcmTrainingContext(TrainingContext):
    base_model: torch.nn.Module = None
    replaced_module_names: List[str] = None
    acm_modules: Dict[str, nn.Module] = None
    distill_criterion_type: Type = None
    base_modules_inputs: Dict = None
    base_modules_outputs: Dict = None
    base_model_hook_handles: List = None


def setup_model(args, tc):
    assert args.model_class == 'acm'
    base_model, base_args, _ = load_model(args, args.base_on, args.exp_id)
    simplify_mha(base_model)
    model_args = args.model_args
    example_input, _ = next(iter(tc.train_eval_loader))
    # wrap and immediately unwrap for correct device placement
    tc.base_model = tc.accelerator.prepare(base_model)
    unwrapped_base_model = tc.accelerator.unwrap_model(tc.base_model)
    tc.model, tc.replaced_module_names = acmize_vit(unwrapped_base_model, example_input=example_input, **model_args)
    tc.acm_modules = {name: get_module_by_name(tc.model, name) for name in tc.replaced_module_names}
    checkpoint_acm = False if args.checkpoint_acm is None else args.checkpoint_acm
    if checkpoint_acm is True:
        logging.info(f'Setting checkpoint mode for ACM forward to save memory.')
    for acm_module in tc.acm_modules.values():
        acm_module.checkpoint = checkpoint_acm
    detach_mode = 'no_detach' if args.acm_detach_mode is None else args.acm_detach_mode
    logging.info(f'Setting ACM train mode: {args.acm_detach_mode}')
    for acm_module in tc.acm_modules.values():
        acm_module.train_mode = detach_mode
    init_fun = INIT_NAME_MAP[args.init_fun]
    if init_fun is not None:
        init_fun(tc.model)
    tc.base_model = tc.accelerator.prepare(base_model)
    tc.model = tc.accelerator.prepare(tc.model)


def set_for_distillation_iteration(tc):
    tc.base_model.eval()
    for m in tc.model.modules():
        if isinstance(m, AdaptiveComputationMLP):
            m.forward_mode = 'all'
            m.train()


def set_for_eval(tc, blocks_to_use):
    tc.model.eval()
    for m in tc.model.modules():
        if isinstance(m, AdaptiveComputationMLP):
            m.forward_mode = blocks_to_use


def set_for_block_eval(tc):
    tc.model.eval()
    for m in tc.model.modules():
        if isinstance(m, AdaptiveComputationMLP):
            m.forward_mode = 'all'


def setup_for_training(args, tc):
    tc.distill_criterion_type = LOSS_NAME_MAP[args.acm_distill_loss_type]
    tc.base_model.eval()
    unwrapped_base_model = tc.accelerator.unwrap_model(tc.base_model)
    tc.base_modules_inputs, tc.base_modules_outputs, tc.base_model_hook_handles = \
        add_save_activations_hook(unwrapped_base_model, tc.replaced_module_names)
    tc.model.eval()
    tc.model.requires_grad_(False)
    for m in tc.model.modules():
        if isinstance(m, AdaptiveComputationMLP):
            assert m.gating_network is None
            m.train()
            m.requires_grad_(True)


def reset_optimizer_state(tc):
    tc.optimizer.state = defaultdict(dict)


def test_representation_distillation(tc,
                                     data_loader: torch.utils.data.DataLoader,
                                     batches: int = 0):
    criterion = tc.distill_criterion_type(reduction='none')
    set_for_block_eval(tc)
    with torch.no_grad():
        losses = defaultdict(list)
        for batch, (X, _) in enumerate(data_loader):
            tc.base_model(X)
            # iterate over each layer, calculate loss for each ACM individually
            for module_name in tc.replaced_module_names:
                module = tc.acm_modules[module_name]
                original_input = tc.base_modules_inputs[module_name][0].detach()
                original_output = tc.base_modules_outputs[module_name].detach()
                output, _ = module(original_input)
                output, original_output = tc.accelerator.gather_for_metrics((output, original_output))
                batch_losses = criterion(output, original_output.unsqueeze(-2).expand(-1, -1, module.num_choices, -1))
                losses[module_name].append(batch_losses.detach().mean(dim=(1, 3)))
                if batch >= batches > 0:
                    break
        for k in losses.keys():
            losses[k] = torch.cat(losses[k]).mean(dim=0)
        return losses


def in_training_eval(args, tc):
    if tc.state.current_batch in tc.eval_batch_list:
        if tc.accelerator.is_main_process:
            tc.writer.add_scalar('Train/Progress',
                                 tc.state.current_batch / tc.last_batch,
                                 global_step=tc.state.current_batch)
        for blocks_to_use in [0.25, 0.5, 0.75, 1.0]:
            if tc.accelerator.is_main_process:
                logging.info(f'Testing on testset for {blocks_to_use} ACM blocks on {args.eval_batches} batches.')
            set_for_eval(tc, blocks_to_use)
            test_loss, test_acc = test_classification(tc.accelerator,
                                                      tc.model,
                                                      tc.test_loader,
                                                      tc.criterion_type,
                                                      batches=args.eval_batches)
            if tc.accelerator.is_main_process:
                logging.info(f'Testing on trainset for {blocks_to_use} ACM blocks on {args.eval_batches} batches.')
            train_loss, train_acc = test_classification(tc.accelerator,
                                                        tc.model,
                                                        tc.train_eval_loader,
                                                        tc.criterion_type,
                                                        batches=args.eval_batches)
            if tc.accelerator.is_main_process:
                tc.writer.add_scalar(f'Eval at {blocks_to_use}/Test loss', test_loss,
                                     global_step=tc.state.current_batch)
                tc.writer.add_scalar(f'Eval at {blocks_to_use}/Test accuracy', test_acc,
                                     global_step=tc.state.current_batch)
                tc.writer.add_scalar(f'Eval at {blocks_to_use}/Train loss', train_loss,
                                     global_step=tc.state.current_batch)
                tc.writer.add_scalar(f'Eval at {blocks_to_use}/Train accuracy', train_acc,
                                     global_step=tc.state.current_batch)


def training_loop(args, tc):
    if tc.accelerator.is_main_process:
        model_saved = datetime.now()
    train_iter = iter(tc.train_loader)
    unwrapped_model = tc.accelerator.unwrap_model(tc.model)
    if args.mixup_alpha is not None or args.cutmix_alpha is not None:
        mixup_mode = 'batch' if args.mixup_mode is None else args.mixup_mode
        mixup_smoothing = 0.1 if args.mixup_smoothing is None else args.mixup_smoothing
        mixup_fn = Mixup(
            mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha, mode=mixup_mode,
            label_smoothing=mixup_smoothing, num_classes=unwrapped_model.number_of_classes)
    else:
        mixup_fn = None
    if args.acm_freeze_mode == 'incremental':
        freeze_schedule_step = tc.last_batch // args.model_args.num_blocks
        freeze_schedule = [
            round(x) for x in
            torch.arange(freeze_schedule_step, tc.last_batch, freeze_schedule_step, device='cpu').tolist()
        ]
        if tc.accelerator.is_main_process:
            logging.info(f'{freeze_schedule=}')
        first_unfrozen_learner = 0
    criterion = tc.distill_criterion_type()
    while tc.state.current_batch <= tc.last_batch:
        # save model conditionally
        if tc.accelerator.is_main_process:
            now = datetime.now()
            if (now - model_saved).total_seconds() > 60 * args.save_every:
                save_state(tc.accelerator, tc.state_path)
                model_saved = datetime.now()
        # model evaluation
        in_training_eval(args, tc)
        # batch preparation
        try:
            X, y = next(train_iter)
        except StopIteration:
            train_iter = iter(tc.train_loader)
            X, y = next(train_iter)
        if mixup_fn is not None:
            X, y = mixup_fn(X, y)
        # training step
        # (save inputs and outputs with hooks)
        set_for_distillation_iteration(tc)
        with torch.no_grad():
            tc.base_model(X)
        # iterate over each layer, calculate loss for each ACM individually
        module_losses = []
        for module_name in tc.replaced_module_names:
            module = tc.acm_modules[module_name]
            original_input = tc.base_modules_inputs[module_name][0].detach()
            original_output = tc.base_modules_outputs[module_name].detach()
            output, _ = module(original_input)
            assert not torch.any(
                torch.isnan(output)), f'NaN present in {module_name} output of batch {tc.state.current_batch}'
            module_loss = criterion(output, original_output.unsqueeze(-2).expand(-1, -1, module.num_choices, -1))
            assert not torch.any(
                torch.isnan(module_loss)), f'NaN present in {module_name} loss for batch {tc.state.current_batch}'
            module_losses.append(module_loss)
            del original_input, original_output, output
            if tc.accelerator.is_main_process:
                tc.writer.add_scalar(f'Train/Module {module_name} loss', module_loss.item(),
                                     global_step=tc.state.current_batch)
        loss = torch.stack(module_losses).mean()
        if tc.accelerator.is_main_process:
            tc.writer.add_scalar(f'Train/Rep. Distill Loss', loss.item(), global_step=tc.state.current_batch)
        loss = loss
        tc.optimizer.zero_grad(set_to_none=True)
        tc.accelerator.backward(loss)
        if args.acm_freeze_mode == 'incremental':
            # hackish way of freezing part of a weight tensor
            if tc.state.current_batch in freeze_schedule:
                # 1. mark parameters as frozen if current batch is in freeze schedule
                first_unfrozen_learner += 1
                # 2. zero out the optimizer stats when freezing any parameters
                # TODO alternatively zero them out only for the parameters being frozen
                reset_optimizer_state(tc)
                if tc.accelerator.is_main_process:
                    logging.info(f'Freezing learners up to (not including) learner {first_unfrozen_learner}.')
            # 3. zero gradient of selected parameters after backward here
            for module_name in tc.replaced_module_names:
                module = tc.acm_modules[module_name]
                module.w1.grad.view(module.hidden_dim,
                                    module.num_blocks,
                                    module.block_dim)[:, :first_unfrozen_learner] = 0.0
                if module.b1 is not None:
                    module.b1.grad.view(module.num_blocks,
                                        module.block_dim)[:first_unfrozen_learner] = 0.0
                module.w2.grad.view(module.num_blocks,
                                    module.block_dim,
                                    module.hidden_dim)[:first_unfrozen_learner] = 0.0
                # note that bias for 0 learners exists, thus the index is:
                # 0 if first_unfrozen_learner == 0
                # first_unfrozen_learner + 1 if first_unfrozen_learner > 0
                if module.b2 is not None:
                    first_for_b2 = first_unfrozen_learner + 1 if first_unfrozen_learner > 0 else 0
                    module.b2.grad.view(module.num_choices,
                                        module.hidden_dim)[:first_for_b2] = 0.0
        if args.clip_grad_norm is not None:
            total_norm = tc.accelerator.clip_grad_norm_(tc.model.parameters(), args.clip_grad_norm)
            if tc.accelerator.is_main_process:
                tc.writer.add_scalar(f'Train/Gradient norm', total_norm.item(), global_step=tc.state.current_batch)
        tc.optimizer.step()
        if tc.scheduler is not None:
            # log LRs
            if tc.accelerator.is_main_process:
                for i, lr in enumerate(get_lrs(tc.optimizer)):
                    tc.writer.add_scalar(f'Train/Group {i} LR', lr, global_step=tc.state.current_batch)
            if args.scheduler_class == 'reduce_on_plateau':
                tc.scheduler.step(loss)
            else:
                tc.scheduler.step()
        if tc.accelerator.is_main_process:
            tc.writer.add_scalar(f'Train/Loss', loss.item(), global_step=tc.state.current_batch)
        # bookkeeping
        tc.state.current_batch += 1


def final_eval(args, tc):
    if not tc.final_path.exists():
        if tc.accelerator.is_main_process:
            save_state(tc.accelerator, tc.state_path)
        # individual block losses
        # (too costly for in_training_eval)
        if tc.accelerator.is_main_process:
            logging.info(f'Testing representation distillation on testset.')
        test_losses = test_representation_distillation(tc, tc.test_loader)
        if tc.accelerator.is_main_process:
            for k, v in test_losses.items():
                for i in range(v.size(0)):
                    tc.writer.add_scalar(f'Eval ACM {k}/Test loss block {i}', v[i], global_step=tc.state.current_batch)
        # whole model testing
        for blocks_to_use in [0.25, 0.5, 0.75, 1.0]:
            set_for_eval(tc, blocks_to_use)
            if tc.accelerator.is_main_process:
                logging.info(f'Testing on testset for {blocks_to_use} ACM blocks.')
            test_loss, test_acc = test_classification(tc.accelerator,
                                                      tc.model,
                                                      tc.test_loader,
                                                      tc.criterion_type)
            if tc.accelerator.is_main_process:
                tc.writer.add_scalar(f'Eval at {blocks_to_use}/Test loss', test_loss,
                                     global_step=tc.state.current_batch)
                tc.writer.add_scalar(f'Eval at {blocks_to_use}/Test accuracy', test_acc,
                                     global_step=tc.state.current_batch)
        if tc.accelerator.is_main_process:
            final_results = {}
            final_results['args'] = args
            # saves results for 1.0 as final results!
            final_results['final_score'] = test_acc
            final_results['final_loss'] = test_loss
            tc.writer.add_scalar(f'Eval/Test loss', test_loss,
                                 global_step=tc.state.current_batch)
            tc.writer.add_scalar(f'Eval/Test accuracy', test_acc,
                                 global_step=tc.state.current_batch)
            remove_hooks(tc.base_model_hook_handles)
            unwrapped_model = tc.accelerator.unwrap_model(tc.model)
            final_results['model_state'] = unwrapped_model.state_dict()
            # save representation distillation results
            final_results['test_losses'] = test_losses
            # benchmark acm cost
            # more detailed benchmarking done in phase 2, when the gating networks are present
            set_for_eval(tc, 1.0)
            model_costs, model_params = benchmark(unwrapped_model, tc.test_loader)
            final_results['model_flops'] = model_costs.total()
            final_results['model_flops_by_module'] = dict(model_costs.by_module())
            final_results['model_flops_by_operator'] = dict(model_costs.by_operator())
            final_results['model_params'] = dict(model_params)
            tc.writer.add_scalar('Eval/Model FLOPs', model_costs.total(), global_step=tc.state.current_batch)
            tc.writer.add_scalar('Eval/Model Params', model_params[''], global_step=tc.state.current_batch)
            save_final(args, tc.final_path, final_results)


def train(args):
    logging.basicConfig(
        format=(
            '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] ' '%(message)s'
        ),
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
        force=True,
    )
    logging.info('Configured logging')
    tc = AcmTrainingContext()
    setup_accelerator(args, tc)
    setup_files_and_logging(args, tc)
    setup_data(args, tc)
    setup_model(args, tc)
    setup_for_training(args, tc)
    setup_optimization(args, tc)
    setup_state(tc)
    training_loop(args, tc)
    final_eval(args, tc)


def main():
    args = OmegaConf.merge(get_default_args(), OmegaConf.from_cli())
    train(args)


if __name__ == '__main__':
    main()
