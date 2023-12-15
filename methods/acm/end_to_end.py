import logging
import math
from datetime import datetime
from typing import Dict, List

import torch
from torch import nn

from architectures.acm import AdaptiveComputationMLP
from common import INIT_NAME_MAP
from eval import test_classification, get_preds_acm, evaluate_classification, benchmark_acm, average_acm_flops
from train import setup_accelerator, setup_files_and_logging, setup_data, setup_optimization, setup_state, \
    TrainingContext
from utils import load_model, find_module_names, get_module_by_name, save_state, get_lrs, \
    save_final, Mixup


class AcmFinetuningContext(TrainingContext):
    replaced_module_names: List[str] = None
    acm_modules: Dict[str, nn.Module] = None
    block_costs: Dict = None


def setup_model(args, tc):
    model, base_args, _ = load_model(args, args.base_on, args.exp_id)
    acm_module_names = find_module_names(model, lambda _, m: isinstance(m, AdaptiveComputationMLP))
    # (re?)-init model weights
    init_fun = INIT_NAME_MAP[args.init_fun]
    if init_fun is not None:
        init_fun(model)
    # setup model
    tc.model = tc.accelerator.prepare(model)
    tc.replaced_module_names = acm_module_names
    unwrapped_model = tc.accelerator.unwrap_model(tc.model)
    tc.acm_modules = {name: get_module_by_name(unwrapped_model, name) for name in acm_module_names}
    checkpoint_acm = False if args.checkpoint_acm is None else args.checkpoint_acm
    if checkpoint_acm is True:
        logging.info(f'Setting checkpoint mode for ACM forward to save memory.')
        for acm_module in tc.acm_modules.values():
            acm_module.checkpoint = checkpoint_acm


def set_for_train_iteration(args, tc):
    if args.acm_finetune_mode == 'entire_model':
        tc.model.train()
        tc.model.requires_grad_(True)
        for m in tc.model.modules():
            if isinstance(m, AdaptiveComputationMLP):
                assert m.gating_network is not None
                m.forward_mode = 'gated'
    elif args.acm_finetune_mode == 'without_gates':
        tc.model.train()
        tc.model.requires_grad_(True)
        for m in tc.model.modules():
            if isinstance(m, AdaptiveComputationMLP):
                assert m.gating_network is not None
                m.forward_mode = 'gated'
                m.gating_network.eval()
                m.gating_network.requires_grad_(False)
    elif args.acm_finetune_mode in ['learners_only', 'acms_only']:
        tc.model.eval()
        tc.model.requires_grad_(False)
        for m in tc.model.modules():
            if isinstance(m, AdaptiveComputationMLP):
                assert m.gating_network is not None
                m.forward_mode = 'gated'
                if args.acm_finetune_mode == 'learners_only':
                    m.train()
                    m.requires_grad_(True)
                    m.gating_network.eval()
                    m.gating_network.requires_grad_(False)
                elif args.acm_finetune_mode == 'acms_only':
                    m.train()
                    m.requires_grad_(True)
    else:
        raise ValueError(f'Invalid value {args.acm_finetune_mode=}')


def setup_for_training(args, tc):
    set_for_train_iteration(args, tc)
    unwrapped_model = tc.accelerator.unwrap_model(tc.model)
    _, tc.block_costs, _, _ = benchmark_acm(unwrapped_model, tc.test_loader)


def set_for_eval_with_gating(tc):
    tc.model.eval()
    for m in tc.model.modules():
        if isinstance(m, AdaptiveComputationMLP):
            m.forward_mode = 'gated'


def in_training_eval(args, tc):
    if tc.state.current_batch in tc.eval_batch_list:
        if tc.accelerator.is_main_process:
            tc.writer.add_scalar('Train/Progress',
                                 tc.state.current_batch / tc.last_batch,
                                 global_step=tc.state.current_batch)
        logging.info(f'Testing on testset on {args.eval_batches} batches.')
        set_for_eval_with_gating(tc)
        test_loss, test_acc = test_classification(tc.accelerator,
                                                  tc.model,
                                                  tc.test_loader,
                                                  tc.criterion_type,
                                                  batches=args.eval_batches)
        logging.info(f'Testing on trainset on {args.eval_batches} batches.')
        train_loss, train_acc = test_classification(tc.accelerator,
                                                    tc.model,
                                                    tc.train_eval_loader,
                                                    tc.criterion_type,
                                                    batches=args.eval_batches)
        if tc.accelerator.is_main_process:
            tc.writer.add_scalar(f'Eval/Test loss', test_loss,
                                 global_step=tc.state.current_batch)
            tc.writer.add_scalar(f'Eval/Test accuracy', test_acc,
                                 global_step=tc.state.current_batch)
            tc.writer.add_scalar(f'Eval/Train loss', train_loss,
                                 global_step=tc.state.current_batch)
            tc.writer.add_scalar(f'Eval/Train accuracy', train_acc,
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
        # forward
        set_for_train_iteration(args, tc)
        y_pred, gating_data = tc.model(X, return_gating_data=True)
        # loss computation
        task_loss = tc.criterion(y_pred, y)
        # get gating network outputs
        if args.routing_loss_weight is not None:
            gating_losses = []
            max_gating_cost = 0.0
            for name in tc.replaced_module_names:
                acm = tc.acm_modules[name]
                gating_output = gating_data[name]
                decoded_costs = torch.tensor(tc.block_costs[name], device=gating_output.device).unsqueeze(0) * \
                                torch.arange(0, acm.num_choices, device=gating_output.device)
                max_gating_cost += decoded_costs[-1].item()
                sample_gating_costs = torch.einsum('nsg,g->ns', gating_output, decoded_costs.to(gating_output.dtype))
                module_gating_costs = sample_gating_costs.mean()
                gating_losses.append(module_gating_costs)
            target_gating_budget = args.routing_loss_target if args.routing_loss_target is not None else 0.0
            if args.routing_loss_type == 'l1' or args.routing_loss_type is None:
                gating_loss = torch.abs((torch.stack(gating_losses).sum() / max_gating_cost) - target_gating_budget)
            elif args.routing_loss_type == 'l2':
                gating_loss = ((torch.stack(gating_losses).sum() / max_gating_cost) - target_gating_budget) ** 2
            elif args.routing_loss_type == 'l2_scaled':
                gating_loss = (((torch.stack(
                    gating_losses).sum() / max_gating_cost) - target_gating_budget) * 100.0) ** 2
            else:
                raise ValueError(f'Invalid value {args.routing_loss_type=}')
        if args.entropy_loss_weight is not None:
            gating_tensor = torch.stack([v.to(torch.float64) for v in gating_data.values()], dim=2)
            # gating_tensor is (batch_size, sequence_length, number_of_acms, learners_per_acm)
            acm_gating_data = gating_tensor.sum(dim=1)  # sums learner choices for each sequence
            gating_distribution = acm_gating_data / acm_gating_data.sum(-1, keepdim=True)  # normalizes
            entropies = -(gating_distribution * (gating_distribution + 1e-20).log()).sum(-1, keepdim=True)
            normalized_entropies = entropies / math.log(gating_distribution.size(-1))
            entropy_loss = -normalized_entropies.mean()
        if args.compute_diversity_loss_weight is not None:
            gating_tensor = torch.stack([v.to(torch.float64) for v in gating_data.values()], dim=2)
            # gating_tensor is (batch_size, sequence_length, number_of_acms, learners_per_acm)
            gating_distribution = (gating_tensor *
                                   torch.arange(0, gating_tensor.size(-1),
                                                device=gating_tensor.device)
                                   .reshape(1, 1, 1, -1)).mean(dim=(1, 2, 3))
            sample_differences = torch.cdist(gating_distribution.reshape(1, -1, 1),
                                             gating_distribution.reshape(1, -1, 1),
                                             p=1.0)
            compute_diversity_loss = -sample_differences.mean() * 2
        loss = task_loss
        if args.routing_loss_weight is not None:
            loss = loss + args.routing_loss_weight * gating_loss
            if tc.accelerator.is_main_process:
                tc.writer.add_scalar(f'Train/Gating loss', gating_loss.item(), global_step=tc.state.current_batch)
        if args.entropy_loss_weight is not None:
            loss = loss + args.entropy_loss_weight * entropy_loss
            if tc.accelerator.is_main_process:
                tc.writer.add_scalar(f'Train/Entropy loss', entropy_loss.item(), global_step=tc.state.current_batch)
        if args.compute_diversity_loss_weight is not None:
            loss = loss + args.compute_diversity_loss_weight * compute_diversity_loss
            if tc.accelerator.is_main_process:
                tc.writer.add_scalar(f'Train/Compute diversity loss', compute_diversity_loss.item(),
                                     global_step=tc.state.current_batch)
        if tc.accelerator.is_main_process:
            tc.writer.add_scalar(f'Train/Task loss', task_loss.item(), global_step=tc.state.current_batch)
            tc.writer.add_scalar(f'Train/Loss', loss.item(), global_step=tc.state.current_batch)
        # gradient computation
        tc.optimizer.zero_grad(set_to_none=True)
        tc.accelerator.backward(loss)
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
        # bookkeeping
        tc.state.current_batch += 1


def final_eval(args, tc):
    if not tc.final_path.exists():
        if tc.accelerator.is_main_process:
            save_state(tc.accelerator, tc.state_path)
        # test on testset
        set_for_eval_with_gating(tc)
        preds, labels, gating_data = get_preds_acm(tc.accelerator,
                                                   tc.model,
                                                   tc.test_loader)
        if tc.accelerator.is_main_process:
            test_loss, test_acc = evaluate_classification(preds, labels, tc.criterion_type)
            tc.writer.add_scalar(f'Eval/Test loss', test_loss,
                                 global_step=tc.state.current_batch)
            tc.writer.add_scalar(f'Eval/Test accuracy', test_acc,
                                 global_step=tc.state.current_batch)
            final_results = {}
            final_results['args'] = args
            final_results['final_score'] = test_acc
            logging.info(f'Final score: {test_acc}')
            final_results['final_loss'] = test_loss
            unwrapped_model = tc.accelerator.unwrap_model(tc.model)
            final_results['model_state'] = unwrapped_model.state_dict()
            # benchmark ACM cost
            # for each ACM the cost of entire ACM, a single learner, and the gating network
            # is saved in addition to the cost of the whole model
            set_for_eval_with_gating(tc)
            cost_without_acms, learner_costs, gating_costs, model_params = benchmark_acm(unwrapped_model,
                                                                                         tc.test_loader)
            average_model_cost, averaged_acm_costs, total_gating_flops = average_acm_flops(cost_without_acms,
                                                                                           learner_costs, gating_costs,
                                                                                           gating_data)
            final_results['model_flops'] = average_model_cost
            final_results['model_params'] = dict(model_params)
            tc.writer.add_scalar('Eval/Model FLOPs', average_model_cost, global_step=tc.state.current_batch)
            tc.writer.add_scalar('Eval/Model Params', model_params[''], global_step=tc.state.current_batch)
            for k, v in averaged_acm_costs.items():
                tc.writer.add_scalar(f'Module eval/ACM {k} average FLOPs', v, global_step=tc.state.current_batch)
            gating_flops_fraction = total_gating_flops / average_model_cost
            tc.writer.add_scalar('Eval/Gating to average FLOPS fraction',
                                 gating_flops_fraction,
                                 global_step=tc.state.current_batch)
            logging.info(f'Gating FLOPs fraction: {gating_flops_fraction}')
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
    tc = AcmFinetuningContext()
    setup_accelerator(args, tc)
    setup_files_and_logging(args, tc)
    setup_data(args, tc)
    setup_model(args, tc)
    setup_for_training(args, tc)
    setup_optimization(args, tc)
    setup_state(tc)
    training_loop(args, tc)
    final_eval(args, tc)
