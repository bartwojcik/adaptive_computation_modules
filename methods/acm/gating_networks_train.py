import logging
from datetime import datetime
from typing import Dict, List

import torch
from torch import nn

from architectures.acm import AdaptiveComputationMLP, add_gating_networks
from architectures.custom import simplify_mha
from common import INIT_NAME_MAP
from eval import test_classification, get_preds_acm, evaluate_classification, benchmark_acm, average_acm_flops
from train import setup_accelerator, setup_files_and_logging, setup_data, setup_optimization, setup_state, \
    TrainingContext
from utils import load_model, find_module_names, get_module_by_name, save_state, get_lrs, save_final, Mixup, \
    add_save_activations_hook, remove_hooks


class AcmLayerwiseGatingTrainingContext(TrainingContext):
    base_model: torch.nn.Module = None
    replaced_module_names: List[str] = None
    acm_modules: Dict[str, nn.Module] = None
    base_modules_inputs: Dict = None
    base_modules_outputs: Dict = None
    base_model_hook_handles: List = None


def setup_model(args, tc):
    model, base_args, _ = load_model(args, args.base_on, args.exp_id)
    base_model_base_on = base_args.base_on
    base_model, _, _ = load_model(args, base_model_base_on, args.exp_id)
    simplify_mha(base_model)
    tc.base_model = tc.accelerator.prepare(base_model)
    # create gating networks
    add_gating_networks(model, args.model_class, args.model_args)
    acm_module_names = find_module_names(model, lambda _, m: isinstance(m, AdaptiveComputationMLP))
    # (re?)-init model weights
    init_fun = INIT_NAME_MAP[args.init_fun]
    if init_fun is not None:
        init_fun(model)
    # setup model
    tc.model = tc.accelerator.prepare(model)
    unwrapped_model = tc.accelerator.unwrap_model(tc.model)
    tc.replaced_module_names = acm_module_names
    tc.acm_modules = {name: get_module_by_name(unwrapped_model, name) for name in acm_module_names}
    checkpoint_acm = False if args.checkpoint_acm is None else args.checkpoint_acm
    if checkpoint_acm is True:
        logging.info(f'Setting checkpoint mode for ACM forward to save memory.')
        for acm_module in tc.acm_modules.values():
            acm_module.checkpoint = checkpoint_acm


def set_for_train_iteration(tc):
    tc.model.eval()
    tc.model.requires_grad_(False)
    for m in tc.model.modules():
        if isinstance(m, AdaptiveComputationMLP):
            assert m.gating_network is not None
            m.gating_network.train()
            m.gating_network.requires_grad_(True)
            m.forward_mode = 'all'


def setup_for_training(tc):
    set_for_train_iteration(tc)
    unwrapped_base_model = tc.accelerator.unwrap_model(tc.base_model)
    tc.base_modules_inputs, tc.base_modules_outputs, tc.base_model_hook_handles = \
        add_save_activations_hook(unwrapped_base_model, tc.replaced_module_names)


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
        # (save outputs with hooks)
        set_for_train_iteration(tc)
        with torch.no_grad():
            tc.base_model(X)
        # iterate over each layer, calculate loss for each gating network individually
        module_losses = []
        for module_name in tc.replaced_module_names:
            module = tc.acm_modules[module_name]
            gating_module = module.gating_network
            original_input = tc.base_modules_inputs[module_name][0].detach()
            original_output = tc.base_modules_outputs[module_name].detach()
            # construct a label for the gating network
            with torch.no_grad():
                output, gating_dict = module(original_input)
                errors = torch.linalg.norm(output - original_output.unsqueeze(-2), ord=2, dim=-1)
                # relative error
                rel_improvement = errors[..., :-1] / errors[..., 1:]
                # selects all the choices that satisfy the gating error factor condition
                labels = (rel_improvement < args.gating_error_factor)
                # label is the smallest number of learners choice that satisfies the above condition
                labels = labels.double().argmax(-1).detach()
                # relative errors is num_learners - 1, so we need to add one to the result
                # also, 0 means we did not match anything for the condition, and need to set the last learner as label
                labels = torch.where(labels == 0, module.num_choices - 1, labels + 1)
                if tc.accelerator.is_main_process:
                    if tc.state.current_batch in tc.eval_batch_list:
                        bincount = labels.flatten().bincount(minlength=module.num_choices)
                        for i in range(module.num_choices):
                            tc.writer.add_scalar(f'Train/Module {module_name} label {i} occurrences', bincount[i],
                                                 global_step=tc.state.current_batch)
            gating_output = gating_module(original_input) + 1e-20
            module_losses.append(tc.criterion(gating_output.flatten(0, 1).log(), labels.flatten(0, 1)))
            del original_input, original_output, output, gating_output, gating_dict, labels, errors, rel_improvement
        loss = torch.stack(module_losses).mean()
        if tc.accelerator.is_main_process:
            tc.writer.add_scalar(f'Train/Gating Loss', loss.item(), global_step=tc.state.current_batch)
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
            remove_hooks(tc.base_model_hook_handles)
            final_results = {}
            final_results['args'] = args
            final_results['final_score'] = test_acc
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
            tc.writer.add_scalar('Eval/Gating to average FLOPS fraction',
                                 total_gating_flops / average_model_cost,
                                 global_step=tc.state.current_batch)
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
    tc = AcmLayerwiseGatingTrainingContext()
    setup_accelerator(args, tc)
    setup_files_and_logging(args, tc)
    setup_data(args, tc)
    setup_model(args, tc)
    setup_for_training(tc)
    setup_optimization(args, tc)
    setup_state(tc)
    training_loop(args, tc)
    final_eval(args, tc)
