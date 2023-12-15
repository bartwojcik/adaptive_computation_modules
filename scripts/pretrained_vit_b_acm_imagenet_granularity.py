import os
from copy import deepcopy
from pathlib import Path

import submitit

from common import get_default_args
from methods.acm.end_to_end import train as train_acm_e2e
from methods.acm.gating_networks_train import train as train_acm_gating
from methods.acm.rep_distill_train import train as train_acm
from train import train
from utils import generate_run_name, submit_job
from visualize.cost_vs_plot import get_default_args as get_default_cost_plot_args
from visualize.cost_vs_plot import main as cost_vs_plot


def main():
    # ════════════════════════ submitit setup ════════════════════════ #

    job_name = 'acm_experiments'

    account = None

    qos = 'normal'

    partition = 'batch'

    timeout = 60 * 24 * 7

    gpus_per_task = 1
    gpu_type = ''
    cpus_per_gpu = 8
    cpus_per_gpu = 16
    mem_per_gpu = '64G'

    exclude_nodes = None

    executor = submitit.AutoExecutor(folder=os.environ['LOGS_DIR'])
    executor.update_parameters(
        stderr_to_stdout=True,
        timeout_min=timeout,
        slurm_job_name=job_name,
        slurm_account=account,
        slurm_qos=qos,
        slurm_partition=partition,
        slurm_ntasks_per_node=1,
        slurm_gpus_per_task=f'{gpu_type}{gpus_per_task}',
        slurm_cpus_per_gpu=cpus_per_gpu,
        slurm_mem_per_gpu=mem_per_gpu,
    )
    if exclude_nodes is not None:
        executor.update_parameters(slurm_exclude=exclude_nodes)

    # ════════════════════════ experiment settings ════════════════════════ #

    common_args = get_default_args()
    # exp_ids = [1, 2, 3]
    exp_ids = [1]
    common_args.runs_dir = Path(os.environ['RUNS_DIR'])
    common_args.dataset = 'imagenet'
    common_args.dataset_args = {}
    common_args.dataset_args.variant = 'deit3_rrc'
    common_args.mixup_alpha = 0.8
    common_args.cutmix_alpha = 1.0
    common_args.mixup_smoothing = 0.1
    common_args.batch_size = 128
    common_args.loss_type = 'ce'
    common_args.loss_args = {}
    common_args.optimizer_class = 'adam'
    common_args.optimizer_args = {}
    common_args.optimizer_args.lr = 0.001
    common_args.optimizer_args.weight_decay = 0.0
    common_args.scheduler_class = 'cosine'
    common_args.scheduler_args = {}
    common_args.scheduler_args.eta_min = 1e-6
    common_args.clip_grad_norm = 1.0
    common_args.epochs = 5
    common_args.eval_points = 20
    common_args.use_wandb = True
    common_args.mixed_precision = None

    jobs = []
    run_to_job_map = {}
    exp_names = []
    display_names = []

    # ════════════════════════ base model settings ════════════════════════ #

    base_model_args = deepcopy(common_args)
    base_model_args.model_class = 'tv_vit_b_16'
    base_model_args.model_args = {}
    base_model_args.epochs = 0  # pretrained
    base_model_args.eval_points = 0

    # ════════════════════════ get pretrained base models ════════════════════════ #

    for exp_id in exp_ids:
        args = deepcopy(base_model_args)
        args.exp_id = exp_id
        job = submit_job(executor, train, args, num_gpus=gpus_per_task)
        jobs.append(job)
        exp_name, run_name = generate_run_name(args)
        run_to_job_map[run_name] = job
    exp_names.append(exp_name)
    display_names.append(f'ViT-B')
    base_exp_name = exp_name

    # ════════════════════════ ACM model settings ════════════════════════ #

    acm_gpus_per_task = 2
    executor.update_parameters(slurm_gpus_per_task=f'{gpu_type}{acm_gpus_per_task}')

    acm_args = deepcopy(common_args)
    acm_args.model_class = 'acm'
    acm_args.model_args = {}
    acm_args.checkpoint_acm = True
    acm_args.acm_distill_loss_type = 'mse'
    acm_args.eval_points = 5

    # ════════════════════════ train ACMed models ════════════════════════ #

    base_acm_exp_names = []
    num_learners_dict = {}
    for opt_class, bs, lr, wd, dropout, bias, detach, ffn, o_proj, q_proj, k_proj, v_proj, total_flops_factor, num_blocks, epochs in [
        # MHA and FFNs
        ('adam', 64, 0.001, 0.0, 0.0, 'hidden_only', 'no_detach', True, True, True, True, True, 1.0, 2, 2),
        ('adam', 64, 0.001, 0.0, 0.0, 'hidden_only', 'no_detach', True, True, True, True, True, 1.0, 4, 2),
        ('adam', 64, 0.001, 0.0, 0.0, 'hidden_only', 'no_detach', True, True, True, True, True, 1.0, 8, 2),
        ('adam', 64, 0.001, 0.0, 0.0, 'hidden_only', 'no_detach', True, True, True, True, True, 1.0, 16, 2),
    ]:
        for exp_id in exp_ids:
            args = deepcopy(acm_args)
            args.base_on = base_exp_name
            args.exp_id = exp_id
            args.epochs = epochs
            args.batch_size = bs
            args.optimizer_class = opt_class
            args.optimizer_args.lr = lr
            args.optimizer_args.weight_decay = wd
            args.acm_detach_mode = detach
            args.model_args.ffn = ffn
            args.model_args.o_proj = o_proj
            args.model_args.q_proj = q_proj
            args.model_args.k_proj = k_proj
            args.model_args.v_proj = v_proj
            args.model_args.num_blocks = num_blocks
            args.model_args.block_flops_factor = total_flops_factor / num_blocks
            args.model_args.dropout = dropout
            args.model_args.attention_dropout = dropout
            args.model_args.bias = bias
            args.model_args.attention_bias = bias
            exp_name, run_name = generate_run_name(args)
            base_run_name = f'{base_exp_name}_{exp_id}'
            if base_run_name in run_to_job_map:
                dependency_str = f"afterany:{run_to_job_map[base_run_name].job_id}"
                executor.update_parameters(slurm_additional_parameters={"dependency": dependency_str})
            else:
                executor.update_parameters(slurm_additional_parameters={})
            # job = submit_job(executor, train_acm, args, num_gpus=acm_gpus_per_task)
            # jobs.append(job)
            # run_to_job_map[run_name] = job
            exp_names.append(exp_name)
            base_acm_exp_names.append(exp_name)
            display_names.append(f'ACM (phase I; {num_blocks} learners)')
            num_learners_dict[exp_name] = num_blocks

    # ════════════════════════ ACM layerwise gating networks settings ════════════════════════ #

    gating_args = deepcopy(common_args)
    gating_args.loss_type = 'nll'
    gating_args.checkpoint_acm = True
    gating_args.clip_grad_norm = 1.0
    min_choice_dict = {'q_proj': 1, 'k_proj': 1, 'v_proj': 1, 'o_proj': 1}
    gating_args.eval_points = 5

    # ════════════════════════ train ACM gating networks in a layerwise manner ════════════════════════ #

    base_gacm_exp_names = []
    for base_on_exp_name in base_acm_exp_names:
        for gating_type, gating_model_args, epochs, bs, opt_class, lr, gating_error_factor in [
            ('acm_deep_gumbel', {'gs_tau': 0.8, 'min_choice': min_choice_dict,
                                 'depth': 3, 'gating_hidden_dim': 16}, 1, 64, 'adam', 0.01, 1.2),
        ]:
            for exp_id in exp_ids:
                args = deepcopy(gating_args)
                args.base_on = base_on_exp_name
                args.exp_id = exp_id
                args.model_class = gating_type
                args.model_args = gating_model_args
                args.gating_error_factor = gating_error_factor
                args.epochs = epochs
                args.batch_size = bs
                args.optimizer_class = opt_class
                args.optimizer_args.lr = lr
                exp_name, run_name = generate_run_name(args)
                base_run_name = f'{base_on_exp_name}_{exp_id}'
                if base_run_name in run_to_job_map:
                    dependency_str = f"afterany:{run_to_job_map[base_run_name].job_id}"
                    executor.update_parameters(slurm_additional_parameters={"dependency": dependency_str})
                else:
                    executor.update_parameters(slurm_additional_parameters={})
                # job = submit_job(executor, train_acm_gating, args, num_gpus=acm_gpus_per_task)
                # jobs.append(job)
                # run_to_job_map[run_name] = job
                exp_names.append(exp_name)
                base_gacm_exp_names.append(exp_name)
                display_names.append(f'ACM (phase II; {num_learners_dict[base_on_exp_name]} learners)')
                num_learners_dict[exp_name] = num_learners_dict[base_on_exp_name]

    # ════════════════════════ ACM finetuning settings ════════════════════════ #

    acm_gpus_per_task = 2
    executor.update_parameters(slurm_gpus_per_task=f'{gpu_type}{acm_gpus_per_task}')

    finetune_args = deepcopy(common_args)
    finetune_args.model_class = 'facm'
    finetune_args.checkpoint_acm = True
    finetune_args.acm_finetune_mode = 'entire_model'
    # bf16 may be risky!
    finetune_args.mixed_precision = 'bf16'

    # ════════════════════════ finetune ACM networks ════════════════════════ #
    for base_on_exp_name in base_gacm_exp_names:
        for epochs, bs, optimizer_class, lr, routing_loss_type, routing_loss_weight, routing_loss_target, \
                entropy_loss_weight, compute_diversity_loss_weight in [
            (5, 256, 'adam', 1e-4, 'l1', 1e-1, 0.60, 0.05, 0.05),
        ]:
            for exp_id in exp_ids:
                args = deepcopy(finetune_args)
                args.base_on = base_on_exp_name
                args.exp_id = exp_id
                args.epochs = epochs
                args.eval_points = epochs + 1
                args.batch_size = bs
                args.optimizer_class = optimizer_class
                args.optimizer_args.lr = lr
                args.routing_loss_type = routing_loss_type
                args.routing_loss_weight = routing_loss_weight
                args.routing_loss_target = routing_loss_target
                args.entropy_loss_weight = entropy_loss_weight
                args.compute_diversity_loss_weight = compute_diversity_loss_weight
                exp_name, run_name = generate_run_name(args)
                base_run_name = f'{base_on_exp_name}_{exp_id}'
                if base_run_name in run_to_job_map:
                    dependency_str = f"afterany:{run_to_job_map[base_run_name].job_id}"
                    executor.update_parameters(slurm_additional_parameters={"dependency": dependency_str})
                else:
                    executor.update_parameters(slurm_additional_parameters={})
                job = submit_job(executor, train_acm_e2e, args, num_gpus=acm_gpus_per_task)
                jobs.append(job)
                run_to_job_map[run_name] = job
                exp_names.append(exp_name)
                num_learners = num_learners_dict[base_on_exp_name]
                display_names.append(f'ACM ({num_learners} learners)')

    # ═════════════════════════════════════════════════════════ #

    print(f"Exp names: {exp_names}")
    print(f"Display names: {display_names}")
    print(f"SLURM JIDs: {[job.job_id for job in jobs]}")
    print(f"Run to JID mapping: {[(k, v.job_id) for k, v in run_to_job_map.items()]}")

    # ════════════════════════ plots ════════════════════════ #

    if len(jobs) > 0:
        dependency_str = f'afterany:{":".join(job.job_id for job in jobs)}'  # wait for all jobs to finish before plotting
        executor.update_parameters(slurm_gpus_per_task=f'{gpu_type}1',
                                   slurm_additional_parameters={"dependency": dependency_str})
    else:
        executor.update_parameters(slurm_gpus_per_task=f'{gpu_type}1')

    out_dir_name = f"{common_args.dataset}_pretrained_vit_b_granularity"
    output_dir = Path(os.environ["RESULTS_DIR"]) / out_dir_name

    # ════════════════════════ plot cost vs acc ════════════════════════ #

    selected_pairs = [(exp_name, display_name) for exp_name, display_name in zip(exp_names, display_names)
                      if 'phase' not in display_name]
    selected_exp_names, selected_display_names = zip(*selected_pairs)

    plot_args = get_default_cost_plot_args()
    plot_args.output_dir = output_dir
    plot_args.runs_dir = common_args.runs_dir
    plot_args.exp_names = selected_exp_names
    plot_args.exp_ids = exp_ids
    plot_args.display_names = selected_display_names
    plot_args.output_name = "cost_vs"
    plot_args.mode = "acc"
    plot_args.use_wandb = False

    submit_job(executor, cost_vs_plot, plot_args)


if __name__ == '__main__':
    main()
