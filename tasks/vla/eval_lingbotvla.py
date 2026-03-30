"""Standalone VLA evaluation: load distributed checkpoint(s) and run validation loss on a LeRobot eval set.

Usage (same launcher as training)::

    bash train.sh tasks/vla/eval_lingbotvla.py configs/vla/eval_example.yaml \\
      --train.load_checkpoint_path /path/to/checkpoints/global_step_5000

Multi-checkpoint sweep (same parent ``checkpoints/`` dir as training)::

    # YAML: eval_checkpoints_dir + eval_checkpoint_start/end/interval, or eval_checkpoint_steps

Override eval data or steps from CLI::

    ... --data.eval_path /path/to/lerobot_root --train.eval_steps 100
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

import datetime
import torch
import torch.distributed as dist

from lingbotvla.checkpoint import build_checkpointer
from lingbotvla.data import VLADataCollatorWithPacking, build_dataloader
from lingbotvla.distributed.offloading import build_activation_offloading_context
from lingbotvla.distributed.parallel_state import init_parallel_state
from lingbotvla.distributed.torch_parallelize import build_parallelize_model
from lingbotvla.models import build_foundation_model, build_processor
from lingbotvla.utils import helper
from lingbotvla.utils.arguments import ModelArguments, normalize_lerobot_roots, parse_args, save_args
from copy import deepcopy

from lingbotvla.models.vla.vision_models.module_utils import build_depth_model

from train_lingbotvla import (
    MyDataArguments,
    MyTrainingArguments,
    _build_vla_val_dataset_pickled,
    _configure_hf_datasets_cache_env,
    _run_vla_validation,
    _val_dataset_pickle_path,
)

logger = helper.create_logger(__name__)

_STEP_DIR = re.compile(r"^global_step_(\d+)$")


@dataclass
class EvalTrainingArguments(MyTrainingArguments):
    """Extra fields for evaluating multiple ``global_step_*`` checkpoints in one run."""

    eval_checkpoints_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Directory containing global_step_* checkpoint subdirectories. "
                "If unset, inferred from train.load_checkpoint_path when that path ends with global_step_N."
            ),
        },
    )
    eval_checkpoint_steps: Optional[List[int]] = field(
        default=None,
        metadata={
            "help": (
                "Explicit global steps to evaluate, e.g. [500, 1000, 1500]. "
                "If set, eval_checkpoint_start/end/interval are ignored."
            ),
        },
    )
    eval_checkpoint_start: int = field(
        default=0,
        metadata={"help": "With eval_checkpoint_interval: first global step (inclusive)."},
    )
    eval_checkpoint_end: int = field(
        default=0,
        metadata={"help": "With eval_checkpoint_interval: last global step (inclusive)."},
    )
    eval_checkpoint_interval: int = field(
        default=0,
        metadata={
            "help": "If > 0 and eval_checkpoint_steps is empty, steps are start, start+interval, ... up to end.",
        },
    )


def _infer_eval_checkpoints_dir(train: EvalTrainingArguments) -> str:
    if train.eval_checkpoints_dir:
        return os.path.abspath(os.path.expanduser(train.eval_checkpoints_dir.strip()))
    lp = train.load_checkpoint_path
    if not lp:
        raise ValueError(
            "Set train.eval_checkpoints_dir, or train.load_checkpoint_path pointing to .../global_step_N "
            "so the parent checkpoints directory can be inferred."
        )
    lp = os.path.abspath(os.path.expanduser(lp.strip()))
    base = os.path.basename(os.path.normpath(lp))
    if _STEP_DIR.match(base):
        return os.path.dirname(lp)
    raise ValueError(
        "Multi-checkpoint eval needs train.eval_checkpoints_dir, or train.load_checkpoint_path like "
        ".../checkpoints/global_step_5000 so the parent dir can be inferred."
    )


def resolve_eval_checkpoint_jobs(train: EvalTrainingArguments) -> List[Tuple[int, str]]:
    """Return sorted ``(global_step, checkpoint_dir)`` list. Single checkpoint if no sweep config."""
    # Explicit list
    if train.eval_checkpoint_steps:
        steps = sorted({int(s) for s in train.eval_checkpoint_steps})
        root = _infer_eval_checkpoints_dir(train)
        return [(s, os.path.join(root, f"global_step_{s}")) for s in steps]

    # Range expansion
    if train.eval_checkpoint_interval > 0 and train.eval_checkpoint_end >= train.eval_checkpoint_start:
        s0, s1, iv = train.eval_checkpoint_start, train.eval_checkpoint_end, train.eval_checkpoint_interval
        steps = list(range(s0, s1 + 1, iv))
        root = _infer_eval_checkpoints_dir(train)
        return [(s, os.path.join(root, f"global_step_{s}")) for s in steps]

    # Single checkpoint (legacy)
    if not train.load_checkpoint_path:
        raise ValueError(
            "Specify one of: train.load_checkpoint_path (single eval), "
            "or train.eval_checkpoint_steps / eval_checkpoint_start+end+interval (sweep)."
        )
    lp = os.path.abspath(os.path.expanduser(train.load_checkpoint_path.strip()))
    base = os.path.basename(os.path.normpath(lp))
    m = _STEP_DIR.match(base)
    step = int(m.group(1)) if m else 0
    return [(step, lp)]


@dataclass
class EvalDataArguments(MyDataArguments):
    """Same as training data config, with optional ``eval_path`` overriding ``val_path`` for eval-only runs."""

    train_path: List[str] = field(
        default_factory=list,
        metadata={
            "help": "Not used when only evaluating; if empty, set from eval_path or val_path.",
        },
    )
    eval_path: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "LeRobot dataset root(s) for evaluation; when set, overrides val_path.",
        },
    )
    eval_episode_sample_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "If set, after resolving val episodes (val_episode_subset / episode_subset), randomly sample "
                "this many episode indices without replacement. Omit to use the full resolved list."
            ),
        },
    )
    eval_episode_sample_seed: int = field(
        default=42,
        metadata={"help": "RNG seed for eval_episode_sample_size (deterministic across runs)."},
    )

    def __post_init__(self):
        if self.eval_path is not None:
            self.eval_path = normalize_lerobot_roots(self.eval_path)
            self.val_path = self.eval_path
        if not self.train_path:
            if self.val_path:
                self.train_path = list(self.val_path)
            else:
                raise ValueError(
                    "Specify data.eval_path or data.val_path (or non-empty data.train_path for schema compatibility)."
                )
        super().__post_init__()
        if not self.val_path:
            raise ValueError("Evaluation requires non-empty data.val_path or data.eval_path.")


@dataclass
class EvalArguments:
    model: ModelArguments = field(default_factory=ModelArguments)
    data: EvalDataArguments = field(default_factory=EvalDataArguments)
    train: EvalTrainingArguments = field(default_factory=EvalTrainingArguments)


def main():
    _configure_hf_datasets_cache_env()
    args = parse_args(EvalArguments)
    logger.info(f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}")
    logger.info_rank0(json.dumps(asdict(args), indent=2))

    jobs = resolve_eval_checkpoint_jobs(args.train)
    if args.train.eval_steps <= 0:
        raise ValueError("train.eval_steps must be positive.")
    if args.train.eval_use_ema and not args.train.use_ema:
        raise ValueError("train.eval_use_ema=true requires train.use_ema=true.")

    torch.cuda.set_device(f"cuda:{args.train.local_rank}")
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(minutes=100),
        device_id=torch.device(f"cuda:{args.train.local_rank}"),
    )
    helper.set_seed(args.train.seed, args.train.enable_full_determinism)
    if args.train.local_rank == 0:
        helper.enable_third_party_logging()

    if args.train.global_rank == 0:
        os.makedirs(args.train.output_dir, exist_ok=True)
        save_args(args, args.train.output_dir)
    logger.info_rank0(f"Eval checkpoint job list ({len(jobs)}): {jobs}")

    Checkpointer = build_checkpointer(dist_backend=args.train.data_parallel_mode, ckpt_manager=args.train.ckpt_manager)

    init_parallel_state(
        dp_size=args.train.data_parallel_size,
        dp_replicate_size=args.train.data_parallel_replicate_size,
        dp_shard_size=args.train.data_parallel_shard_size,
        tp_size=args.train.tensor_parallel_size,
        ep_size=args.train.expert_parallel_size,
        pp_size=args.train.pipeline_parallel_size,
        cp_size=args.train.context_parallel_size,
        ulysses_size=args.train.ulysses_parallel_size,
        dp_mode=args.train.data_parallel_mode,
    )

    logger.info_rank0("Building model (base weights from model.model_path; checkpoint applied after FSDP).")
    config_kwargs = {
        "vlm_repo_id": getattr(args.model, "vlm_repo_id", None),
        "action_dim": getattr(args.train, "action_dim", 7),
        "max_action_dim": getattr(args.train, "max_action_dim", 32),
        "max_state_dim": getattr(args.train, "max_state_dim", 32),
        "chunk_size": getattr(args.train, "chunk_size", 50),
        "tokenizer_path": getattr(args.model, "tokenizer_path", None),
        "post_training": getattr(args.model, "post_training", False),
        "incremental_training": getattr(args.model, "incremental_training", False),
        "depth_incremental_training": getattr(args.model, "depth_incremental_training", False),
        "norm_qkv": getattr(args.train, "norm_qkv", False),
        "enable_expert_vision": args.train.enable_expert_vision,
        "expert_vision_type": getattr(args.train, "expert_vision_type", None),
        "expert_vision_path": getattr(args.train, "expert_vision_path", None),
        "adanorm_time": getattr(args.model, "adanorm_time", False),
        "split_gate_liner": getattr(args.model, "split_gate_liner", False),
        "nosplit_gate_liner": getattr(args.model, "nosplit_gate_liner", False),
        "separate_time_proj": getattr(args.model, "separate_time_proj", False),
        "old_adanorm": getattr(args.model, "old_adanorm", False),
        "final_norm_adanorm": getattr(args.model, "final_norm_adanorm", False),
        "loss_type": getattr(args.train, "loss_type", "fm"),
        "align_params": getattr(args.train, "align_params", None),
    }
    if not getattr(args.model, "adanorm_time", False):
        assert not getattr(args.model, "separate_time_proj", False), (
            "separate_time_proj should be dropped when we do not apply adanorm_time!!"
        )
    if getattr(args.model, "old_adanorm", False):
        assert getattr(args.model, "adanorm_time", False), "Apply old_adanorm should apply adanorm_time!!"
    if args.train.enable_expert_vision and not args.model.post_training:
        assert args.train.expert_vision_path is not None, (
            "expert_vision_path is required when enable_expert_vision is True!!!"
        )

    model = build_foundation_model(
        config_path=args.model.config_path,
        weights_path=args.model.model_path,
        torch_dtype="float32" if args.train.enable_mixed_precision else "bfloat16",
        init_device=args.train.init_device,
        freeze_vision_encoder=args.train.freeze_vision_encoder,
        tokenizer_max_length=args.train.tokenizer_max_length,
        vocab_size=args.model.vocab_size,
        use_lm_head=args.model.use_lm_head,
        force_use_huggingface=args.model.force_use_huggingface,
        config_kwargs=config_kwargs,
    )

    use_depth_align = args.train.align_params != {}
    depth_model_type = None
    moge_model, morgbd_model = None, None
    if use_depth_align:
        assert args.model.moge_path is not None and args.model.morgbd_path is not None, (
            "Depth models must be set when align_params is non-empty (LingBot-VLA-Depth)."
        )
        args.train.align_params["visual_dir"] = os.path.join(args.train.output_dir, "images")
        args.train.align_params["depth"]["moge_path"] = args.model.moge_path
        args.train.align_params["depth"]["morgbd_path"] = args.model.morgbd_path
        depth_model_type = args.train.align_params["depth"]["model_type"]
        moge_model, morgbd_model = build_depth_model(args.train.align_params)
        if args.train.use_compile:
            moge_model = torch.compile(moge_model)
            morgbd_model = torch.compile(morgbd_model)
        os.makedirs(args.train.align_params["visual_dir"], exist_ok=True)

    helper.print_device_mem_info("VRAM usage after building model")

    processor = build_processor(args.model.tokenizer_path)

    if args.data.datasets_type != "vla":
        raise ValueError("eval_lingbotvla.py only supports data.datasets_type=vla.")
    if args.data.dataloader_type != "native":
        raise ValueError("eval_lingbotvla.py only supports data.dataloader_type=native.")
    if args.train.rmpad:
        raise ValueError("Qwen2-VL does not support rmpad. Use `rmpad_with_pos_ids` instead.")

    data_collate_fn = [VLADataCollatorWithPacking()]

    args.data.chunk_size = args.train.chunk_size
    mfirst = args.data.dataset_init_main_process_first and args.train.world_size > 1
    val_cache_path = _val_dataset_pickle_path(args.data.dataset_pickle_cache_path)
    val_cache_path = val_cache_path.replace(".pkl", "_eval.pkl")

    val_dataset = _build_vla_val_dataset_pickled(
        args, model, processor, use_depth_align, mfirst, val_cache_path
    )
    val_dataloader = build_dataloader(
        dataset=val_dataset,
        micro_batch_size=args.train.micro_batch_size,
        global_batch_size=args.train.global_batch_size,
        dataloader_batch_size=args.train.dataloader_batch_size,
        seed=args.train.seed,
        collate_fn=data_collate_fn,
        max_seq_len=args.data.max_seq_len,
        train_steps=args.train.eval_steps,
        rmpad=args.train.rmpad,
        rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
        bsz_warmup_ratio=args.train.bsz_warmup_ratio,
        dyn_bsz_margin=args.train.dyn_bsz_margin,
        dyn_bsz_buffer_size=args.train.dyn_bsz_buffer_size,
        num_workers=args.data.num_workers,
        drop_last=False,
        pin_memory=args.data.pin_memory,
        prefetch_factor=args.data.prefetch_factor if args.data.num_workers > 0 else None,
        shuffle=False,
    )

    fsdp_kwargs = {}
    if args.train.freeze_vit:
        model.visual.requires_grad_(False)
        if args.train.data_parallel_mode == "fsdp1":
            fsdp_kwargs["use_orig_params"] = True

    model_ema = deepcopy(model).eval() if args.train.use_ema else None

    model = build_parallelize_model(
        model,
        enable_full_shard=args.train.enable_full_shard,
        enable_mixed_precision=args.train.enable_mixed_precision,
        enable_fp32=args.train.enable_fp32,
        enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
        init_device=args.train.init_device,
        enable_fsdp_offload=args.train.enable_fsdp_offload,
        fsdp_kwargs=fsdp_kwargs,
        basic_modules=model._no_split_modules if args.train.module_fsdp_enable else None,
        enable_reentrant=args.train.enable_reentrant,
        enable_forward_prefetch=args.train.enable_forward_prefetch,
        fsdp_llm_blocks=False,
        ignore_norm=False,
        use_depth_align=use_depth_align,
        ignore_depth=args.train.ignore_depth,
    )
    if model_ema is not None:
        model_ema = build_parallelize_model(
            model_ema,
            enable_full_shard=args.train.enable_full_shard,
            enable_mixed_precision=args.train.enable_mixed_precision,
            enable_fp32=args.train.enable_fp32,
            enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
            init_device=args.train.init_device,
            enable_fsdp_offload=args.train.enable_fsdp_offload,
            fsdp_kwargs=fsdp_kwargs,
            basic_modules=model_ema._no_split_modules if args.train.module_fsdp_enable else None,
            enable_reentrant=args.train.enable_reentrant,
            enable_forward_prefetch=args.train.enable_forward_prefetch,
            fsdp_llm_blocks=False,
            ignore_norm=False,
            use_depth_align=use_depth_align,
            ignore_depth=args.train.ignore_depth,
        )

    if args.train.use_compile:
        model = torch.compile(model)
        if model_ema is not None:
            model_ema = torch.compile(model_ema)

    use_ema_for_eval = args.train.eval_use_ema and model_ema is not None
    eval_net = model_ema if use_ema_for_eval else model

    if use_ema_for_eval:
        state = {"model": model, "ema": model_ema}
    else:
        state = {"model": model}

    model_fwd_context, _ = build_activation_offloading_context(
        args.train.enable_activation_offload,
        args.train.enable_gradient_checkpointing,
        args.train.activation_gpu_limit,
    )

    src = "ema" if use_ema_for_eval else "train"
    results: List[dict] = []

    for step_i, ckpt in jobs:
        if not os.path.isdir(ckpt):
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt} (global_step={step_i})")

        logger.info_rank0(f"Loading checkpoint global_step={step_i} from {ckpt}")
        Checkpointer.load(ckpt, state)
        dist.barrier()

        val_iter = [None]

        val_loss, val_vla, val_depth = _run_vla_validation(
            eval_net,
            val_dataloader,
            args.train.eval_steps,
            args,
            use_depth_align,
            depth_model_type,
            moge_model,
            morgbd_model,
            model_fwd_context,
            val_iter,
        )

        logger.info_rank0(
            f"Eval ({src}) global_step={step_i}: loss={val_loss:.4f}, vla_loss={val_vla:.4f}, depth_loss={val_depth:.4f}"
        )
        results.append(
            {
                "global_step": step_i,
                "checkpoint": ckpt,
                "loss": val_loss,
                "vla_loss": val_vla,
                "depth_loss": val_depth,
            }
        )

    if args.train.global_rank == 0:
        summary = {
            "eval_dataset": list(args.data.val_path),
            "eval_steps": args.train.eval_steps,
            "source": src,
            "checkpoints": results,
        }
        out_path = os.path.join(args.train.output_dir, "eval_metrics.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info_rank0(f"Wrote metrics to {out_path}")

        lines = [
            "Validation loss (summary)",
            f"  source={src}  eval_steps={args.train.eval_steps}",
            "  global_step | loss | vla_loss | depth_loss",
        ]
        for row in results:
            lines.append(
                f"  {row['global_step']:>11d} | {row['loss']:.6f} | {row['vla_loss']:.6f} | {row['depth_loss']:.6f}"
            )
        table = "\n".join(lines)
        print(table, flush=True)
        logger.info_rank0("\n" + table)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
