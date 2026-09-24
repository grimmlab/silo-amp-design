"""Run the existing SILO policy as deterministic, inference-only generation.

This is deliberately a root-level entry point: it loads the project's real
PyTorch checkpoint, builds the existing ``SequenceTransformer``/APEX/Ray
runtime, and then uses the generated package's selection and artifact code.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any
from .config import SequenceConfig
from .sequence_evaluator import SequenceEvaluator, SelectionPolicy, select_candidates
from .utils import inference, set_seed, write_submission_artifacts, candidate_records_from_metrics
from .evaluation_metrics.utils import read_fasta_return_sequence_list, BigLibraryMetrics
from .model.transformer_architecture import SequenceTransformer
import pandas as pd
import ray, torch, os, argparse, copy
from pathlib import Path
import logging
project_root = Path(__file__).resolve().parent.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SILO for AMP design reproducible inference")
    parser.add_argument("--checkpoint", type=Path, default='./inference_model')
    parser.add_argument("--output_dir", type=Path, default='./generate')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--total-peptide-count", type=int, default=50000)
    parser.add_argument("--top-k", type=int, default=100)
    return parser


def run_inference(args: argparse.Namespace) -> dict[str, Any]:

    """Generate, select, and write candiates using the SILO."""

    checkpoint_path = project_root.parent / "generate"
    output_dir = args.output_dir
    output_path = Path(args.output_dir)
    if not output_path.is_absolute():
        output_path = (project_root / output_path).resolve()

    output_dir = str(output_path)
    os.makedirs(output_dir, exist_ok=True)

    if not checkpoint_path:
        raise ValueError(f"checkpoint not found: {checkpoint_path}")
    if args.total_peptide_count < 1 or args.top_k < 1:
        raise ValueError("total-peptide-count and top-k must be positive")
    if args.top_k > args.total_peptide_count:
        raise ValueError("top-k cannot exceed total-peptide-count")

    config = SequenceConfig(args)
    config.results_path = str(output_dir)
    config.total_peptide_count = args.total_peptide_count
    config.top_k_peptides = args.top_k
    config.training_device = args.device
    config.do_inference = True
    config.self_improvement_learning["devices_for_workers"] = [args.device]
    config.self_improvement_learning["beam_width"] = 32

    network = SequenceTransformer(config, config.training_device)
    set_seed(args.seed)
    checkpoint = torch.load(os.path.join(checkpoint_path, "best_model.pt"), weights_only=False)
    network.load_state_dict(checkpoint["model_weights"])
    print(f"Loading checkpoint from path {checkpoint_path} for inference")

    optimizer = torch.optim.Adam(network.parameters(), lr=config.optimizer["lr"], weight_decay=config.optimizer["weight_decay"])
    optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer_state"])) 

    output_rel = output_path.relative_to(project_root).as_posix()
    runtime_env={
        "working_dir": str(project_root),
        "excludes": [
            ".git/**",
            f"{output_rel}/**",
            ".venv/**",
            "SILO_amp/.venv/**",
            "SILO_amp/OmegAMP/data/generative-model-data/**",
            "SILO_amp/OmegAMP/data/activity-data/**",
        ],}
    
    logging.getLogger("ray._private.runtime_env.packaging").setLevel(logging.ERROR)
    ray.init(runtime_env=runtime_env)
    
    print(f"Policy network is on device {config.training_device}")
    network.to(network.device)
    network.eval()
    evaluator = SequenceEvaluator(config, torch.device(args.device))
    big_library_worker = BigLibraryMetrics(config, config.training_device, evaluator)

    network_weights = copy.deepcopy(network.get_weights())

    print("---Running SILO under reproducible sampling conditions to generate 50K library and top 100 candidate list ---")

    generated_50k_fasta_path = inference(
        epoch="submission",
        config=config,
        network_weights=network_weights,
        evalutor=evaluator)

    generated_50k_df = big_library_worker.calculate_metrics_big_library(config, generated_50k_fasta_path)
    records = candidate_records_from_metrics(generated_50k_df.to_dict("records"))
    references = read_fasta_return_sequence_list(config.antibacterial_fasta)
    marlys_set = [sequence for _, sequence in read_fasta_return_sequence_list(config.marlys_fasta)]
    training_set = read_fasta_return_sequence_list(config.training_fasta)
    selection = select_candidates(records, references=references, training_amps=training_set, marlys_references=marlys_set, policy=SelectionPolicy(), top_k=args.top_k)
    return write_submission_artifacts(
        output_dir,
        selection.valid_50k,
        selection.selected,
        metadata={
            "mode": "root-legacy-runtime",
            "seed": args.seed,
            "selection_rejection_counts": selection.rejection_counts,
        },
        expected_50k=args.total_peptide_count,
        expected_top_k=args.top_k,
        config=config
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.epoches = 1
    args.comments=None
    run_inference(args)
    ray.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
