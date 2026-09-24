import torch, os, time, pickle, random, mlflow, argparse
from .core import sequence_fitness_dataset
import numpy as np
from .model.transformer_architecture import SequenceTransformer
from .sequence_evaluator import SequenceEvaluator
from .sequence_dataset import PolicyTrainingDataset
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import autocast
from .config import SequenceConfig
import pandas as pd 
from .evaluation_metrics.utils import read_fasta_return_sequence_list, save_fasta
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def save_checkpoint(checkpoint: dict, filename: str, config: SequenceConfig):
    os.makedirs(config.results_path, exist_ok=True)
    path = os.path.join(config.results_path, filename)
    torch.save(checkpoint, path)
    


def save_all_peptides_pickle_file(config, sequences, path_to_pickle_file):


    """
        Takes stored pickle file after wor sampling as input. If the dataset already exists at the path where to save, we load it, merge them into the rest
        of the dataset.
    
    """

    destination_path = path_to_pickle_file

    gen_seqs = sorted(sequences, key=lambda x:  x["objective"], reverse=config.max_objective)
    merged_seqs = gen_seqs

    temp_d = {x["peptide"]: x for x in merged_seqs}

    if destination_path is not None:
        if os.path.isfile(destination_path):
            with open(destination_path, "rb") as f:
                existing_seqs = pickle.load(f)  # list of dicts

            temp_d = {x["peptide"]: x for x in existing_seqs + merged_seqs}
            merged_seqs = list(temp_d.values())
        else:
            merged_seqs = sorted(merged_seqs, key=lambda x: x["objective"], reverse=config.max_objective)
        
        # Pickle the generated data again
        with open(destination_path, "wb") as f:
            pickle.dump(merged_seqs, f)

def save_pickle_file_for_training(config, sequences, path_to_pickle_file):


    """
        Takes stored pickle file after wor sampling as input. If the dataset already exists at the path where to save, we load it, merge them and take the best from the
            merged dataset.

            Then returns the following dictionary:
            - "mean_gen_obj": Mean generated obj. -> over the unmerged best sequences generated
            - "best_gen_obj": Best generated obj. -> Best obj. of the unmerged sequences generated
            - "worst_gen_obj": Worst generated obj. -> Worst obj. of the unmerged sequences generated
            - "mean_top_20_obj": Mean top 20 obj. -> over the merged best sequences
            - "mean_kept_obj": Mean of num_trajectories_to_keep sequences 
            - "top_20_sequences": A list with obj. of the top 20 obj.
    
    """

    metrics_return = dict()
    destination_path = path_to_pickle_file

    gen_seqs = sorted(sequences, key=lambda x:  x["objective"], reverse=config.max_objective)
    generated_objs = np.array([x["apex_mean_score"]for x in gen_seqs])
    metrics_return["mean_gen_obj"] = generated_objs.mean()
    metrics_return["best_gen_obj"] = generated_objs[0]
    metrics_return["worst_gen_obj"] = generated_objs[-1]

    merged_seqs = sequences

    if destination_path is not None:
        if os.path.isfile(destination_path):
            with open(destination_path, "rb") as f:
                existing_seqs = pickle.load(f)  # list of dicts
            temp_d = {x["peptide"]: x for x in existing_seqs + merged_seqs}
            merged_seqs = list(temp_d.values())
            merged_seqs = sorted(merged_seqs, key=lambda x: x["objective"], reverse=config.max_objective)[:config.self_improvement_learning['num_trajectories_to_keep']]
        else:
            merged_seqs = sorted(merged_seqs, key=lambda x: x["objective"], reverse=config.max_objective)[
                                :config.self_improvement_learning['num_trajectories_to_keep']]
        # Pickle the generated data again
        with open(destination_path, "wb") as f:
            pickle.dump(merged_seqs, f)

    all_generated_seqs = sorted(merged_seqs, key=lambda x: x["objective"], reverse=config.max_objective)
    metrics_return["mean_top_20_obj"] = np.array([x["objective"] for x in all_generated_seqs[:20]]).mean()
    metrics_return["mean_kept_obj"] = np.array([x["objective"]for x in all_generated_seqs]).mean()
    metrics_return["top_20_sequences"] = [{x["identifier"]: x["objective"]for x in all_generated_seqs[:20]}]

    return metrics_return, merged_seqs 



def train_for_one_cycle(epoch: int, config: SequenceConfig, network: SequenceTransformer, network_weights: dict,
                        optimizer: torch.optim.Optimizer, objective_evaluator: SequenceEvaluator, best_objective: float,
                        seen_protein_smiles=None, logger=None):
    
    """
        Main training loop for updating the policy in each learning round.

        Overview:
            Each round consists of (1) sampling candidate sequences using the current policy and (2) updating the policy using high-quality trajectories selected via
            objective function feedback.
    
    """

    sequence_fitness_datasets = sequence_fitness_dataset.SequenceFitnessDataset(config=config, 
                                           seen_protein_smiles=seen_protein_smiles)
    

    candidates = sequence_fitness_datasets.generate_dataset(network_weights, best_objective=best_objective, memory_aggressive=False)

    # do basic and synthesis related checks 
    passed_masked = objective_evaluator.peptide_checks.basic_validity_mask(candidates, seen_protein_smiles)
    passed_sequences = [seq for seq, passed in zip(candidates, passed_masked) if passed]
    final_candidates = objective_evaluator.peptide_checks.synthesis_based_masking(passed_sequences)
    
    # Save sequences in csv file and pickle file 
    logger.save_results_csv(config= config, trajectories = final_candidates, path_csv_file=os.path.join(config.results_path, f"generated_peptides.csv"))    
    save_all_peptides_pickle_file(config, final_candidates, os.path.join(config.results_path, f"global_peptide.pickle"))    
    metrics, picked_candidates = save_pickle_file_for_training(config, final_candidates, os.path.join(config.results_path, f"peptides_for_training.pickle"))
    for cand in picked_candidates:
        seen_protein_smiles.append(cand['peptide'])

    
    if len(metrics) > 0:

        print("Generated Sequences")
        print(f"Mean obj. over fresh best seqs: {metrics['mean_gen_obj']:.3f}")
        print(f"Best / worst obj. over fresh best seqs: {metrics['best_gen_obj']:.3f}, {metrics['worst_gen_obj']:.3f}")

        torch.cuda.empty_cache()
        time.sleep(1)
        print("---- Loading dataset")

        dataset = PolicyTrainingDataset(config, os.path.join(config.results_path, f"peptides_for_training.pickle"), batch_size=config.batch_size_training,
                                    custom_num_batches=config.num_batches_per_epoch, no_random=False)

        dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=False, 
                                persistent_workers=False)
        
        scaler = torch.amp.GradScaler()
        criterion = CrossEntropyLoss(reduction="mean", ignore_index=-1)

        network.train()

        # freeze layers except the last
        for parameter in network.parameters():
            parameter.requires_grad = False

        for parameter in network.action_head.parameters():
            parameter.requires_grad = True

        # unfreeze last transformer block
        for parameter in network.encoders[-3:].parameters():
            parameter.requires_grad = True

        # Train for n epochs 

        accumulated_total_loss = 0
        num_batches = len(dataloader)
        progress_bar = tqdm(range(num_batches))
        data_iter = iter(dataloader)

        for _ in progress_bar:
            data = next(data_iter)
            input_data = {k: v[0].to(network.device) for k, v in data["input"].items()}
            
            # targets for the logits
            targets = data["targets"].squeeze(0).to(network.device)

            # Optimization step
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=config.training_device, dtype=torch.bfloat16): 
                
                logits = network(input_data)

                # We mask the output according to feasibility
                logits[input_data["feasibility_mask"]] = float("-inf")
                
                # loss is calculated only for the output logits
                loss = criterion(logits, targets)
                loss = torch.tensor(0.) if torch.isnan(loss) else loss 

            scaler.scale(loss).backward()
            if config.optimizer["gradient_clipping"] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=config.optimizer["gradient_clipping"])

            scaler.step(optimizer)

            # Updates the scale for next iteration.
            scaler.update()

            accumulated_total_loss += loss.item()

        print(f"FT cycle {epoch+1}/{config.num_epochs} | " f"average loss={accumulated_total_loss/ num_batches:.4f}")

        del data 

        metrics["total_loss"] = accumulated_total_loss /num_batches

        del metrics["top_20_sequences"]

        return metrics, final_candidates


def set_seed(seed=0, full_deterministic=True):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if full_deterministic:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
            torch.use_deterministic_algorithms(True, warn_only=False)
            # Enable CuDNN deterministic mode
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def inference(epoch: int, config: SequenceConfig, network_weights: dict, logger=None, 
        evalutor: SequenceEvaluator = None):
    
    """
        Run the inference pipeline for a trained model. 

    """

    seen_protein_smiles = read_fasta_return_sequence_list(config.antibacterial_fasta) 
    generated_peptides = []
    sampling_round = 0

    with tqdm(total=config.total_peptide_count, desc="Generating peptides", unit="seq") as pbar:
        while len(generated_peptides) < config.total_peptide_count:

            sequence_fitness_datasets = sequence_fitness_dataset.SequenceFitnessDataset(config=config, seen_protein_smiles=seen_protein_smiles, sampling_round= sampling_round)
            initial_candidates = sequence_fitness_datasets.generate_dataset(network_weights, memory_aggressive=False)

            # do basic and synthesis related checks 
            passed_masked = evalutor.peptide_checks.basic_validity_mask(initial_candidates, seen_protein_smiles)
            passed_sequences = [seq for seq, passed in zip(initial_candidates, passed_masked) if passed]    

            for cand in passed_sequences:
                seen_protein_smiles.append(cand['peptide'])

            remaining = (config.total_peptide_count - len(generated_peptides))
            candidates_to_add = passed_sequences[:remaining]

            # Append generated peptides in a global log
            for candidate in candidates_to_add:
                generated_peptides.append((candidate['identifier'], candidate['peptide'])) 

            sampling_round += 1

            pbar.update(len(candidates_to_add))

            if len(generated_peptides) >= config.total_peptide_count:
                break

    generated_peptides = generated_peptides[:config.total_peptide_count]

    # Save
    generated_50k_fasta_path = os.path.join(config.results_path, "generated_50k_peptides_library.fasta")
    save_fasta(generated_peptides, generated_50k_fasta_path)

    return generated_50k_fasta_path


class ArtifactError(RuntimeError):
    """Raised when an artifact cannot satisfy its declared contract."""

def _write_fasta(path: Path, rows: list[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(f">{row['id']}\n{row['sequence']}\n")

def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    if "id" in fields:
        fields.remove("id")
        fields.insert(0, "id")
    if "sequence" in fields:
        fields.remove("sequence")
        fields.insert(1 if "id" in fields else 0, "sequence")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def write_submission_artifacts(
    output_dir: str | Path,
    valid_50k: Iterable[Mapping[str, Any]],
    selected_top100: Iterable[Mapping[str, Any]],
    *,
    config: SequenceConfig,
    metadata: Mapping[str, Any],
    expected_50k = 50000,
    expected_top_k = 100,
) -> dict[str, Any]:
    """Write complete candidate artifacts or fail with a diagnostic report."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    population = [dict(row) for row in valid_50k]
    selected = [dict(row) for row in selected_top100]
    reasons: list[str] = []
    if len(population) != expected_50k:
        reasons.append(f"50k_count:{len(population)} != {expected_50k}")
    if len(selected) != expected_top_k:
        reasons.append(f"top_k_count:{len(selected)} != {expected_top_k}")
    population_sequences = [row.get("sequence") for row in population]
    if any(not isinstance(sequence, str) for sequence in population_sequences):
        reasons.append("missing_population_sequence")
    if len(set(population_sequences)) != len(population_sequences):
        reasons.append("duplicate_population_sequence")
    if reasons:
        report = {"error": "artifact_shortfall", "reasons": reasons, "metadata": dict(metadata)}
        report_path = directory / "diagnostic_error.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise ArtifactError(f"Cannot write complete submission artifacts: {', '.join(reasons)}")

    population.sort(key=lambda row: (str(row["sequence"]), str(row.get("id", ""))))
    selected.sort(key=lambda row: (int(row.get("rank", 0)), str(row["sequence"]), str(row.get("id", ""))))
    files = {
        "population_fasta": directory / "library.fasta",
        "population_csv": directory / "library_50k.csv",
        "top100_fasta": directory / "top.fasta",
        "top100_csv": directory / "top_100.csv",
    }
    _write_fasta(files["population_fasta"], population)
    _write_csv(files["population_csv"], population)
    _write_fasta(files["top100_fasta"], selected)
    _write_csv(files["top100_csv"], selected)
    manifest = {
        "artifact_version": 1,
        "counts": {"library_50k": len(population), "top_100": len(selected)},
        "metadata": dict(metadata),
        "files": {name: {"path": path.name, "sha256": _sha256(path)} for name, path in files.items()},
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


class LegacyRuntimeError(RuntimeError):
    """Raised when a legacy SILO value cannot satisfy the integrated contract."""

def _max_hydrophobic_run(sequence: str) -> int:
    hydrophobic = frozenset("AVILMFWY")
    longest = run = 0
    for residue in sequence:
        if residue in hydrophobic:
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    return longest


def _as_bool(value: Any) -> bool:
    """Treat missing/NaN metric values as failed checks, never truthy values."""
    try:
        return bool(value) and value == value
    except (TypeError, ValueError):
        return False

def candidate_records_from_metrics(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalize real ``calculate_metrics_big_library`` rows for selection.

    The original table calls the primary score ``apex_mean_mic`` and the
    MarLys predicate ``passes_marlys_80``.  The generated selector intentionally
    uses neutral names, so conversion lives here rather than duplicating a
    selection implementation in the legacy runtime.
    """
    records: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        sequence = row.get("sequence")
        identifier = row.get("id")
        if not isinstance(sequence, str) or not isinstance(identifier, str):
            raise LegacyRuntimeError("metrics rows require string id and sequence fields")
        if row.get("apex_mean_mic") is None:
            raise LegacyRuntimeError(f"metrics row {identifier!r} has no apex_mean_mic")
        records.append({
            **row,
            "id": identifier,
            "sequence": sequence,
            "mean_predicted_mic": row["apex_mean_mic"],
            "charge": row.get("charge"),
            "hydrophobicity": row.get("hydrophobicity"),
            "cysteine_count": sequence.count("C"),
            "proline_per": sequence.count("P") / len(sequence),
            "max_hydrophobic_run": _max_hydrophobic_run(sequence),
            "marlys_identity_pass": _as_bool(row.get("passes_marlys_80")),
            "amphipathicity": row.get("hydrophobic_moment"),
        })
    return records