from typing import List, Union
import numpy as np
from .config import SequenceConfig
import os, ray, torch
from .sequence_design import SequenceDesign
from .evaluation_metrics.utils import APEXEnsemble, OmegAMPScorer, read_fasta_sequences, check_amp_synthesizability
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence
from Bio import Align
from .evaluation_metrics.metrics_utils import local_similarity, novelty_against_reference
STANDARD_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")
from pathlib import Path
PACKAGE_DIR = Path(__file__).resolve().parent

@ray.remote
class PredictorWorker:
    def __init__(self, config: SequenceConfig, device: torch.device):

        if config.CUDA_VISIBLE_DEVICES:
            # override ray's limiting of GPUs
            os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES
        self.device = device
        self.config = config

class SequenceEvaluator:
    def __init__(self, config: SequenceConfig, device: torch.device = None):
        self.config = config
        self.device = torch.device("cpu") if device is None else device
        self.predictor_workers = [PredictorWorker.remote(self.config, self.device) for _ in range(self.config.num_predictor_workers)] 
        self.apex_ensemble = APEXEnsemble(self.config, self.device)
        self.OmegAMPScorer = OmegAMPScorer('./SILO_amp/OmegAMP')
        self.peptide_checks = PeptideChecks(self.config)

    def calculate_apex_scores(self, sequences:List[Union[SequenceDesign, str]]):
        
        mic_scores = self.apex_ensemble.calculate_mic_scores(sequences)

        for i, (seq, scores) in enumerate(zip(sequences, mic_scores)):
            seq.apex_dict["A_baumannii"] = scores[0]
            seq.apex_dict["E_coli_11775"] = scores[1]
            seq.apex_dict["E_coli_AIC221"] = scores[2]
            seq.apex_dict["E_coli_AIC222"] = scores[3]
            seq.apex_dict["K_pneumoniae"] = scores[4]
            seq.apex_dict["P_aeruginosa_PAO1"] = scores[5]
            seq.apex_dict["P_aeruginosa_PA14"] = scores[6]
            seq.apex_dict["S_aureus"] = scores[7]
            seq.apex_dict["MRSA"] = scores[8]
            seq.apex_dict["VRE_faecalis"] = scores[9]
            seq.apex_dict["VRE_faecium"] = scores[10]
            seq.apex_mean_score = float(np.mean(scores))
            seq.apex_dict["apex_mic50"] = float(np.median(scores))
            seq.apex_dict["apex_mic90"] = float(np.quantile(scores, 0.90))

        return np.mean(mic_scores, axis=1)
    
    def calculate_omegAMP_probs(self, sequences:List[Union[SequenceDesign, str]]):
        if not isinstance(sequences[0], str): 
            seq_list = [seq.seq_string for seq in sequences]
        else:
            seq_list = sequences
        seq_list = np.array(seq_list)
        prob_scores = self.OmegAMPScorer.predict(seq_list)
        for i, (seq, scores) in enumerate(zip(sequences, prob_scores['omegamp_amp_prob'])):
            seq.omegAMP_prob = scores
        return prob_scores['omegamp_amp_prob']
    
class PeptideChecks:
    def __init__(self, config):
        STANDARD_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")
        self.config = config
        self.reference_fasta_path = config.antibacterial_fasta
        self.reference_sequences = read_fasta_sequences(self.reference_fasta_path)
        self.marlys_reference = read_fasta_sequences(config.marlys_fasta)
        self.training_data = read_fasta_sequences(config.training_fasta)
        self.STANDARD_AMINO_ACIDS = STANDARD_ALPHABET
           
    def basic_validity_mask(self, 
                            candidates,
                            seen):
    
        mask: list[bool] = []
        min_length = self.config.min_max_seq_length[0]
        max_length = self.config.min_max_seq_length[1]
        for seq in candidates:
            valid = (
                min_length <= len(seq['peptide']) <= max_length
                and set(seq['peptide']).issubset(self.STANDARD_AMINO_ACIDS)
                and seq['peptide'] not in seen
                and seq['peptide'] not in self.reference_sequences
                and seq['peptide'] not in self.marlys_reference
                and seq['peptide'] not in self.training_data)
            
            mask.append(valid)

        return mask
    
@dataclass(frozen=True)
class SelectionPolicy:
    min_length: int = 8
    max_length: int = 50
    mic_threshold: float = 64.0
    marlys_identity_limit: float = 0.80
    charge_range: tuple[float, float] = (2.0, 10.0)
    hydrophobicity_range: tuple[float, float] = (-0.5, 0.8)
    hydrophobic_moment_cutoff: tuple[float, float]  = (0.3, 0.6)
    max_cysteines: int = 1
    max_hydrophobic_run: int = 3
    diversity_similarity_limit: float = 0.40


@dataclass
class SelectionResult:
    selected: list[dict[str, Any]]
    valid_50k: list[dict[str, Any]]
    rejection_counts: dict[str, int] = field(default_factory=dict)
    rejection_reasons: dict[str, list[str]] = field(default_factory=dict)

def _inc(result: SelectionResult, candidate_id: str, reason: str) -> None:
    result.rejection_counts[reason] = result.rejection_counts.get(reason, 0) + 1
    result.rejection_reasons.setdefault(candidate_id, []).append(reason)

def _valid_top(candidate: Mapping[str, Any], policy: SelectionPolicy) -> str | None:
    charge = candidate.get("charge")
    hydrophobicity = candidate.get("hydrophobicity")
    hydrophobic_moment = candidate.get("amphipathicity")
    if charge is None or not policy.charge_range[0] <= float(charge) <= policy.charge_range[1]:
        return "charge"
    if hydrophobicity is None or not policy.hydrophobicity_range[0] <= float(hydrophobicity) <= policy.hydrophobicity_range[1]:
        return "hydrophobicity"
    if hydrophobic_moment is None or not policy.hydrophobic_moment_cutoff[0]  <= float(hydrophobic_moment) <= policy.hydrophobic_moment_cutoff[1]:
        return "amphipathicity"
    if int(candidate.get("cysteine_count")) > policy.max_cysteines + 1:
        return "cysteine_count"
    if int(candidate.get("proline_per")) > 0.20:
        return "proline content more than 20% of residues"
    if int(candidate.get("max_hydrophobic_run", policy.max_hydrophobic_run + 1)) > policy.max_hydrophobic_run:
        return "hydrophobic_run"
    if "GGG" in candidate.get("sequence"):
        return "more than 2 consecutive glycines"
    if not bool(candidate.get("marlys_identity_pass", False)):
        return "marlys identity over 80%: reject a candidate if an actual local alignment has pident > 80% and covers at least 80% of that generated candidate"
    return None

def get_novelty(candidate, novelty_cache, 
                      training_ref_amps, antibacterial_ref_amps):
    """Run novelty check once per unique sequence."""
    sequence = candidate["sequence"]

    if sequence not in novelty_cache:
        novelty_cache[sequence] = novelty_against_reference(
            candidate,
            training_ref_amps,
            antibacterial_ref_amps,
        )

    return novelty_cache[sequence]


def passes_diversity(candidate, policy, selected):
    """
        Candidate must be sufficiently different from
        every previously selected peptide.

        Reject if local similarity is strictly greater
        than the configured diversity limit.
    """
    sequence = candidate["sequence"]

    for chosen in selected:
        similarity = local_similarity(
            sequence,
            chosen,
        )
        if similarity > policy.diversity_similarity_limit:
            return False

    return True


def try_add_candidate(candidate, selected_sequences, result, novelty_cache, 
                      training_ref_amps, antibacterial_ref_amps, policy):
    """
    Add candidate to final selection only if it:
      - has not already been selected
      - passes diversity
      - passes both reference novelty checks
    """

    sequence = candidate["sequence"]
    candidate_id = str(
        candidate.get("id", "<missing-id>")
    )

    # 1. Duplicate check
    if sequence in selected_sequences:
        return False, {}

    # 2. Diversity check: reject if too similar to a previously selected peptide
    if not passes_diversity(candidate, policy, selected_sequences):
        _inc(result, candidate_id, "diversity")
        return False, {}

    # 3. Reference novelty check: Check novelty against training + antibacterial references
    novelty = get_novelty(candidate, novelty_cache, 
                      training_ref_amps, antibacterial_ref_amps)

    if not (
        novelty["passes_local_similarity_check"]
        and novelty["passes_ab_novelty"]
    ):
        _inc(result, candidate_id, "reference_novelty")
        return False, novelty

    return True, novelty


def select_candidates(
    candidates: Iterable[Mapping[str, Any]],
    *,
    references: Iterable[str] = (),
    marlys_references: Iterable[str] = (),
    training_amps: Iterable[str] = (),
    policy: SelectionPolicy = SelectionPolicy(),
    top_k: int = 100,
    activity_threshold: float = 64.0, 

) -> SelectionResult:
    """
    Validate a candidate population and greedily select a diverse top-K.

    Valid candidates are ranked by lower MIC, then sequence, then identifier.

    Duplicate sequences are rejected before scoring.

    Diversity is checked against all previously selected candidates.
    A candidate is rejected only when local similarity is strictly
    greater than the diversity threshold.
    
    """
    if top_k < 1:
        raise ValueError("top_k must be positive")
    reference_set = set([sequence for _, sequence in references])
    marlys_set = set(marlys_references)
    trainings_set = set([sequence for _, sequence in training_amps])

    result = SelectionResult(selected=[], valid_50k=[])
    seen: set[str] = set()

    # 1. Initial candidate validation and deduplication

    for raw in candidates:
        candidate = dict(raw)
        candidate_id = str(candidate.get("id", "<missing-id>"))
        sequence = candidate.get("sequence")
        basic_reason = _valid_basic(candidate, reference_set, marlys_set, trainings_set, policy)
        if basic_reason:
            _inc(result, candidate_id, basic_reason)
            continue
        if sequence in seen:
            _inc(result, candidate_id, "duplicate_sequence")
            continue
        seen.add(sequence)
        result.valid_50k.append(candidate)

    # 2. Apply synthesizability-based selection criteria 

    top_valid: list[dict[str, Any]] = []
    for candidate in result.valid_50k:
        candidate_id = str(candidate.get("id", "<missing-id>"))
        top_reason = _valid_top(candidate, policy)
        if top_reason:
            _inc(result, candidate_id, top_reason)
            continue
        top_valid.append(candidate)


    #  3. Construct category-specific candidate pools

    # Broad-spectrum pool:
    # Similar GP and GN MIC50 values and low overall MIC90.
    broad_pool = sorted([x for x in top_valid if (min(x["apex_GP_mic50"], x["apex_GN_mic50"]) / max(x["apex_GP_mic50"], x["apex_GN_mic50"])) >= 0.9  
                   and x["apex_mic90"] <= activity_threshold], key=lambda x: (
        # Most important: activity across many strains
        (x["apex_mic90"]), str(x["sequence"]),),)
    
    # Selection for GP pool
    # Prioritize lower GP MIC90, then selectivity.
    gp_pool = sorted(
        [
            x for x in top_valid
            if x["gram_positive_selectivity"] < 0.8
            and x["apex_GP_mic90"] <= activity_threshold
        ],
        key=lambda x: (
            x["apex_GP_mic90"],  # descending MIC90
            x["gram_positive_selectivity"], # tie-breaker: lower selectivity first
            str(x["sequence"]),
        ),
    )
    
    # Selection for GN pool
    # Prioritize lower GN MIC90, then selectivity.
    gn_pool = sorted(
        [
            x for x in top_valid
            if x["gram_negative_selectivity"] < 0.5
            and x["apex_GN_mic90"] <= activity_threshold
        ],
        key=lambda x: (
            x["apex_GN_mic90"],  # descending MIC90
            x["gram_negative_selectivity"], # tie-breaker: lower selectivity first
            str(x["sequence"]),
        ),
    )

    # MDR pool:
    # Prioritize lower MDR MIC90.
    mdr = sorted([x for x in top_valid if (x["apex_mdr_mic90"]) <= activity_threshold], 
                        key=lambda x: (x["apex_mdr_mic90"], str(x["sequence"]),),)

    # Overall activity pool:
    # Prioritize lower overall MIC90.
    overall_pool = sorted([x for x in top_valid if x["apex_mic90"] <= activity_threshold],
    key=lambda x: (
        x["apex_mic90"],
        str(x["sequence"]),
        str(x.get("id", "")),
    ))

    # Equal quotas by default.
    # top_k=100 -> exactly 25 each.
    base = top_k // 5
    quotas = {
        "overall": base + (top_k % 5),
        "GN_selective": base,
        "GP_selective": base,
        "broad_spectrum": base,
        "mdr": base

    }

    # GP-selective fallback
    # If the initial GP pool is too small, extend it with
    # additional candidates satisfying the apex_gram_positive_mean criterion.

    if len(gp_pool) < quotas["GP_selective"]:
        existing_gp_sequences = {str(x["sequence"]) for x in gp_pool}
        gp_fallback = sorted(
            [
                x for x in top_valid
                if str(x["sequence"]) not in existing_gp_sequences
                and x["gram_positive_selectivity"] < 0.8
                and x["apex_gram_positive_mean"] <= activity_threshold
            ],
            key=lambda x: (
                x["apex_gram_positive_mean"],              # fallback: lower apex_gram_positive_mean is better
                x["gram_positive_selectivity"],
                str(x["sequence"]),
            ),
        )

        gp_pool.extend(gp_fallback)

    # mdr fallback
    # If the initial mdr pool is too small, extend it with
    # additional candidates satisfying the mdr MIC50 criterion.
    

    if len(mdr) < quotas["mdr"]:
        existing_mdr_sequences = {str(x["sequence"]) for x in mdr}
        mdr_fallback = sorted(
            [
                x for x in top_valid
                if str(x["sequence"]) not in existing_mdr_sequences
                and x["apex_mdr_mean"] <= activity_threshold
            ],
            key=lambda x: (
                x["apex_mdr_mean"],              # fallback: lower MIC50 is better
                str(x["sequence"]),
            ),
        )

        mdr.extend(mdr_fallback)

    # Broad-spectrum fallback
    # Preserve the original broad_pool.
    # If it is too small, extend it with mean-mic-eligible broad-spectrum candidates.

    if len(broad_pool) < quotas["broad_spectrum"]:
        existing_broad_sequences = {str(x["sequence"]) for x in broad_pool}

        bs_fallback = sorted([x for x in top_valid if (str(x["sequence"]) not in existing_broad_sequences) and 
                              (min(x["apex_GP_mic50"], x["apex_GN_mic50"]) / max(x["apex_GP_mic50"], x["apex_GN_mic50"])) >= 0.9  
                   and x["apex_mean_mic"] <= activity_threshold],   
                   key=lambda x: (
        # Most important: activity across many strains
        (x["apex_mean_mic"]), str(x["sequence"]),),)

        broad_pool.extend(bs_fallback)

    # 4. Construct the final category pools
    pools = {
    "overall": overall_pool,
    "mdr": mdr,
    "GN_selective": gn_pool,
    "GP_selective": gp_pool,
    "broad_spectrum": broad_pool,}

    # 5. Shared final-selection
    category_order = ["GP_selective", "GN_selective", "broad_spectrum", "overall", "mdr"]
    selected = []
    selected_sequences = set()
    novelty_cache = {}

    for category in category_order:
        count = 0
        for candidate in pools[category]:
            if count >= quotas[category]:
                break
            state, novelty = try_add_candidate(candidate, selected_sequences, result, novelty_cache, training_amps
                                                ,references, policy)
            if state:
                selected_candidate = dict(candidate)
                selected_candidate["selection_category"] = category
                selected_candidate.update(novelty)
                selected.append(selected_candidate)
                selected_sequences.add(candidate["sequence"])
                count += 1

    # backfill if a category could not be completely filled 
    if len(selected) < top_k:
        for candidate in overall_pool:
            if len(selected) >= top_k:
                break
            state, novelty = try_add_candidate(candidate, selected_sequences, result, novelty_cache, training_amps
                                                ,references, policy)
            if state:
                selected_candidate = dict(candidate)
                selected_candidate["selection_category"] = "overall"
                selected_candidate.update(novelty)
                selected.append(selected_candidate)
                selected_sequences.add(candidate["sequence"])

    result.selected = [dict(item, rank=rank) for rank, item in enumerate(selected, 1)]
    return result


def _valid_basic(candidate: Mapping[str, Any], references: set[str], marlys_set: set[str], training_set:set[str], policy: SelectionPolicy) -> str | None:
    sequence = candidate.get("sequence")
    if not isinstance(sequence, str):
        return "invalid_sequence"
    if not policy.min_length <= len(sequence) <= policy.max_length:
        return "length"
    if not set(sequence).issubset(STANDARD_ALPHABET):
        return "alphabet"
    if sequence in references:
        return "antibacterial.fasta_exact_match"
    if sequence in marlys_set:
        return "marlys_database_exact_match"
    if sequence in training_set:
        return "training_set_exact_match"
    return None





    








    
