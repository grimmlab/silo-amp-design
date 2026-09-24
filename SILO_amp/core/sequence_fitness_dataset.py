
import copy, os, sys, ray, torch 
from ..model.transformer_architecture import SequenceTransformer
from ..sequence_design import SequenceDesign
import numpy as np
from ray.thirdparty_files import psutil
from tqdm import tqdm
from ..core.abstract import Config, Instance
from ..core import stochastic_beam_search as sbs
from typing import List, Tuple, Any, Optional
from ..core.incremental_sbs import IncrementalSBS
from ..config import SequenceConfig
from ..sequence_evaluator import SequenceEvaluator
from ..evaluation_metrics.utils import candidate_selection_for_SILO_training
os.environ["RAY_DEDUP_LOGS"] = "0"
from itertools import product
import random
import hashlib


@ray.remote
class JobPool:
    def __init__(self, problem_instances: List[Instance]):
        self.jobs = [(i, instance) for i, instance in enumerate(problem_instances)]
        self.job_results = []

    def get_jobs(self, n_items: int):
        if len(self.jobs) > 0:
            items = self.jobs[:n_items]
            self.jobs = self.jobs[n_items:]
            return items
        else:
            return None

    def push_results(self, results: List[Tuple[int, Any]]):
        self.job_results.extend(results)

    def fetch_results(self):
        results = self.job_results
        self.job_results = []
        return results


class SequenceFitnessDataset:
    def __init__(self, config: SequenceConfig,
                 seen_protein_smiles: dict = None, 
                 sampling_round: int = None, 
                ):
        self.config = config
        self.self_improvement_learning = config.self_improvement_learning
        self.devices_for_workers: List[str] = self.self_improvement_learning["devices_for_workers"]
        self.seen_protein_smiles = seen_protein_smiles
        self.sampling_round = sampling_round

    def generate_dataset(self, network_weights: dict, best_objective: Optional[float] = None, memory_aggressive: bool = False):
        """
        Parameters:
            network_weights: [dict] Network weights to use for generating data.
            memory_aggressive: [bool] If True, IncrementalSBS is performed "memory aggressive" meaning that
                intermediate states in the search tree are not stored after transitioning from them, only their
                policies.
        """
        batch_size_gpu, batch_size_cpu = (self.self_improvement_learning["batch_size_per_worker"],
                                          self.self_improvement_learning["batch_size_per_cpu_worker"])

        problem_instances = []
        vocabulary_residue_idcs = list(range(0, len(self.config.residue_vocabulary))) 

        for i in range(self.config.multiplier):
            problem_instance = SequenceDesign.design_sequences(config=self.config, initial_seed=[vocabulary_residue_idcs[i]])
            problem_instances.append(problem_instance)
        

        job_pool = JobPool.remote(copy.deepcopy(problem_instances))
        results = [None] * len(problem_instances)
        self.num_trajectories_to_keep =  self.config.self_improvement_learning['num_trajectories_to_keep']

        # Check if we should pin the workers to core
        cpu_cores = [None] * len(self.devices_for_workers)
        if self.self_improvement_learning["pin_workers_to_core"] and sys.platform == "linux":
            # Get available core IDs
            affinity = list(os.sched_getaffinity(0))
            cpu_cores = [affinity[i % len(cpu_cores)] for i in range(len(self.devices_for_workers))]

        # Kick off workers
        future_tasks = [async_sbs_worker.remote
                        (self.config, job_pool, network_weights, device, batch_size_gpu if device != "cpu" else batch_size_cpu, 
                                    cpu_cores[i], best_objective, memory_aggressive, seen_protein_smiles= self.seen_protein_smiles, sampling_round=self.sampling_round)
            for i, device in enumerate(self.devices_for_workers)] 

        with tqdm(total=len(problem_instances)) as progress_bar:
            while True:
                # Check if all workers are done. If so, break after this iteration
                do_break = len(ray.wait(future_tasks, num_returns=len(future_tasks), timeout=0.5)[1]) == 0
                fetched_results = ray.get(job_pool.fetch_results.remote()) 
                for (i, result) in fetched_results:
                    results[i] = result
                if len(fetched_results):
                    progress_bar.update(len(fetched_results))
                if do_break:
                    break

        ray.get(future_tasks)
        del job_pool
        del network_weights
        torch.cuda.empty_cache()

        results_as_dict = self.sequence_object_to_dict(results)
        final_candidates = candidate_selection_for_SILO_training(trajectories=results_as_dict, seen_protein_smiles=self.seen_protein_smiles, config=self.config)
            
        return final_candidates

    def sequence_object_to_dict(self, results):

        """
            Processes the results from wor search into a dict to save it to as a pickle. Each trajectory will be represented as a dict with the
            following keys and values
            "identifier": Unique identifier for the trajectory/sequence
            "action_seq": List[List[int]] Actions which need to be taken on each index to create the sequence
            "residues": Index-level representation of sequence string.
            "level_list": A list of level number corresponding to action sequence 
            "seq_string": [str] Corresponding sequence string as a list
            "objective": [float] Objective function evaluation 
            "peptide": String representation of the sequence
            "objective_dict": Dictionary containing all computed objective components (e.g., surrogate, alanine scan, etc.).
            "num_masked_sites": Number of mutation sites considered."
            "seed sequence": Original sequence before applying edits 

        """

                
        all_results = []
        for i in range(0, len(results)):
            for res in results[i]:
                all_results.append(res)

        instances_dict = dict() 

        for seq in all_results:
            peptide=(''.join(seq.seq_string)), 
            instances_dict[peptide] = dict(
            identifier= seq.identifier, 
            length=len(seq.residues),
            peptide=seq.seq_string, 
            action_seq=seq.history,
            seq_list=seq.seq_list,
            residues = seq.residues, 
            objective = seq.objective,
            apex_mean_score = seq.apex_mean_score, 
            objective_dict= seq.apex_dict,
            omegaAMP_prob = seq.omegAMP_prob)
        return instances_dict
    
    

@ray.remote(max_calls=1)
def async_sbs_worker(config: Config, job_pool: JobPool, network_weights: dict,
                     device: str, batch_size: int,
                     cpu_core: Optional[int] = None,
                     best_objective: Optional[float] = None,
                     memory_aggressive: bool = False,
                     seen_protein_smiles = None, 
                     sampling_round = None
                     ):
    def child_log_probability_fn(trajectories: List[SequenceDesign]) -> [np.array]:
        return SequenceDesign.log_probability_fn(config = config, trajectories=trajectories, network=network, device=device)
    
    def get_apex_category_scores(seq, keys):
        return np.array(
            [seq.apex_dict[k] for k in keys],
            dtype=float)

    def get_apex_category_mean(seq, keys):
        scores = get_apex_category_scores(seq, keys)
        return scores
    
    
    def batch_leaf_evaluation_fn(trajectories: List[SequenceDesign]) -> np.array:

        """
           Scoring generated peptides using APEX pathogen 

        """

        APEX_GRAM_NEGATIVE = ["A_baumannii", "E_coli_11775", "E_coli_AIC221", "E_coli_AIC222", "K_pneumoniae",
        "P_aeruginosa_PAO1", "P_aeruginosa_PA14"]
        APEX_GRAM_POSITIVE = ["S_aureus", "MRSA", "VRE_faecalis", "VRE_faecium"]

        objs = objective_evaluator.calculate_apex_scores(trajectories)
        amp_probs= objective_evaluator.calculate_omegAMP_probs(trajectories)

        for seq in trajectories:
            GN_scores = get_apex_category_mean(seq, APEX_GRAM_NEGATIVE)
            GP_scores = get_apex_category_mean(seq, APEX_GRAM_POSITIVE)
            seq.apex_dict["apex_gram_negative_mean"] = float(np.mean(GN_scores))
            seq.apex_dict["apex_gram_positive_mean"] = float(np.mean(GP_scores))

            seq.objective = 0.5 * seq.apex_dict["apex_gram_negative_mean"] + seq.apex_dict["apex_gram_positive_mean"]

            # Calculate selectivity: 
            # Metric taken from https://www.nature.com/articles/s41551-024-01201-x 
            gram_neg_median = np.median(GN_scores)
            gram_pos_median = np.median(GP_scores)

            seq.apex_dict["gram_negative_selectivity"] = (gram_neg_median / gram_pos_median)
            seq.apex_dict["gram_positive_selectivity"] = (gram_pos_median / gram_neg_median)

        return objs

    
    def child_transition_fn(trajectory_action_pairs: List[Tuple[SequenceDesign, int]]):
        return [traj.transition_fn(action) for traj, action in trajectory_action_pairs]
    
    def make_seed(base_seed, sampling_round, batch_idx):
        key = f"{base_seed}:{sampling_round}:{batch_idx}"

        return int.from_bytes(
            hashlib.sha256(key.encode("utf-8")).digest()[:4],
            byteorder="little",
        )
 
    # Pin worker to core if wanted
    if cpu_core is not None:
        os.sched_setaffinity(0, {cpu_core})
        psutil.Process().cpu_affinity([cpu_core])

    with torch.no_grad():
        if config.CUDA_VISIBLE_DEVICES:
            # override ray's limiting of GPUs
            os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES

        device = torch.device(device)
        network = SequenceTransformer(config, config.training_device)
        network.load_state_dict(network_weights)
        network.eval()
        network.to(network.device)
        
        objective_evaluator = SequenceEvaluator(config)

        while True:
            batch = ray.get(job_pool.get_jobs.remote(batch_size))

            if batch is None:
                break

            idx_list = [i for i, _ in batch]
            root_nodes = [instance for _, instance in batch]

            seed = make_seed(base_seed=config.seed, sampling_round=sampling_round, batch_idx=idx_list[0],)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            if config.self_improvement_learning["search_type"] == "beam_search":
                # Deterministic beam search.
                beam_leaves_batch: List[List[sbs.BeamLeaf]] = sbs.stochastic_beam_search(
                    child_log_probability_fn=child_log_probability_fn,
                    child_transition_fn=child_transition_fn,
                    root_states=root_nodes,
                    beam_width=config.self_improvement_learning["beam_width"],
                    deterministic=True
                )
            else:
                inc_sbs = IncrementalSBS(config, root_nodes, child_log_probability_fn, child_transition_fn,
                                         leaf_evaluation_fn=SequenceDesign.to_max_evaluation_fn,
                                         batch_leaf_evaluation_fn=batch_leaf_evaluation_fn,
                                         memory_aggressive=False)
                
                if config.self_improvement_learning["search_type"] == "wor":
                    beam_leaves_batch: List[List[sbs.BeamLeaf]] = inc_sbs.perform_incremental_sbs(
                        beam_width=config.self_improvement_learning["beam_width"],
                        num_rounds=config.self_improvement_learning["num_rounds"],
                        nucleus_top_p=config.self_improvement_learning["nucleus_top_p"],
                        sbs_keep_intermediate=config.self_improvement_learning["keep_intermediate_trajectories"],
                        best_objective=best_objective
                    )
                else:
                    raise ValueError(f"Unknown search_type: {config.self_improvement_learning['search_type']}. ""Expected 'wor' or 'beam_search'.") 

            results_to_push = []
            for j, result_idx in enumerate(idx_list):
                result: List[SequenceDesign] = [x.state for x in beam_leaves_batch[j]]
                # Check if they need objective evaluation (this will only be true for deterministic beam search)
                if result[0].objective is None:
                    batch_leaf_evaluation_fn(result)
                results_to_push.append((result_idx, result))
            ray.get(job_pool.push_results.remote(results_to_push)) 

            if device != "cpu":
                torch.cuda.empty_cache()

    del network
    del network_weights
    torch.cuda.empty_cache()



