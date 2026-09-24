from pathlib import Path
from Bio import Align
from ..config import SequenceConfig
import torch, os, sys, math, glob, json, tempfile
import pandas as pd 
import numpy as np
from ..apex import APEX_models
from ..apex.utils import onehot_encoding, make_vocab
sys.modules["APEX_models"] = APEX_models
from ..evaluation_metrics.metrics_utils import mmseqs_marlys_similarity, calculate_physchem_prop, calculate_hydrophobicmoment, calculate_charge, calculate_hydrophobicity
from Bio import SeqIO

OMEGAMP_ROOT = Path(__file__).resolve().parents[1] / "OmegAMP"
if str(OMEGAMP_ROOT) not in sys.path:
    sys.path.insert(0, str(OMEGAMP_ROOT))

from project.classifiers import AMPClassifier


class OmegAMPScorer:
    MODEL_FILES = {
        "amp": "broad-classifier.json",
        "A_baumannii":
            "species-acinetobacterbaumannii-classifier.json",
        "E_coli":
            "species-escherichiacoli-classifier.json",
        "K_pneumoniae":
            "species-klebsiellapneumoniae-classifier.json",
        "P_aeruginosa":
            "species-pseudomonasaeruginosa-classifier.json",
        "S_aureus":            "species-staphylococcusaureus-classifier.json",
    }

    def __init__(self, omegamp_dir):
        model_dir = Path(omegamp_dir) / "models"

        self.models = {
            name: AMPClassifier(
                model_path=str(model_dir / filename)
            ).eval()
            for name, filename in self.MODEL_FILES.items()
        }

    def predict(self, sequences):

        sequences = list(sequences)
        features = AMPClassifier(model_path=None).get_input_features(sequences)

        scores = {}

        for name, model in self.models.items():
            scores[f"omegamp_{name}_prob"] = (
                model.predict_from_features(
                    features,
                    proba=True
                )
            )

        return scores
    
class APEXEnsemble:
    def __init__(self, config, device):
        # Load the 8 pretrained APEX-pathogen models (relative to this file, not the cwd).
        MODEL_DIR = os.path.join(config.apex_work_dir, "APEX_pathogen_models")
        self.APEX_models = []
        self.device = device
        self.max_len = 52 #maximum seq length; 52 = start character + maximum peptide length (50 aa) + end character; longer peptides will be truncated
        self.word2idx, idx2word = make_vocab()
        self.batch_size = 1000 
        self.STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")

        for a_model in sorted(glob.glob(os.path.join(MODEL_DIR, "APEX_*"))):
            model = torch.load(a_model, map_location=self.device, weights_only=False)
            model.eval()
            self.APEX_models.append(model)

    def calculate_mic_scores(self, sequences):
        if not isinstance(sequences[0], str): 
            seq_list = [seq.seq_string for seq in sequences]
        else:
            seq_list = sequences
        seq_list = np.array(seq_list)

        for ensemble_id in range(len(self.APEX_models)):
            if self.device != 'cpu':
                AMP_model = self.APEX_models[ensemble_id].cuda().eval()
            else:
                AMP_model = self.APEX_models[ensemble_id].cpu().eval()

            data_len = len(seq_list)
            
            for i in range(int(math.ceil(data_len/float(self.batch_size)))):

                seq_batch = seq_list[i*self.batch_size:(i+1)*self.batch_size]
                seq_rep = onehot_encoding(seq_batch, self.max_len, self.word2idx) #make input

                if self.device != 'cpu':
                    X_seq = torch.LongTensor(seq_rep).cuda()
                    AMP_pred_batch = AMP_model(X_seq).cpu().detach().numpy() #make predictions
                else:
                    X_seq = torch.LongTensor(seq_rep)
                    AMP_pred_batch = AMP_model(X_seq).detach().numpy() #make predictions

                AMP_pred_batch = 10**(6-AMP_pred_batch) #transform back to MICs; When training the APEX models, MICs were transformed by: -np.log10(MICs/float(1000000))

                if i == 0:
                    AMP_pred = AMP_pred_batch
                else:
                    AMP_pred = np.vstack([AMP_pred, AMP_pred_batch])

            #sum up the predictions made by different APEX models
            if ensemble_id == 0:
                AMP_sum = AMP_pred
            else:
                AMP_sum += AMP_pred


        AMP_pred = AMP_sum /len(self.APEX_models)

        return AMP_pred
        
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
    

class BigLibraryMetrics:
    def __init__(self, config: SequenceConfig, device: torch.device = None, evaluator = None):
        self.config = config
        self.device = torch.device("cpu") if device is None else device
        self.evaluator = evaluator
        self.peptide_checks = PeptideChecks(config)
    
    def calculate_metrics_big_library(self, config, path_to_generated_peptides):

        """
            Calculate MIC calculation, AMP probability from OmegAMP, and synthesizability related properties
        """

        print("------")
        print("Running evaluation metrics on generated peptide library. This may take a while :/")
        print("------")

        generated_peptides = read_fasta_return_sequence_list(path_to_generated_peptides)
        lengths = [len(seq[1]) for seq in generated_peptides]
        generated_peptides_with_lengths = []
        for length, seq in zip(lengths, generated_peptides):
            generated_peptides_with_lengths.append((seq[0], seq[1], length))

        assert len(generated_peptides_with_lengths) == config.total_peptide_count

        generated_df = pd.DataFrame(generated_peptides_with_lengths, columns=["id", "sequence", "length"])
        generated_peptides_list = [seq[1] for seq in generated_peptides]

        #1. MIC calculation
        apex_pathogen_scores = self.evaluator.apex_ensemble.calculate_mic_scores(generated_peptides_list)
        apex_mean_scores = np.mean(apex_pathogen_scores, axis=1)

        apex_df = pd.DataFrame({"id": generated_df["id"].values, "apex_mean_mic": apex_mean_scores, 
                                "A_baumannii": apex_pathogen_scores[:, 0], "E_coli_11775": apex_pathogen_scores[:, 1], 
                                "E_coli_AIC221": apex_pathogen_scores[:, 2], "E_coli_AIC222": apex_pathogen_scores[:, 3],
                                "K_pneumoniae": apex_pathogen_scores[:, 4], "P_aeruginosa_PAO1": apex_pathogen_scores[:, 5],
                                "P_aeruginosa_PA14": apex_pathogen_scores[:, 6], "S_aureus": apex_pathogen_scores[:, 7],
                                "MRSA": apex_pathogen_scores[:, 8], "VRE_faecalis": apex_pathogen_scores[:, 9], "VRE_faecium": apex_pathogen_scores[:, 10],
                                "apex_mic50": self.apex_metrics("apex_mic50", apex_pathogen_scores), "apex_mic90": self.apex_metrics("apex_mic90", apex_pathogen_scores), "apex_gram_positive_mean": self.apex_metrics("apex_gram_positive_mean", apex_pathogen_scores), 
                                "apex_gram_negative_mean": self.apex_metrics("apex_gram_negative_mean", apex_pathogen_scores), "gram_negative_selectivity": self.apex_metrics("gram_negative_selectivity", apex_pathogen_scores),  
                                "gram_positive_selectivity": self.apex_metrics("gram_positive_selectivity", apex_pathogen_scores), "apex_GP_mic50": self.apex_metrics("apex_GP_mic50", apex_pathogen_scores), "apex_GP_mic90": self.apex_metrics("apex_GP_mic90", apex_pathogen_scores),
                                "apex_GN_mic50": self.apex_metrics("apex_GN_mic50", apex_pathogen_scores), "apex_GN_mic90": self.apex_metrics("apex_GN_mic90", apex_pathogen_scores), 
                                "apex_mdr_mean": self.apex_metrics("apex_mdr_mean", apex_pathogen_scores), "apex_mdr_mic50": self.apex_metrics("apex_mdr_mic50", apex_pathogen_scores), 
                                "apex_mdr_mic90": self.apex_metrics("apex_mdr_mic90", apex_pathogen_scores), 
                                })
        

        omegamp_prob_scores =  self.evaluator.OmegAMPScorer.predict(generated_peptides_list)
        omegaamp_df = pd.DataFrame({"id": generated_df["id"].values, "omegAMP_scores":omegamp_prob_scores['omegamp_amp_prob']})

        #2. Property distribution
        physchem_properites = calculate_physchem_prop(generated_peptides_list)
        property_df = pd.DataFrame({"id": generated_df["id"].values, "hydrophobicity": physchem_properites["hydrophobicity"], 
                            "hydrophobic_moment": physchem_properites["hydrophobic_moment"], "charge": physchem_properites["charge"],
                            "isoelectric_point": physchem_properites["isoelectric_point"]
                            })
        
        #3. Marlys MMSeq2 check 
        print("------")
        print("MMseqs cluster analysis against MarLys database")
        print("------")
        marlys_results = mmseqs_marlys_similarity(config=config, query_fasta=path_to_generated_peptides, marlys_fasta=config.marlys_fasta)
        mmseqs_df = pd.DataFrame.from_dict(marlys_results, orient="index").reset_index(drop=True)
        generated_df = (generated_df.merge(apex_df, on="id", how="left", validate="one_to_one").merge(omegaamp_df, on="id", how="left", validate="one_to_one").merge(mmseqs_df, on="id", how="left", validate="one_to_one").merge(property_df, on="id", how="left", validate="one_to_one"))
        #generated_df_path = os.path.join(config.results_path, "generated_50k.csv")
        #generated_df.to_csv(generated_df_path)

        return generated_df
    

    
    def apex_metrics(self, metrics, scores):
        
        gram_neg_scores = scores[:, 0:6]
        gram_pos_scores = scores[:, 7:11]
        gram_mdr_scores = gram_mdr_scores = scores[:, [3, 8, 9, 10]] # rows corresponding to MDR strains

        if metrics == 'apex_mic50':
            return np.median(scores, axis=1)
        if metrics == 'apex_mic90':
            return np.quantile(scores,0.90,axis=1,method="higher")
        if metrics == 'apex_gram_positive_mean':
            return np.mean(gram_pos_scores, axis=1)
        if metrics == 'apex_GP_mic50':
            return np.median(gram_pos_scores, axis=1)
        if metrics == 'apex_GP_mic90':
            return np.quantile(gram_pos_scores,0.90,axis=1,method="higher")
        if metrics == 'apex_gram_negative_mean':
            return np.mean(gram_neg_scores, axis=1)
        if metrics == 'apex_GN_mic50':
            return np.median(gram_neg_scores, axis=1)
        if metrics == 'apex_GN_mic90':
            return np.quantile(gram_neg_scores,0.90,axis=1,method="higher")
        if metrics == 'apex_mdr_mean':
            return np.mean(gram_mdr_scores, axis=1)
        if metrics == 'apex_mdr_mic50':
            return np.median(gram_mdr_scores, axis=1)
        if metrics == 'apex_mdr_mic90':
            return np.quantile(gram_mdr_scores,0.90,axis=1,method="higher")

        
        if metrics == 'gram_positive_selectivity' or metrics == 'gram_negative_selectivity':

            gram_neg_median = np.median(gram_neg_scores, axis=1)
            gram_pos_median = np.median(gram_pos_scores, axis=1)

            if metrics == 'gram_positive_selectivity':
                return (gram_pos_median / gram_neg_median)
            else:
                return (gram_neg_median / gram_pos_median)
            


def dataframe_to_temp_fasta(df):
    """
    Create a temporary FASTA file from a DataFrame
    containing 'id' and 'sequence' columns.

    Returns the path to the temporary FASTA file.
    """
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".fasta",
        delete=False
    )

    for _, row in df.iterrows():
        tmp.write(f">{row['id']}\n{row['sequence']}\n")

    tmp.close()

    return tmp.name


def log_full_results(config, log_metrics):
    os.makedirs(config.results_path, exist_ok=True)
    file_log_path = os.path.join(
        config.results_path,
        "global_metrics.txt"
    )
    with open(file_log_path, "a+") as f:
        f.write(json.dumps(log_metrics))
        f.write("\n")


def normalize_sequence(sequence: str) -> str:
    
    """
        Normalize a peptide sequence. 
        Whitespace is removed and letters are converted to uppercase.
    """
    return "".join(str(sequence).split()).upper()


def read_fasta_sequences(path: str | Path) -> list[str]:
    
    """
        Read sequences from a FASTA file.

    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"FASTA file not found: {path}")

    sequences: list[str] = []
    sequence_parts: list[str] = []

    with path.open("r") as handle:
        for raw_line in handle:
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if sequence_parts:
                    sequences.append(
                        normalize_sequence("".join(sequence_parts))
                    )
                    sequence_parts = []
            else:
                sequence_parts.append(line)

    if sequence_parts:
        sequences.append(
            normalize_sequence("".join(sequence_parts))
        )

    return sequences


def read_fasta_return_sequence_list(path_to_fasta_file):
    all_sequences = []
    for seq_record in SeqIO.parse(path_to_fasta_file, "fasta"):
        all_sequences.append((seq_record.id, str(seq_record.seq)))
    return all_sequences

def save_fasta(sequences, path):
    with open(path, "w") as f:
        for identifier, sequence in sequences:
            f.write(f">{identifier}\n{sequence}\n")


def check_amp_synthesizability(
    sequence: str,
    charge_range: tuple[float, float] = (2.0, 10.0),
    hydrophobicity_range: tuple[float, float] = (-0.5, 0.8),
    hydrophobic_moment_cutoff: float = (0.3, 0.6),
    max_cysteines: int = 1,
) -> dict:
    
    """
    Apply physicochemical/developability filters to an AMP candidate.

        Returns True if all filters are passed 

    """


    # Synthesizability criteria:
    # Net charge (2-10) # AMPs need net positive charge at pH 7.4: 
    # Hydrophobicity (-0.5-0.8)
    # amphipathicity: Hydrophobic moment HM (0.3-0.6)
    # Not more than 3 consecutive hydrophobic residues 
    # Not more than 1 Cys disulfide scrambling during/after SPPS
    # No run of more than 2 consecutive glycines
    # Proline content is not more than 20% of residues

    # Taken from OmegaAMP https://arxiv.org/html/2504.17247 and https://www.biorxiv.org/content/10.64898/2026.09.01.747572v1.full.pdf


    hydrophobic_mom = calculate_hydrophobicmoment([sequence])
    hydrophobicity = calculate_hydrophobicity([sequence])
    charge = calculate_charge([sequence])
    n_cys = sequence.count("C")

    if not charge_range[0] <= charge <= charge_range[1]:
        return False
    
    if not hydrophobicity_range[0] <= hydrophobicity <= hydrophobicity_range[1]:
        return False

    if not hydrophobic_moment_cutoff[0] <= hydrophobic_mom <= hydrophobic_moment_cutoff[1]:
        return False
    
    if sequence.count("P") / len(sequence) > 0.20:
        return False
    
    if not check_sequence_for_hydrophobic_clusters(sequence):
        return False
    
    if "GGG" in sequence:
        return False

    if n_cys > max_cysteines + 1:
        return False
    
    return True

def check_sequence_for_hydrophobic_clusters(sequence: str, max_run: int = 3) -> bool:

    # Filter out of three hydrophobic residues consecutively
    
    HYDROPHOBIC_AA = set("FILVWMA")
    run = 0

    for aa in sequence:
        if aa in HYDROPHOBIC_AA:
            run += 1

            if run >= max_run + 1:
                return False
        else:
            run = 0

    return True

def candidate_selection_for_SILO_training(trajectories, 
                          seen_protein_smiles, config) -> np.array:
    

    """
    Candidate selection for choosing top trajectories to train from. The following requirements should be met:
        -- Use only the 20 standard proteinogenic amino acids (checked in basic_validity_mask function)
        -- the length of the peptide must be better a permissible range (8 to 50) (checked in basic_validity_mask function)
        -- Be unique (no duplicates) (checked in basic_validity_mask function)
        -- Must be unique from the known antibacterial peptides in antibacterial.fasta file. 
    """

    local_seen = set()
    new_unique = []

    if isinstance(trajectories, dict):
        iterator = [traj for _, traj in trajectories.items()]
    elif isinstance(trajectories, list):
        iterator = [traj for traj in trajectories] 

    # Identify which trajectories are new unique SMILES
    for traj in iterator:
        peptide = traj['peptide']

        # Only keep peptides between 8 and 50 amino acids
        if not config.min_max_seq_length[0] <= len(peptide) <= config.min_max_seq_length[1]:
            continue

        # duplicate within this batch
        if peptide in local_seen:
            continue
        local_seen.add(peptide)

        # already evaluated previously (global cache, which includes all generated sequences as well as sequences from antibacterial.fasta)
        if peptide not in seen_protein_smiles:
            new_unique.append(traj)

    final_trajs = sorted(new_unique, key=lambda x: x['objective'])
    return final_trajs
