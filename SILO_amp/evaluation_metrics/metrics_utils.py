import tempfile, subprocess, os, Levenshtein
from Bio import Align, SeqIO
import pandas as pd 
import modlamp.analysis as manalysis
import numpy as np
from typing import List, Dict
from tqdm import tqdm
import pickle
import numpy as np


def local_similarity(a: str, b: str) -> float:
    _aligner = Align.PairwiseAligner()
    _aligner.mode = "local"
    _aligner.match_score = 1.0
    _aligner.mismatch_score = -1.0
    _aligner.open_gap_score = -1.0
    _aligner.extend_gap_score = -1.0

    """Normalized Smith-Waterman local-alignment similarity between two peptides, in [0, 1]."""
    score = _aligner.score(a, b)
    return max(0.0, score) / max(len(a), len(b))
    

def calculate_hydrophobicity(data: List[str]) -> np.ndarray:
    """
        Taken from https://github.com/szczurek-lab/hydramp/blob/master/amp/utils/phys_chem_propterties.py 
    """
    h = manalysis.GlobalAnalysis(data)
    h.calc_H(scale='eisenberg')
    return h.H[0]


def calculate_hydrophobicmoment(data: List[str]) -> np.ndarray:
    h = manalysis.PeptideDescriptor(data, 'eisenberg')
    h.calculate_moment()
    return h.descriptor.flatten()


def calculate_charge(data: List[str]) -> np.ndarray:
    h = manalysis.GlobalAnalysis(data)
    h.calc_charge()
    return h.charge[0]


def calculate_isoelectricpoint(data: List[str]) -> np.ndarray:
    h = manalysis.GlobalDescriptor(data)
    h.isoelectric_point()
    return h.descriptor.flatten()


def calculate_length(sequences: List[str]) -> np.ndarray:
    return np.array([len(seq) for seq in sequences])


def calculate_physchem_prop(sequences: List[str]) -> Dict[str, object]:
    return {
        "length": calculate_length(sequences).tolist(),
        "hydrophobicity": calculate_hydrophobicity(sequences).tolist(),
        "hydrophobic_moment": calculate_hydrophobicmoment(sequences).tolist(),
        "charge": calculate_charge(sequences).tolist(),
        "isoelectric_point": calculate_isoelectricpoint(sequences).tolist(),
    }
    

def read_fasta_return_sequence_list(path_to_fasta_file):
     
    all_sequences = []

    for seq_record in SeqIO.parse(path_to_fasta_file, "fasta"):
        all_sequences.append((seq_record.id, str(seq_record.seq)))

    return all_sequences


def mmseqs_marlys_similarity(config, query_fasta, marlys_fasta: str, identity_threshold: float = 80.0) -> dict:
    
    """
    Read MMseqs2 search results and return the best MarLys match
    for each generated peptide.
    """
    generated_peptides_list = read_fasta_return_sequence_list(query_fasta)
    generated_peptides_dict = dict(generated_peptides_list)


    mmseq_dir = os.path.join(config.results_path, "mmseq")
    os.makedirs(mmseq_dir, exist_ok=True)
    output_tsv = os.path.join(mmseq_dir, "marlys_hits.tsv")
    mmseqs_tmp = os.path.join(mmseq_dir, "mmseqs_tmp")

    # run mmseq2 command 
    # Parameters for running for peptides from here https://www.biorxiv.org/content/10.64898/2026.09.01.747572v1.full.pdf

    command = ["mmseqs", "easy-search", query_fasta, marlys_fasta, output_tsv,  mmseqs_tmp, "--comp-bias-corr", "0",
        "--prefilter-mode", "2", "-e", "1000", "--alignment-mode", "3", "-c", "0.8", "--cov-mode", "2", "--format-output", "query,target,pident,alnlen,qlen,tlen,evalue,bits", 
        "--threads", "16"]
    
    subprocess.run(command,check=True)

    # read outputs 
    columns = ["query", "target", "pident","alnlen","qlen","tlen","evalue","bits"]
    hits = pd.read_csv(output_tsv,sep="\t",names=columns)
    
    # load marlys sequences
    marlys_sequences = {record.id: str(record.seq)for record in SeqIO.parse(marlys_fasta, "fasta")}

    # choose best hit 
    best_hits = (hits.sort_values(["query", "pident", "alnlen", "bits"],ascending=[True, False, False, False],).drop_duplicates("query"))

    results = {}

    for _, row in best_hits.iterrows():
        query_id = str(row["query"])
        seq = generated_peptides_dict[query_id]
        identity = float(row["pident"])
        target_id = row["target"]

        results[query_id] = {
            "id": query_id,
            "max_marlys_identity": identity,
            "closest_marlys_id": target_id,
            "closest_marlys_sequence": marlys_sequences.get(target_id),
            "marlys_bitscore": float(row["bits"]),
            "marlys_evalue": float(row["evalue"]),
            "passes_marlys_80": identity <= identity_threshold,
        }

    return results

def novelty_against_reference(seq, training_amps, antibacterial_reference_amps):

    """
    references should look like:
    [
        {
            "id": "DRAMP00001",
            "sequence": "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK"
        },
        ...
    ]
    """
    train_best_similarity = -1.0
    train_best_reference = None
    
    anti_best_similarity = -1.0
    anti_best_reference = None

    for ref in training_amps: 
        similarity = local_similarity(seq["sequence"], ref[1])
        if similarity > train_best_similarity:
            train_best_similarity = similarity
            train_best_reference = ref

    for anti_ref in antibacterial_reference_amps:
        lv_similarity = Levenshtein.ratio(seq["sequence"], anti_ref[1])
        if lv_similarity > anti_best_similarity:
            anti_best_similarity = lv_similarity
            anti_best_reference = anti_ref

    return {
        # for training data matching
        "training_data_novelty": 1.0 - train_best_similarity,
        "max_train_reference_similarity": train_best_similarity,
        "passes_local_similarity_check": train_best_similarity < 0.60,
        "closest_train_reference_id": train_best_reference[0],
        "closest_train_reference_sequence": train_best_reference[1],

        # for antibacterial.fasta matching 
        "ab_data_novelty": 1.0 - anti_best_similarity,
        "max_ab_reference_similarity": anti_best_similarity,
        "passes_ab_novelty": anti_best_similarity < 0.80,
        "closest_ab_reference_id": anti_best_reference[0],
        "closest_ab_reference_sequence": anti_best_reference[1],

    }

def read_fasta_return_sequence_list(path_to_fasta_file):
     
    all_sequences = []
    for seq_record in SeqIO.parse(path_to_fasta_file, "fasta"):
        all_sequences.append((seq_record.id, str(seq_record.seq)))
    return all_sequences

def save_fasta(sequences, path):
    with open(path, "w") as f:
        for identifier, sequence in sequences:
            f.write(f">{identifier}\n{sequence}\n")
