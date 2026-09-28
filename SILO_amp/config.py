from pathlib import Path
SILO_DIR = Path(__file__).resolve().parent 

class SequenceConfig:
    def __init__(self, args):

        # Network 
        self.latent_dimension = 512
        self.num_transformer_blocks = 10
        self.num_heads = 16
        self.dropout = 0.0

         # Environment  
        self.residue_vocabulary = {  
            "A":  {"allowed": True},
            "R":  {"allowed": True},
            "N":  {"allowed": True},
            "D":  {"allowed": True},
            "C":  {"allowed": True},
            "E":  {"allowed": True},
            "Q":  {"allowed": True},
            "G":  {"allowed": True},
            "H":  {"allowed": True},
            "I":  {"allowed": True},
            "L":  {"allowed": True},
            "K":  {"allowed": True},
            "M":  {"allowed": True},
            "F":  {"allowed": True},
            "P":  {"allowed": True},
            "S":  {"allowed": True},
            "T":  {"allowed": True},
            "W":  {"allowed": True},
            "Y":  {"allowed": True},
            "V":  {"allowed": True}
        }

        self.seed = args.seed # set a random seed number  
        self.training_device = args.device # set device either as cuda:gpu_num ('cuda:0') or cpu 
        self.total_peptide_count = 50000
        self.top_k_peptides = 100
        self.do_inference = False

        self.num_predictor_workers = 1 
        self.training_cycles = args.epoches # number of active learning rounds 
        self.min_max_seq_length= [8, 50] # change this after pretraining
        self.multiplier = 20

        # Training for policy 
        self.num_dataloader_workers = 3  # Number of workers for creating batches for training
        self.num_epochs = 1
        self.CUDA_VISIBLE_DEVICES = "0,1"  # Must be set, as ray can have problems detecting multiple GPUs
        self.batch_size_training = 8
        self.training_fasta = SILO_DIR / "data/training.fasta" 
        self.marlys_fasta = SILO_DIR/ "data/marlys.fasta"
        self.antibacterial_fasta = SILO_DIR/ "data/antibacterial.fasta"
        
        self.load_checkpoint_from_path = None
        self.if_pretrain = False
        self.num_batches_per_epoch = None  # Can be None, then we just do one pass through generated dataset

        # Optimizer for policy 
        self.mlflow_experiment = 'finetuning'
        self.optimizer = {
            "lr": 1e-4,  # learning rate
            "weight_decay": 0,
            "gradient_clipping": 1.,  # Clip gradient to given L2-norm. Set to 0 if no clipping should be performed.
            "schedule": {
                "decay_lr_every_epochs": 5,
                "decay_factor": 0.8
            }
        }

        self.log_to_file = True
        self.max_objective = False
        
        # Self-improvement sequence decoding
        self.self_improvement_learning = {
            "num_trajectories_to_keep": 100, # Number of trajectories with the the highest objective function evaluation to keep for training
            "keep_intermediate_trajectories": False,
            "devices_for_workers": [f"{self.training_device}"] * 1,
            "batch_size_per_worker": 4, 
            "batch_size_per_cpu_worker": 4,
            "search_type": "wor",
            "beam_width": 64,
            "num_rounds": 1,  # if it's a tuple, then we sample as long as it takes to obtain a better trajectory, but for a minimum of first entry rounds and a maximum of second entry rounds
            "deterministic": False,  # when True, switches to regular beam search.
            "nucleus_top_p": 1.,
            "pin_workers_to_core": False, 
            "num_traj_test": 100, 
        }
        
        self.APEX_COLUMN_MAP = {
            "A_baumannii": "A. baumannii ATCC 19606",
            "E_coli_11775": "E. coli ATCC 11775",
            "E_coli_AIC221": "E. coli AIC221",
            "E_coli_AIC222": "E. coli AIC222",
            "K_pneumoniae": "K. pneumoniae ATCC 13883",
            "P_aeruginosa_PAO1": "P. aeruginosa PA01",
            "P_aeruginosa_PA14": "P. aeruginosa PA14",
            "S_aureus": "S. aureus ATCC 12600",
            "MRSA": "S. aureus (ATCC BAA-1556) - MRSA",
            "VRE_faecalis": "vancomycin-resistant E. faecalis ATCC 700802",
            "VRE_faecium": "vancomycin-resistant E. faecium ATCC 700221",
        }
