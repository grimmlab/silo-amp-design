import argparse, copy, os, time, ray, torch, datetime
from .logger import Logger 
import numpy as np
from .config import SequenceConfig
from .model.transformer_architecture import SequenceTransformer, dict_to_cpu
from .sequence_evaluator import SequenceEvaluator
from .utils import save_checkpoint, train_for_one_cycle, set_seed
from .evaluation_metrics.utils import read_fasta_return_sequence_list
from .generate import run_inference
from torch.optim.lr_scheduler import LambdaLR

def parse_args():

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Random seed")
    
    parser.add_argument("--epoches",
                        type=int,
                        default=100,
                        help="Number of epoches for training")
    
    parser.add_argument("--device",
                        type=str,
                        default='cuda:0',
                        help="specify device name: either cuda:gpu_num (cuda:0) or cpu")
    
    parser.add_argument("--results",
                        type=str,
                        default="./results/",
                        help="specify directory for storing results")
    
    parser.add_argument("--comments",
                        type=str,
                        help="type of experiment")
    
    
    args = parser.parse_args()
    return args


def main(args):

    os.environ["RAY_DEDUP_LOGS"]="0"
    os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"]="1"


    print("------")
    print(">> Protein Sequence Design using SILO v2.0")

    config = SequenceConfig(args=args)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES
    num_gpus = len(config.CUDA_VISIBLE_DEVICES.split(","))
    ray.init(num_gpus=num_gpus, log_to_driver=False, logging_level="info") 
    print(ray.available_resources())

    config.results_path = os.path.join(f"{args.results}", f"{args.seed}")
    os.makedirs(config.results_path, exist_ok=True)

    logger = Logger(config, config.results_path, config.log_to_file)
    logger.log_hyperparams(config)
    set_seed(config.seed)
    sequence_evaluator = SequenceEvaluator(config)
    seen_protein_smiles = read_fasta_return_sequence_list(config.antibacterial_fasta) # to ensure no generated sequence is identical to that in antibacterial.fasta 

    # Setup the policy network for training
    network = SequenceTransformer(config, config.training_device)

    # Initalize checkpoint dict
    if config.load_checkpoint_from_path is not None:
        print(f"Loading checkpoint from path {config.load_checkpoint_from_path} for further finetuning")
        checkpoint = torch.load(config.load_checkpoint_from_path, weights_only=False)
        print(f"{checkpoint['pretrain_epochs_trained']} episodes have been trained in the loaded checkpoint.")
    
    else: 
        checkpoint = {
            "model_weights": None,
            "best_model_weights": None,
            "optimizer_state": None,
            "best_optimizer_state": None,
            "epochs_trained": 0,
            "validation_metric": float("inf"),   # objective of the best sequence designed during validation.
            "best_validation_metric": float("inf"),  # corresponding to best model weights
        }

    # Select the best model/optimizer pair.
    best_model_weights = copy.deepcopy(checkpoint["model_weights"]) 
    best_optimizer_state = copy.deepcopy(checkpoint["optimizer_state"]) 
    best_validation_metric = float("inf")


    print(f"Policy network is on device {config.training_device}")
    network.to(network.device)

    start_time = time.perf_counter()
    logger.log_metrics({"event": "training_started", "timestamp": datetime.datetime.now().isoformat()})

    current_model_weights = best_model_weights
    current_optimizer_state = best_optimizer_state

    # Continue training from the latest model, not necessarily the best model.
    if current_model_weights is not None:
        network.load_state_dict(current_model_weights)

    print("------")
    print("Setting up optimizer for policy.")
    optimizer = torch.optim.Adam(network.parameters(), lr=config.optimizer["lr"], weight_decay=config.optimizer["weight_decay"])

    print("Setting up LR scheduler")
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.optimizer["schedule"]["decay_lr_every_epochs"], gamma=config.optimizer["schedule"]["decay_factor"])

    if current_optimizer_state is not None:
        print("Loading optimizer state from checkpoint.")
        optimizer.load_state_dict(copy.deepcopy(current_optimizer_state))

    patience = config.optimizer.get("early_stopping_patience", 10)
    min_delta = config.optimizer.get("early_stopping_min_delta", 1e-4)
    num_bad_epochs = 0
    
    if config.training_cycles != 0: 
        print("------")
        print(f"Starting finetuning for {config.training_cycles} cycles.")

        for outer in range(config.training_cycles):


            # -------------------
            # Training loop
            # -------------------

            print("------")
            print(f"Round {outer + 1}.")
            print(f"Generating Mutant Sequences.")
            
            # These generation weights now exactly match the live policy network that will be trained.
            # model used for sampling BEFORE doing updates
            sampling_model_weights = copy.deepcopy(network.get_weights())
            sampling_optimizer_state = copy.deepcopy(dict_to_cpu(optimizer.state_dict()))

            generated_loggable_dict, generated_trajectories = train_for_one_cycle(epoch=outer, config=config, network=network, 
                            network_weights=sampling_model_weights, optimizer=optimizer, objective_evaluator=sequence_evaluator, 
                            best_objective=best_validation_metric, 
                            seen_protein_smiles=seen_protein_smiles, logger=logger)
            
            scheduler.step()
            
            # Train_for_one_cycle updated the live network.
            current_model_weights = copy.deepcopy(network.get_weights())
            current_optimizer_state = copy.deepcopy(dict_to_cpu(optimizer.state_dict()))

            checkpoint["model_weights"] = current_model_weights
            checkpoint["optimizer_state"] = current_optimizer_state

            # measure by the best mean 20 objective found during sampling
            val_metric = generated_loggable_dict["mean_top_20_obj"]   

            checkpoint["validation_metric"] = val_metric
            checkpoint["epochs_trained"] += 1

            #metric is based on MIC, and we minimize it         
            if val_metric <= best_validation_metric - min_delta:
                print(">> Got new best model.")
                best_model_weights = copy.deepcopy(sampling_model_weights)
                best_optimizer_state =  copy.deepcopy(sampling_optimizer_state)
                best_validation_metric = val_metric
                checkpoint["best_model_weights"] = copy.deepcopy(best_model_weights)
                checkpoint["best_optimizer_state"] = copy.deepcopy(best_optimizer_state)
                checkpoint["best_validation_metric"] = best_validation_metric
                num_bad_epochs = 0

                save_checkpoint(checkpoint, "best_model.pt", config)
            
            else:
                num_bad_epochs += 1
                print(f">> Validation mean MIC for top 20 {val_metric}. Best mean MIC for top 20 sequences so far: {best_validation_metric}")
                print(f"No improvement for {num_bad_epochs}/{patience} epochs.")

            save_checkpoint(checkpoint, "last_model.pt", config)
        
            # early stopping trigger
            if num_bad_epochs >= patience:
                print("Early stopping triggered.")
                break


        print("------")
        print('Training ended for policy')
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        logger.log_metrics({"event":"total_training_time_sec", "time_elapsed": elapsed})

    print("------")
    print('Inference with trained policy')
    args.checkpoint = config.results_path
    args.output_dir = config.results_path
    args.seed = config.seed
    args.device = config.training_device
    args.total_peptide_count = config.total_peptide_count
    args.top_k = config.top_k_peptides

    run_inference(args)
    
    print("Finished. Shutting down ray.")
    ray.shutdown()
        

if __name__=='__main__':
    args = parse_args()
    main(args)



