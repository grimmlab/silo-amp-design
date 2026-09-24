import argparse, copy, importlib, os, mlflow, datetime

from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import autocast
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from logger import Logger
from sequence_dataset import PolicyTrainingDataset

import torch
import numpy as np
from ..config import SequenceConfig
from ..utils import save_checkpoint, save_checkpoint, set_seed
from ..model.transformer_architecture import SequenceTransformer, dict_to_cpu


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed",
                    type=int,
                    default=42,
                    help="Random seed")
    
    parser.add_argument("--device",
                    type=str,
                    default='cuda:1',
                    help="specify device name: either cuda:gpu_num (cuda:0) or cpu")

    parser.add_argument("--results",
                    type=str,
                    default="./results",
                    help="specify directory for storing results")
    
    parser.add_argument("--pretrain_train_dataset",
                    type=str,
                    default="/home/akhanna/AMP/SILO_amp/pretrain/final_train_pretrain.pickle",
                    help="pickle file for training")
    
    parser.add_argument("--pretrain_val_dataset",
                    type=str,
                    default="/home/akhanna/AMP/SILO_amp/pretrain/final_validation_pretrain.pickle",
                    help="pickle file for validation")
    
    args = parser.parse_args()
    return args

def train_for_one_epoch(epoch: int, config: SequenceConfig, network: SequenceTransformer,
                        optimizer: torch.optim.Optimizer, dataset: PolicyTrainingDataset, is_validation=False):
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False, persistent_workers=False)

    metrics = dict()
    scaler = torch.amp.GradScaler()
    criterion = CrossEntropyLoss(reduction="mean", ignore_index=-1)

    # Train for one epoch
    network.train() if not is_validation else network.eval()

    accumulated_loss = 0

    num_batches = len(dataloader)
    progress_bar = tqdm(range(num_batches))
    data_iter = iter(dataloader)

    for _ in progress_bar:
        data = next(data_iter)
        input_data = {k: v[0].to(network.device) for k, v in data["input"].items()}

        # targets for the logit levels
        targets = data["targets"].squeeze(0).to(network.device)

        with autocast(device_type=config.training_device, dtype=torch.bfloat16): 

            logits = network(input_data)

            # We mask the output according to feasibility            
            
            logits[input_data["feasibility_mask"]] = float("-inf")
            
            # loss is calculated only for the logits of the selected position
            loss = criterion(logits, targets)
            loss = torch.tensor(0.) if torch.isnan(loss) else loss

        if not is_validation:
            # Optimization step
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            if config.optimizer["gradient_clipping"] > 0:
                torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=config.optimizer["gradient_clipping"])

            scaler.step(optimizer)

            # Updates the scale for next iteration.
            scaler.update()

        batch_loss = loss.item()
        accumulated_loss += loss.item()

        progress_bar.set_postfix({"batch_loss": batch_loss})

        del data

    metric_prefix = "" if not is_validation else "val_"
    metrics[f"{metric_prefix}full_loss"] = accumulated_loss / num_batches

    return metrics

def main(args):
    pretrain_num_epochs = 400
    batch_size = 128
    num_batches_per_epoch = 64
    batch_size_validation = 128
    training_device = "cuda:1"  # Device on which to train.
    num_dataloader_workers = 10  # Number of dataloader workers for creating batches for training
    load_checkpoint_from_path = None

    print(">> Pretraining SILO v2.0")
    config = SequenceConfig(args=args)
    config.results_path = os.path.join(f"{args.results}/pretrain/8_50_PE_valid_mask_LD_512_BL_10_head_16", f"{args.seed}")
    os.makedirs(config.results_path, exist_ok=True)
    print(f"Results path: {config.results_path}")
    config.num_dataloader_workers = num_dataloader_workers
    config.if_pretrain = True
    logger = Logger(args, config.results_path, config.log_to_file)
    logger.log_hyperparams(config)

    # set up mlflow connection 
    '''set_mlflow_connection() 
    model_start_time = f"pretrain_{config.tasks_configs['task']}_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if config.mlflow_experiment is None:
        mlflow.set_experiment(experiment_name=config.protein_name + '_' + config.mode + '_' + 'replan_20_num_of_traj_10' + '_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))                        
    else:
        mlflow.set_experiment(config.mlflow_experiment)'''

    # Fix random number generator seed for better reproducibility
    set_seed(config.seed)

    # Setup the policy network for training
    network = SequenceTransformer(config, config.training_device)

    # Load checkpoint if needed
    if load_checkpoint_from_path is not None:
        print(f"Loading checkpoint from path {load_checkpoint_from_path}")
        checkpoint = torch.load(load_checkpoint_from_path)
        print(f"{checkpoint['pretrain_epochs_trained']} episodes have been trained in the loaded checkpoint.")
    else:
        checkpoint = {
            "model_weights": None,
            "best_model_weights": None,
            "optimizer_state": None,
            "pretrain_epochs_trained": 0,
            "pretrain_best_validation_loss": float("inf"),
            "epochs_trained": 0,
            "validation_metric": float("-inf"),   # objective of the best sequence designed during validation.
            "best_validation_metric": float("-inf")  # corresponding to best model weights
        }
    if checkpoint["model_weights"] is not None:
        network.load_state_dict(checkpoint["model_weights"])

    print(f"Policy network is on device {training_device}")
    network.to(network.device)
    network.eval()

    if pretrain_num_epochs > 0:
        # Training loop
        print(f"Starting pre-training for {pretrain_num_epochs} epochs.")
        best_validation_metric = checkpoint["pretrain_best_validation_loss"]
        print("Setting up optimizer.")
        optimizer = torch.optim.Adam(
            network.parameters(),
            lr=config.optimizer["lr"],
            weight_decay=config.optimizer["weight_decay"]
        )
        if checkpoint["optimizer_state"] is not None and config.load_optimizer_state:
            print("Loading optimizer state from checkpoint.")
            optimizer.load_state_dict(
                checkpoint["optimizer_state"]
            )

        print("Setting up LR scheduler")
        _lambda = lambda epoch: config.optimizer["schedule"]["decay_factor"] ** (
                    checkpoint["pretrain_epochs_trained"] // config.optimizer["schedule"]["decay_lr_every_epochs"])
        
        scheduler = LambdaLR(optimizer, lr_lambda=_lambda)

        train_dataset = PolicyTrainingDataset(config, args.pretrain_train_dataset,
                                              batch_size=batch_size,
                                              custom_num_batches=num_batches_per_epoch, 
                                              no_random=False, if_pretrain=True, if_train=True)
        
        val_dataset = PolicyTrainingDataset(config, args.pretrain_val_dataset,
                                            batch_size=batch_size_validation,
                                            custom_num_batches=None,
                                            no_random=True, if_pretrain=True, if_train=False)
        
        #with mlflow.start_run(run_name = model_start_time):
            
            #mlflow.log_params({k: v for k, v in vars(config).items() if isinstance(v, (int, float, str, bool))})

        patience = config.optimizer.get("early_stopping_patience", 10)
        min_delta = config.optimizer.get("early_stopping_min_delta", 1e-4)
        num_bad_epochs = 0

        for epoch in range(pretrain_num_epochs):
            print("Training.")

            generated_loggable_dict = train_for_one_epoch(epoch, config, network, optimizer, train_dataset)
            
            checkpoint["pretrain_epochs_trained"] += 1
            scheduler.step()
            print(f">> Epoch {checkpoint['pretrain_epochs_trained']}. Avg loss: {generated_loggable_dict['full_loss']}")
            logger.log_metrics(generated_loggable_dict, step=epoch)

            # log metrics per epoch 
            '''for key, val in generated_loggable_dict.items():
                mlflow.log_metric(key, val, step=epoch)'''

            print("Validating...")
            torch.cuda.empty_cache()
            with torch.no_grad():
                validation_metrics = train_for_one_epoch(
                    None, config, network, None, val_dataset, is_validation=True
                )
            
            # log metrics per epoch 
            '''for key, val in validation_metrics.items():
                mlflow.log_metric(key, val, step=epoch)'''

            checkpoint["model_weights"] = copy.deepcopy(network.get_weights())
            checkpoint["optimizer_state"] = copy.deepcopy(dict_to_cpu(optimizer.state_dict()))
            save_checkpoint(checkpoint, "last_model.pt", config)

            current_val_loss = validation_metrics["val_full_loss"]
            logger.log_metrics(validation_metrics, step=epoch)
            
            if current_val_loss  < checkpoint["pretrain_best_validation_loss"] - min_delta:
                print(">> Got new best model.")
                checkpoint["pretrain_best_validation_loss"] = current_val_loss
                num_bad_epochs = 0

                print(f">> New best validation loss: {current_val_loss}")
                print(f">> Validation. Avg loss: {validation_metrics['val_full_loss']}")
                save_checkpoint(checkpoint, "best_model.pt", config)

            else:
                    num_bad_epochs += 1
                    print(f">> Validation. Avg loss: {validation_metrics['val_full_loss']}")
                    print(f"No improvement for {num_bad_epochs}/{patience} epochs.")
            
            # early stopping trigger
            if num_bad_epochs >= patience:
                print("Early stopping triggered.")
                break

if __name__=='__main__':
    args = parse_args()
    main(args)