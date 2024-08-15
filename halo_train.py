import os
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from models.halo.model import HALOModel
from models.halo.config import HALOConfig
import argparse

def parse_arguments():
    prs = argparse.ArgumentParser(
        prog='halo_train.py',
        description='Train HALO model',
        epilog='Copyright (C) 2023',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    prs.add_argument("--seed", type=int, default=4)
    prs.add_argument("--dataset_path", type=str, default="data/halo/mimiciii_halo_processed.pt")
    prs.add_argument("--mask_path", type=str, default="data/halo/mimiciii_halo_mask.pt")
    prs.add_argument("--save_path", type=str, default="results/models/mimiciii_halo.pt")
    prs.add_argument("--n_layer", type=int, default=12)
    prs.add_argument("--n_head", type=int, default=18)
    prs.add_argument("--n_embd", type=int, default=1440)
    prs.add_argument("--n_positions", type=int, default=150)
    prs.add_argument("--n_ctx", type=int, default=150)
    return prs.parse_args()

def train_val_split(data, mask, val_size=0.2):
    n = len(data)
    indices = np.arange(n)
    np.random.shuffle(indices)
    split = int(np.floor(val_size * n))
    train_indices, val_indices = indices[split:], indices[:split]
    train_data, val_data = data[train_indices], data[val_indices]
    train_mask, val_mask = mask[train_indices], mask[val_indices]
    return train_data, val_data, train_mask, val_mask

class HALODataset:
    def __init__(self, processed_data, mask):
        self.processed_data = processed_data
        self.mask = mask
    
    def __len__(self):
        return self.processed_data.shape[0]

    def __getitem__(self, idx):
        return self.processed_data[idx], self.mask[idx]

class HALOTrainer:
    def __init__(self, args, HaloModel, config, train_dataloader, val_dataloader):
        self.args = args
        self.model = HaloModel
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.global_loss = 1e10
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def train(self):
        global_loss = 1e10
        for e in tqdm(range(config.epoch)):
            for i, (data, mask) in enumerate(self.train_dataloader):
                self.model.train()

                data = data.to(self.device)
                mask = mask.to(self.device)
                self.optimizer.zero_grad()
                loss, _, _ = self.model(data, position_ids=None, ehr_labels=data, ehr_masks=mask)
                loss.backward()
                self.optimizer.step()

                if i % 50 == 0:
                    print(f"Epoch {e}, Iteration {i}, Loss {loss.item()}")

                if i % 250 == 0 and i != 0:
                    self.validated = True
                    self.model.eval()
                    with torch.no_grad():
                        val_l = []
                        for j, (val_data, val_mask) in enumerate(self.val_dataloader):
                            val_data = val_data.to(self.device)
                            val_mask = val_mask.to(self.device)
                            val_loss, _, _ = self.model(val_data, position_ids=None, ehr_labels=val_data, ehr_masks=val_mask)
                            val_l.append((val_loss).cpu().detach().numpy())
                        
                        cur_val_loss = np.mean(val_l)
                        print(f"Epoch {e} Validation Loss: {cur_val_loss}")
                        if cur_val_loss < global_loss:
                            global_loss = cur_val_loss
                            state = {
                                'model': self.model.state_dict(),
                                'optimizer': self.optimizer.state_dict(),
                                'iteration': i
                            }
                            torch.save(state, self.args.save_path)
                            print('\n------------ Save best model ------------\n')
    
    def save(self):
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, self.args.save_path)
        print(f"Model saved at {self.args.save_path}")

if __name__ == "__main__":
    args = parse_arguments()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    local_rank = -1
    fp16 = False
    if local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_dataset = torch.load(args.dataset_path)
    train_mask = torch.load(args.mask_path)

    config = HALOConfig(
        total_vocab_size = train_dataset.shape[2],
        n_layer = args.n_layer,
        n_head = args.n_head,
        n_embd = args.n_embd,
        n_positions = args.n_positions,
        n_ctx = args.n_ctx,
    )

    train_data, val_data, train_mask, val_mask = train_val_split(train_dataset, train_mask)

    train_dataset = HALODataset(train_data, train_mask)
    val_dataset = HALODataset(val_data, val_mask)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    trainer = HALOTrainer(args, HALOModel(config), config, train_dataloader, val_dataloader)
    trainer.train()

    if not hasattr(trainer, 'validated'):
        trainer.save()