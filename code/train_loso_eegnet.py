import os
import time
import argparse
import sys
import gc
import csv
from pathlib import Path
import logging

import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score

import mne
import braindecode
from braindecode.preprocessing import (
    Preprocessor, preprocess, create_windows_from_events, exponential_moving_standardize
)
from braindecode.datasets import MOABBDataset
from braindecode.models import EEGNetv4
from braindecode.util import set_random_seeds

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# --- DEVICE DETECTION & PRINT ---
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def get_preprocessors(sfreq):
    def convert_v_to_uv(x): return x * 1e6
    return [
        Preprocessor('pick_types', eeg=True, eog=False, stim=False, meg=False),
        Preprocessor(convert_v_to_uv),
        # Φίλτρο 4-38Hz (όπως στο paper)
        Preprocessor('filter', l_freq=4, h_freq=38),
        Preprocessor("resample", sfreq=sfreq),
        Preprocessor(exponential_moving_standardize, factor_new=1e-3, init_block_size=1000),
    ]

def prepare_data(subjects, sfreq, num_workers):
    all_data = {}
    logger.info("--- Starting Data Preparation ---")
    preprocessors = get_preprocessors(sfreq)
    for subject_id in subjects:
        try:
            logger.info(f"Processing Subject {subject_id}...")
            dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=[subject_id])
            preprocess(dataset, preprocessors, n_jobs=num_workers)
            
            reject_criteria = {"eeg": 800} 
            windows_ds = create_windows_from_events(
                dataset, trial_start_offset_samples=int(0.5 * sfreq), trial_stop_offset_samples=0,
                window_size_samples=int(2.0 * sfreq), window_stride_samples=int(2.0 * sfreq),
                drop_last_window=True, preload=True, drop_bad_windows=True,
                reject=reject_criteria, use_mne_epochs=True, on_missing="ignore", n_jobs=num_workers,
            )
            all_data[subject_id] = windows_ds
        except Exception as e:
            logger.error(f"Failed Subject {subject_id}: {e}")
    return all_data

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.best = None
        self.wait = 0
    def step(self, val_loss):
        if self.best is None or val_loss < self.best:
            self.best = val_loss
            self.wait = 0
            return False
        self.wait += 1
        return self.wait > self.patience

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for X, y, _ in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0; preds = []; targets = []
    with torch.no_grad():
        for X, y, _ in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            total_loss += criterion(outputs, y).item() * X.size(0)
            preds.extend(outputs.argmax(dim=1).cpu().numpy())
            targets.extend(y.cpu().numpy())
    return total_loss / len(loader.dataset), accuracy_score(targets, preds), cohen_kappa_score(targets, preds)

def run_loso(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # --- CHECK DEVICE ---
    device = get_device()
    print("\n" + "="*40)
    print(f"DEVICE IN USE: {device.upper()}")
    print("="*40 + "\n")
    
    set_random_seeds(seed=args.seed, cuda=(device=="cuda"))
    all_subjects_list = list(range(1, 10))
    data = prepare_data(all_subjects_list, args.sfreq, args.num_workers if args.num_workers > 0 else 1)
    
    for test_sub in args.subjects:
        if test_sub not in data: continue
        t0 = time.time()
        
        train_ds = ConcatDataset([data[s] for s in all_subjects_list if s != test_sub and s in data])
        val_len = int(len(train_ds) * args.val_frac)
        train_sub, val_sub = random_split(train_ds, [len(train_ds) - val_len, val_len])
        
        train_dl = DataLoader(train_sub, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_dl = DataLoader(val_sub, batch_size=args.batch_size, num_workers=0)
        test_dl = DataLoader(data[test_sub], batch_size=args.batch_size, num_workers=0)
        
        # --- MODEL SETUP ---
        model = EEGNetv4(
            n_chans=22, 
            n_outputs=4, 
            n_times=int(args.sfreq*2), 
            final_conv_length='auto',
            drop_prob=0.25      # Dropout probability for Cross-Subject
             
        ).to(device)
        
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
        stopper = EarlyStopping(patience=args.patience)
        
        best_state = None
        history = {'train_loss': [], 'val_loss': []}
        
        pbar = tqdm(range(args.epochs), desc=f"Subj {test_sub}", file=sys.stdout)
        for epoch in pbar:
            t_loss = train_epoch(model, train_dl, optimizer, CrossEntropyLoss(), device)
            
            v_loss = None; v_acc = 0; v_kappa = 0
            if val_dl:
                v_loss, v_acc, v_kappa = evaluate(model, val_dl, CrossEntropyLoss(), device)
                pbar.set_postfix({'val_loss': f"{v_loss:.4f}", 'acc': f"{v_acc:.2f}", 'kappa': f"{v_kappa:.2f}"})
            
            scheduler.step()
            
            history['train_loss'].append(t_loss)
            history['val_loss'].append(v_loss if v_loss else np.nan)
            
            if v_loss is not None:
                if stopper.best is None or v_loss < stopper.best:
                    best_state = model.state_dict()
                if stopper.step(v_loss):
                    break

        if best_state: model.load_state_dict(best_state)
        test_loss, test_acc, test_kappa = evaluate(model, test_dl, CrossEntropyLoss(), device)
        
        # Καθαρό CSV μόνο με τα απαραίτητα
        res = {
            'subject': test_sub, 'acc': test_acc, 'kappa': test_kappa,
            'epochs': args.epochs, 'lr': args.lr,'batch size': args.batch_size,
            'time': time.time() - t0
        }
        
        csv_file = os.path.join(args.output_dir, 'results_log.csv')
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=res.keys())
            if not file_exists: w.writeheader()
            w.writerow(res)
            
        torch.save({'model': model.state_dict(), 'args': vars(args), 'history': history}, 
                   os.path.join(args.output_dir, f"model_subj_{test_sub}.pt"))
        gc.collect()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--subjects', type=int, nargs='+', default=[1])
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--patience', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--val_frac', type=float, default=0.1)
    p.add_argument('--sfreq', type=int, default=128)
    p.add_argument('--num_workers', type=int, default=-1)
    p.add_argument('--output_dir', type=str, default='./experiments')
    p.add_argument('--seed', type=int, default=42)
    run_loso(p.parse_args())

