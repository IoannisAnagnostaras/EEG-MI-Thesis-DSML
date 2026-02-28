import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.linalg import fractional_matrix_power
import selfeeg.augmentation as aug
from selfeeg.losses import simclr_loss
import logging
import time
import random
import os

# =============================================================================
# 0. SETUP
# =============================================================================
logging.getLogger('mne').setLevel(logging.WARNING)
logging.getLogger('moabb').setLevel(logging.WARNING)
from moabb.datasets import BNCI2014_001, BNCI2014_004, Schirrmeister2017
from moabb.paradigms import MotorImagery

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"Random Seed fixed to {seed}")

# =============================================================================
# 1. DATA PROCESSING (EA & MASKING)
# =============================================================================
def euclidean_alignment(X):
    n_trials, n_channels, n_samples = X.shape
    covariances = []
    for i in range(n_trials):
        trial_data = X[i]
        cov = np.dot(trial_data, trial_data.T) / (n_samples - 1)
        covariances.append(cov)
    R = np.mean(covariances, axis=0)
    R_inv_sqrt = fractional_matrix_power(R, -0.5)
    X_aligned = np.zeros_like(X)
    for i in range(n_trials):
        X_aligned[i] = np.dot(R_inv_sqrt, X[i])
    return X_aligned

def apply_ea_per_subject(X, metadata):
    subjects = np.unique(metadata['subject'])
    X_final = np.zeros_like(X)
    for sub in subjects:
        idx = np.where(metadata['subject'] == sub)[0]
        X_final[idx] = euclidean_alignment(X[idx])
    return X_final

def apply_maeeg_mask(x, mask_ratio=0.75):
    """Hybrid Masking: 75% Single Chunk for Reconstruction"""
    B, _, C, T = x.shape
    x_masked = x.clone()
    mask_len = int(mask_ratio * T)
    for i in range(B):
        start_idx = np.random.randint(0, T - mask_len)
        x_masked[i, :, :, start_idx : start_idx + mask_len] = 0
    return x_masked

# --- DATA LOADERS  ---
def get_bci_iv_2a(subject_ids):
    dataset = BNCI2014_001()
    dataset.subject_list = subject_ids
    paradigm = MotorImagery(n_classes=4, channels=None, resample=250, tmin=0, tmax=4.0 - 1/250)
    print(f"[BCI 2a] Loading Subjects: {subject_ids}")
    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=subject_ids)
    le = LabelEncoder()
    y = le.fit_transform(y)
    if '0train' in metadata['session'].unique():
        mask = metadata['session'] == '0train'
        X, y, metadata = X[mask], y[mask], metadata[mask]
    X = apply_ea_per_subject(X, metadata)
    return X, y, metadata

def get_bci_iv_2b(subject_ids):
    dataset = BNCI2014_004()
    dataset.subject_list = subject_ids
    paradigm = MotorImagery(n_classes=2, channels=['C3', 'Cz', 'C4'], resample=250, tmin=0, tmax=4.0 - 1/250)
    print(f"[BCI 2b] Loading Subjects: {subject_ids}")
    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=subject_ids)
    le = LabelEncoder()
    y = le.fit_transform(y)
    if '2train' in metadata['session'].unique():
        mask = metadata['session'] == '2train'
        X, y, metadata = X[mask], y[mask], metadata[mask]
    X = apply_ea_per_subject(X, metadata)
    return X, y, metadata

def get_hgd(subject_ids):
    dataset = Schirrmeister2017()
    dataset.subject_list = subject_ids
    target_channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'M1', 'M2', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'PO10', 'O1', 'Oz', 'O2', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FT9', 'FT10', 'FCz', 'CPz']
    paradigm = MotorImagery(n_classes=4, channels=target_channels, resample=250, tmin=0, tmax=4.0 - 1/250)
    print(f"[HGD] Loading Subjects: {subject_ids}")
    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=subject_ids)
    le = LabelEncoder()
    y = le.fit_transform(y)
    # HGD usually has '0' for training session in MOABB
    if '0' in metadata['session'].unique():
        mask = metadata['session'] == '0'
        X, y, metadata = X[mask], y[mask], metadata[mask]
    X = apply_ea_per_subject(X, metadata)
    return X, y, metadata

# =============================================================================
# 2. AUGMENTATIONS
# =============================================================================
def amplitude_addition(x, factor_range=(1, 4)):
    val = np.random.uniform(factor_range[0], factor_range[1])
    if np.random.random() > 0.5: val = -val
    return x + val

def cutout_and_resize_custom(x, segments_range=(4, 10)):
    L = x.shape[-1]
    n_seg = np.random.randint(segments_range[0], segments_range[1] + 1)
    seg_len = L // n_seg
    drop_idx = np.random.randint(0, n_seg)
    start = drop_idx * seg_len
    end = start + seg_len
    if isinstance(x, torch.Tensor):
        part1, part2 = x[..., :start], x[..., end:]
        new_x = torch.cat((part1, part2), dim=-1)
        if new_x.ndim == 2:
            new_x = F.interpolate(new_x.unsqueeze(0), size=L, mode='linear', align_corners=False).squeeze(0)
        elif new_x.ndim == 3:
            new_x = F.interpolate(new_x, size=L, mode='linear', align_corners=False)
    return new_x

def permutation_custom(x, segments_range=(4, 10)):
    m = np.random.randint(segments_range[0], segments_range[1] + 1)
    chunks = list(torch.chunk(x, m, dim=-1))
    np.random.shuffle(chunks)
    return torch.cat(chunks, dim=-1)

def crop_and_resize_wrapper(x, N_cut_ratio_range=(0.4, 0.8), segments=10):
    keep_ratio = np.random.uniform(N_cut_ratio_range[0], N_cut_ratio_range[1])
    n_keep = int(round(keep_ratio * segments))
    n_cut = max(1, min(segments - n_keep, segments - 1))
    return aug.crop_and_resize(x, segments=segments, N_cut=n_cut)

def get_sscl_augmenter():
    return aug.RandomAug(
        aug.StaticSingleAug(amplitude_addition, arguments={'factor_range': [1, 4]}),
        aug.DynamicSingleAug(aug.scaling, range_arg={'value': [2, 4]}, range_type={'value': False}), 
        aug.DynamicSingleAug(aug.warp_signal, 
                             range_arg={'segments': [4, 10], 'stretch_strength': [2, 4], 'squeeze_strength': [0.25, 0.5]}, 
                             range_type={'segments': True, 'stretch_strength': False, 'squeeze_strength': False},
                             discrete_arg={'batch_equal': [False]}),
        aug.DynamicSingleAug(aug.masking, 
                             range_arg={'masked_ratio': [0.1, 0.25]}, 
                             range_type={'masked_ratio': False}, 
                             discrete_arg={'mask_number': [1], 'batch_equal': [False]}),
        aug.StaticSingleAug(crop_and_resize_wrapper, arguments={'N_cut_ratio_range': [0.4, 0.8], 'segments': 10}),
        aug.StaticSingleAug(aug.flip_horizontal),
        aug.StaticSingleAug(permutation_custom, arguments={'segments_range': [4, 10]}),
        aug.StaticSingleAug(cutout_and_resize_custom, arguments={'segments_range': [4, 10]})
    )

class SSCL_Augmentation_Pipeline:
    def __init__(self): self.augmenter = get_sscl_augmenter()
    def __call__(self, x):
        if isinstance(x, np.ndarray): x = torch.from_numpy(x).float()
        added_dim = False
        if x.ndim == 2: x = x.unsqueeze(0); added_dim = True
        t_i = self.augmenter(x)
        t_j = self.augmenter(x)
        if added_dim: t_i = t_i.squeeze(0); t_j = t_j.squeeze(0)
        return t_i, t_j

class SSCLDataset(Dataset):
    def __init__(self, X, y): self.X = X; self.y = y; self.aug_pipeline = SSCL_Augmentation_Pipeline()
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        v1, v2 = self.aug_pipeline(self.X[idx])
        return v1, v2, torch.tensor(self.y[idx], dtype=torch.long)

class StandardDataset(Dataset):
    def __init__(self, X, y): self.X = X; self.y = y
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return torch.from_numpy(self.X[idx]).float(), torch.tensor(self.y[idx], dtype=torch.long)

# =============================================================================
# 3. MODEL: SSCL-CSD HYBRID
# =============================================================================
class SENet_Layer(nn.Module):
    def __init__(self, in_channels, r=2):
        super(SENet_Layer, self).__init__()
        red = int(in_channels // r)
        self.fc1 = nn.Linear(in_channels, red, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(red, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b, c, _, _ = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        y = self.sigmoid(self.fc2(self.relu(self.fc1(y)))).view(b, c, 1, 1)
        return x * y

class Decoder(nn.Module):
    def __init__(self, n_channels=22, original_len=1000):
        super(Decoder, self).__init__()
        self.original_len = original_len
        self.up1 = nn.Upsample(scale_factor=16, mode='linear', align_corners=False)
        self.conv1 = nn.Sequential(nn.Conv1d(16, 16, 3, padding=1), nn.BatchNorm1d(16), nn.ELU())
        self.up2 = nn.Upsample(scale_factor=8, mode='linear', align_corners=False)
        self.conv2 = nn.Sequential(nn.Conv1d(16, 16, 3, padding=1), nn.BatchNorm1d(16), nn.ELU())
        self.final_conv = nn.Conv1d(16, n_channels, kernel_size=1)

    def forward(self, z):
        x = self.conv1(self.up1(z))
        x = self.conv2(self.up2(x))
        if x.shape[2] != self.original_len:
            x = F.interpolate(x, size=self.original_len, mode='linear', align_corners=False)
        return self.final_conv(x)

class SSCL_CSD_Hybrid(nn.Module):
    def __init__(self, nb_classes=4, Chans=22, Samples=1000, dropoutRate=0.2):
        super(SSCL_CSD_Hybrid, self).__init__()
        self.temp_conv = nn.Sequential(nn.Conv2d(1, 8, (1, 128), padding='same', bias=False), SENet_Layer(8), nn.BatchNorm2d(8))
        self.spatial_conv = nn.Sequential(nn.Conv2d(8, 16, (Chans, 1), groups=8, bias=False), SENet_Layer(16), nn.BatchNorm2d(16), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(dropoutRate))
        self.feat_conv = nn.Sequential(nn.Conv2d(16, 16, (1, 32), padding='same', groups=16, bias=False), nn.Conv2d(16, 16, (1, 1), bias=False), SENet_Layer(16), nn.BatchNorm2d(16), nn.ELU(), nn.AvgPool2d((1, 16)), nn.Dropout(dropoutRate))
        
        with torch.no_grad(): 
            dummy = self.feat_conv(self.spatial_conv(self.temp_conv(torch.zeros(1, 1, Chans, Samples))))
            self.flat_dim = dummy.numel()
        
        self.decoder = Decoder(n_channels=Chans, original_len=Samples)
        self.classifier = nn.Sequential(nn.Linear(self.flat_dim, 32), nn.ELU(), nn.Dropout(dropoutRate), nn.Linear(32, nb_classes))

    def forward(self, x, mode='classify'):
        if x.ndim == 3: x = x.unsqueeze(1)
        x_enc = self.feat_conv(self.spatial_conv(self.temp_conv(x))) 
        z_flat = x_enc.view(x.size(0), -1)
        
        if mode == 'sscl': return z_flat
        elif mode == 'reconstruct': return self.decoder(x_enc.squeeze(2))
        return self.classifier(z_flat)

# =============================================================================
# 4. PLOTTING FUNCTIONS
# =============================================================================
def plot_tsne_numpy(X, y, subject_id, stage_name, save_dir):
    print(f"   Visualizing {stage_name}...")
    X_flat = X.reshape(X.shape[0], -1)
    tsne = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=42)
    X_emb = tsne.fit_transform(X_flat)
    class_names = ['Left', 'Right', 'Foot', 'Tongue'] if len(np.unique(y)) > 2 else ['Left', 'Right']
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_emb[:,0], y=X_emb[:,1], hue=[class_names[i] for i in y], palette='viridis', s=60, alpha=0.8)
    plt.title(f"t-SNE Subject {subject_id} - {stage_name}")
    plt.legend(title='Class')
    plt.savefig(f"{save_dir}/tsne_hybrid_sub{subject_id}_{stage_name.replace(' ', '_')}.png", bbox_inches='tight')
    plt.close()

def plot_tsne_model(model, loader, device, subject_id, stage_name, save_dir):
    print(f"   Visualizing {stage_name}...")
    model.eval()
    feats_list, labels_list = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            feats_list.append(model(inputs, mode='sscl').cpu().numpy())
            labels_list.append(labels.numpy())
    X_emb = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=42).fit_transform(np.concatenate(feats_list))
    y = np.concatenate(labels_list)
    class_names = ['Left', 'Right', 'Foot', 'Tongue'] if len(np.unique(y)) > 2 else ['Left', 'Right']
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_emb[:,0], y=X_emb[:,1], hue=[class_names[i] for i in y], palette='viridis', s=60, alpha=0.8)
    plt.title(f"t-SNE Subject {subject_id} - {stage_name}")
    plt.legend(title='Class')
    plt.savefig(f"{save_dir}/tsne_hybrid_sub{subject_id}_{stage_name.replace(' ', '_')}.png", bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, subject_id, acc, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    class_names = ['Left', 'Right', 'Foot', 'Tongue'] if len(np.unique(y_true)) > 2 else ['Left', 'Right']
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f"CM Subject {subject_id} (Acc: {acc:.2f}%)")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f"{save_dir}/cm_hybrid_sub{subject_id}.png", bbox_inches='tight')
    plt.close()

# =============================================================================
# 5. MAIN EXECUTION LOOP
# =============================================================================
if __name__ == "__main__":
    seed_everything(42)
    
    # --- CONFIG ---
    # OPTIONS: '2a', '2b', 'hgd'
    DATASET_NAME = '2a' 
    
    # Set SUBJECTS based on Dataset
    if DATASET_NAME == '2a' or DATASET_NAME == '2b':
        SUBJECTS = range(1, 10)
    elif DATASET_NAME == 'hgd':
        SUBJECTS = range(1, 15)
    
    EPOCHS_SSL = 100
    EPOCHS_FT = 300
    BATCH_SIZE_SSL = 64
    BATCH_SIZE_FT = 64
    
    # OPTIMIZER SETTINGS (Restored)
    LR = 0.001
    WEIGHT_DECAY = 0.001
    MOMENTUM_BETA1 = 0.9
    
    LAMBDA_RECON = 10.0 
    
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    elif torch.backends.mps.is_available():
        DEVICE = 'mps' 
    else:
        DEVICE = 'cpu'
    print(f"Starting SSCL-CSD HYBRID Experiment on {DEVICE} | Dataset: {DATASET_NAME}")
    
    save_dir = f"results_hybrid_{DATASET_NAME}"
    model_dir = f"models_hybrid_{DATASET_NAME}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    results = []

    for subject in SUBJECTS:
        print(f"\n{'='*50}")
        print(f"PROCESSING SUBJECT {subject} (HYBRID - {DATASET_NAME})")
        print(f"{'='*50}")
        
        # --- DATA LOADING ---
        all_subs = list(SUBJECTS)
        train_subs = [s for s in all_subs if s != subject]
        
        if DATASET_NAME == '2a':
            X_train, y_train, _ = get_bci_iv_2a(train_subs)
            X_test, y_test, _ = get_bci_iv_2a([subject])
        elif DATASET_NAME == '2b':
            X_train, y_train, _ = get_bci_iv_2b(train_subs)
            X_test, y_test, _ = get_bci_iv_2b([subject])
        elif DATASET_NAME == 'hgd':
            X_train, y_train, _ = get_hgd(train_subs)
            X_test, y_test, _ = get_hgd([subject])
        
        # Calculate Input Params dynamically
        n_classes = len(np.unique(y_train))
        n_chans = X_train.shape[1]
        n_samples = X_train.shape[2]
        print(f"   [INFO] Channels: {n_chans}, Samples: {n_samples}, Classes: {n_classes}")

        plot_tsne_numpy(X_test, y_test, subject, "1. After EA", save_dir)
        
        X_train_ft, X_val_ft, y_train_ft, y_val_ft = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        sscl_loader = DataLoader(SSCLDataset(X_train, y_train), batch_size=BATCH_SIZE_SSL, shuffle=True, drop_last=True)
        raw_loader = DataLoader(StandardDataset(X_train, y_train), batch_size=BATCH_SIZE_SSL, shuffle=True, drop_last=True)
        ft_train_loader = DataLoader(StandardDataset(X_train_ft, y_train_ft), batch_size=BATCH_SIZE_FT, shuffle=True)
        ft_val_loader = DataLoader(StandardDataset(X_val_ft, y_val_ft), batch_size=BATCH_SIZE_FT, shuffle=False)
        test_loader = DataLoader(StandardDataset(X_test, y_test), batch_size=BATCH_SIZE_FT, shuffle=False)
        
        # Init Model with dynamic parameters
        model = SSCL_CSD_Hybrid(nb_classes=n_classes, Chans=n_chans, Samples=n_samples).to(DEVICE)
        
        # --- STAGE 1: HYBRID PRE-TRAINING ---
        print(f"  [Stage 1] Pre-training...")
        # ADDED: Betas and Weight Decay
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(MOMENTUM_BETA1, 0.999))
        mse_crit = nn.MSELoss()
        
        model.train()
        for epoch in range(EPOCHS_SSL):
            total_loss = 0
            for (v1, v2, _), (raw_x, _) in zip(sscl_loader, raw_loader):
                v1, v2, raw_x = v1.to(DEVICE), v2.to(DEVICE), raw_x.to(DEVICE)
                if raw_x.ndim == 3: raw_x = raw_x.unsqueeze(1)
                
                optimizer.zero_grad()
                
                # A. SSCL
                z1 = model(v1, mode='sscl')
                z2 = model(v2, mode='sscl')
                loss_sscl = simclr_loss(torch.cat([z1, z2], dim=0), temperature=0.5)
                
                # B. Reconstruction
                masked_input = apply_maeeg_mask(raw_x, mask_ratio=0.75)
                recon_out = model(masked_input, mode='reconstruct')
                loss_recon = mse_crit(recon_out, raw_x.squeeze(1))
                
                loss = loss_sscl + (LAMBDA_RECON * loss_recon)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            if (epoch+1) % 20 == 0:
                print(f"    Epoch {epoch+1}: Hybrid Loss = {total_loss/len(sscl_loader):.4f}")
        
        torch.save(model.state_dict(), f"{model_dir}/encoder_hybrid_sub{subject}.pth")
        
        # --- STAGE 2: FINE-TUNING ---
        print(f"  [Stage 2] Fine-tuning...")
        # ADDED: Betas and Weight Decay
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(MOMENTUM_BETA1, 0.999))
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0.0
        best_model_path = f"{model_dir}/best_hybrid_sub{subject}.pth"
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(1, EPOCHS_FT + 1):
            model.train()
            l_sum, corr, tot = 0, 0, 0
            for x, y in ft_train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                out = model(x, mode='classify')
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                l_sum += loss.item()
                _, p = torch.max(out, 1)
                corr += (p==y).sum().item()
                tot += y.size(0)
            
            history['train_loss'].append(l_sum/len(ft_train_loader))
            history['train_acc'].append(corr/tot)
            
            # Validation (Every Epoch)
            model.eval()
            v_corr, v_tot, v_loss = 0, 0, 0
            with torch.no_grad():
                for x, y in ft_val_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    out = model(x, mode='classify')
                    v_loss += criterion(out, y).item()
                    _, p = torch.max(out, 1)
                    v_corr += (p==y).sum().item()
                    v_tot += y.size(0)
            
            val_acc = v_corr/v_tot
            history['val_loss'].append(v_loss/len(ft_val_loader))
            history['val_acc'].append(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
            
            if epoch % 10 == 0:
                print(f"    Epoch {epoch}: Train Acc {corr/tot:.3f} | Val Acc {val_acc:.3f}")

        # --- TEST & PLOTS ---
        print(f"  [Test] Testing Best Model...")
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        preds, all_lbls = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(DEVICE)
                _, p = torch.max(model(x, mode='classify'), 1)
                preds.extend(p.cpu().numpy())
                all_lbls.extend(y.numpy())
        
        acc = accuracy_score(all_lbls, preds)
        kap = cohen_kappa_score(all_lbls, preds)
        print(f"  Subject {subject}: Acc {acc*100:.2f}% | Kappa {kap:.4f}")
        results.append({'Subject': subject, 'Accuracy': acc, 'Kappa': kap})
        
        plot_confusion_matrix(all_lbls, preds, subject, acc*100, save_dir)
        plot_tsne_model(model, test_loader, DEVICE, subject, "2. After SSCL-CSD", save_dir)
        
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train')
        plt.plot(history['val_loss'], label='Val')
        plt.title('Loss'); plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train')
        plt.plot(history['val_acc'], label='Val')
        plt.title('Accuracy'); plt.legend()
        plt.savefig(f"{save_dir}/curve_hybrid_sub{subject}.png"); plt.close()

    pd.DataFrame(results).to_csv(f"final_results_hybrid_{DATASET_NAME}.csv", index=False)
