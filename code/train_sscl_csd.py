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
from scipy.linalg import fractional_matrix_power
import selfeeg.augmentation as aug
from selfeeg.losses import simclr_loss
import logging
import time
import random
import os

# =============================================================================
# 0. SETUP & REPRODUCIBILITY
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
    torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random Seed fixed to {seed}")

# =============================================================================
# 1. DATA PROCESSING (EUCLIDEAN ALIGNMENT & LOADERS)
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
    
    # Label Encoding
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
    
    # Label Encoding
    le = LabelEncoder()
    y = le.fit_transform(y)

    target_session = '0' 
    if target_session in metadata['session'].unique():
        mask = metadata['session'] == target_session
        X, y, metadata = X[mask], y[mask], metadata[mask]
    X = apply_ea_per_subject(X, metadata)
    return X, y, metadata

# =============================================================================
# 2. AUGMENTATIONS & DATASETS
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
        
        aug.DynamicSingleAug(aug.scaling, 
                             range_arg={'value': [2, 4]}, 
                             range_type={'value': False}), 
                             
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
# 3. MODEL (SSCL_CSD)
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

class SSCL_CSD(nn.Module):
    def __init__(self, nb_classes=4, Chans=22, Samples=1000, dropoutRate=0.2):
        super(SSCL_CSD, self).__init__()
        self.temp_conv = nn.Sequential(nn.Conv2d(1, 8, (1, 128), padding='same', bias=False), SENet_Layer(8), nn.BatchNorm2d(8))
        self.spatial_conv = nn.Sequential(nn.Conv2d(8, 16, (Chans, 1), groups=8, bias=False), SENet_Layer(16), nn.BatchNorm2d(16), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(dropoutRate))
        self.feat_conv = nn.Sequential(nn.Conv2d(16, 16, (1, 32), padding='same', groups=16, bias=False), nn.Conv2d(16, 16, (1, 1), bias=False), SENet_Layer(16), nn.BatchNorm2d(16), nn.ELU(), nn.AvgPool2d((1, 16)), nn.Dropout(dropoutRate))
        with torch.no_grad(): self.flat_dim = self.feat_conv(self.spatial_conv(self.temp_conv(torch.zeros(1, 1, Chans, Samples)))).numel()
        print(f"[{Chans}Ch, {Samples}Hz] -> Feature Dimension (z): {self.flat_dim}")
        self.classifier = nn.Sequential(nn.Linear(self.flat_dim, 32), nn.ELU(), nn.Dropout(dropoutRate), nn.Linear(32, nb_classes))

    def forward(self, x, mode='classify'):
        if x.ndim == 3: x = x.unsqueeze(1)
        feats = self.feat_conv(self.spatial_conv(self.temp_conv(x))).view(x.size(0), -1)
        if mode == 'pretrain': return feats
        return self.classifier(feats)

# =============================================================================
# 4. PLOTTING FUNCTIONS
# =============================================================================
def plot_tsne_numpy(X, y, subject_id, stage_name, save_dir="results"):
    print(f"   Visualizing {stage_name}...")
    X_flat = X.reshape(X.shape[0], -1)
    tsne = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=42)
    X_emb = tsne.fit_transform(X_flat)
    class_names = ['Left', 'Right', 'Foot', 'Tongue'] if len(np.unique(y)) > 2 else ['Left', 'Right']
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_emb[:,0], y=X_emb[:,1], hue=[class_names[i] for i in y], palette='viridis', s=60, alpha=0.8)
    plt.title(f"t-SNE Subject {subject_id} - {stage_name}")
    plt.legend(title='Class')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/tsne_sub{subject_id}_{stage_name.replace(' ', '_')}.png", bbox_inches='tight')
    plt.close()

def plot_tsne_model(model, loader, device, subject_id, stage_name, save_dir="results"):
    print(f"   Visualizing {stage_name}")
    model.eval()
    feats_list, labels_list = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            feats_list.append(model(inputs, mode='pretrain').cpu().numpy())
            labels_list.append(labels.numpy())
    X_emb = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=42).fit_transform(np.concatenate(feats_list))
    y = np.concatenate(labels_list)
    class_names = ['Left', 'Right', 'Foot', 'Tongue'] if len(np.unique(y)) > 2 else ['Left', 'Right']
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_emb[:,0], y=X_emb[:,1], hue=[class_names[i] for i in y], palette='viridis', s=60, alpha=0.8)
    plt.title(f"t-SNE Subject {subject_id} - {stage_name}")
    plt.legend(title='Class')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/tsne_sub{subject_id}_{stage_name.replace(' ', '_')}.png", bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, subject_id, acc, save_dir="results"):
    cm = confusion_matrix(y_true, y_pred)
    class_names = ['Left', 'Right', 'Foot', 'Tongue'] if len(np.unique(y_true)) > 2 else ['Left', 'Right']
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f"CM Subject {subject_id} (Acc: {acc:.2f}%)")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/cm_sub{subject_id}.png", bbox_inches='tight')
    plt.close()

# =============================================================================
# 5. MAIN EXECUTION LOOP
# =============================================================================
if __name__ == "__main__":
    seed_everything(42)
    
    # --- CONFIG ---
    DATASET_NAME = '2a'     # OPTIONS: '2a', '2b', 'hgd'
    SUBJECTS = range(1, 10) 
    EPOCHS_SSL = 100    
    EPOCHS_FT = 300 
    BATCH_SIZE_SSL = 64        
    BATCH_SIZE_FT = 64         
    LR = 0.001              
    WEIGHT_DECAY = 0.001    
    MOMENTUM_BETA1 = 0.9    

    # --- DEVICE SELECTION ---
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    elif torch.backends.mps.is_available():
        DEVICE = 'mps' 
    else:
        DEVICE = 'cpu'
        
    print(f"Starting Experiment on {DEVICE} | Dataset: {DATASET_NAME}")
    
    # Create directory for saving models
    os.makedirs("saved_models", exist_ok=True)
    
    results = []

    for subject in SUBJECTS:
        print(f"\n{'='*60}\nPROCESSING SUBJECT {subject}\n{'='*60}")
        
        # 1. LOAD DATA
        all_subs = list(SUBJECTS)
        train_subs = [s for s in all_subs if s != subject]
        
        if DATASET_NAME == '2a': load_fn = get_bci_iv_2a
        elif DATASET_NAME == '2b': load_fn = get_bci_iv_2b
        elif DATASET_NAME == 'hgd': load_fn = get_hgd
        else: raise ValueError("Invalid Dataset Name")
            
        print(f"   Loading Train: {train_subs} | Test: {subject}")
        X_train, y_train, _ = load_fn(train_subs)
        X_test, y_test, _ = load_fn([subject])
        
        # VISUALIZATION 1: Baseline t-SNE
        plot_tsne_numpy(X_test, y_test, subject, stage_name="1. After EA ")
        
        ssl_loader = DataLoader(SSCLDataset(X_train, y_train), batch_size=BATCH_SIZE_SSL, shuffle=True, drop_last=True)
        ft_loader = DataLoader(StandardDataset(X_train, y_train), batch_size=BATCH_SIZE_FT, shuffle=True)
        test_loader = DataLoader(StandardDataset(X_test, y_test), batch_size=BATCH_SIZE_FT, shuffle=False)
        
        # 2. INIT MODEL
        n_classes = len(np.unique(y_train))
        n_chans = X_train.shape[1]
        model = SSCL_CSD(nb_classes=n_classes, Chans=n_chans, Samples=1000).to(DEVICE)
        
        # 3. STAGE 1: PRE-TRAINING (Self-Supervised)
        print(f"   [Stage 1] Pre-training ({EPOCHS_SSL} epochs)")
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(MOMENTUM_BETA1, 0.999))
        model.train()
        for epoch in range(EPOCHS_SSL):
            total_loss = 0
            for t1, t2, _ in ssl_loader:
                t1, t2 = t1.to(DEVICE), t2.to(DEVICE)
                z1 = model(t1, mode='pretrain') 
                z2 = model(t2, mode='pretrain')
                projections = torch.cat([z1, z2], dim=0)
                optimizer.zero_grad()
                # Ignore labels here, only use augmented views
                loss = simclr_loss(projections, temperature=0.5)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch+1) % 20 == 0: print(f"Epoch {epoch+1}: Loss = {total_loss/len(ssl_loader):.4f}")
        
        # --- SAVE PRE-TRAINED MODEL ---
        save_path = f"saved_models/sscl_encoder_sub{subject}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Encoder saved to {save_path}")
                
        # 4. STAGE 2: FINE-TUNING
        print(f"   [Stage 2] Fine-tuning ({EPOCHS_FT} epochs)")
        
        # Optional: Load the weights back 
        model.load_state_dict(torch.load(save_path, map_location=DEVICE))
        
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(MOMENTUM_BETA1, 0.999))
        criterion = nn.CrossEntropyLoss()
        for epoch in range(EPOCHS_FT):
            model.train()
            for inputs, labels in ft_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(inputs, mode='classify'), labels)
                loss.backward()
                optimizer.step()
                
        # 5. EVALUATION
        model.eval()
        preds, labels_all = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(DEVICE)
                _, predicted = torch.max(model(inputs, mode='classify'), 1)
                preds.extend(predicted.cpu().numpy())
                labels_all.extend(labels.numpy())
                
        acc = accuracy_score(labels_all, preds)
        kappa = cohen_kappa_score(labels_all, preds)
        print(f"Subject {subject}: Acc = {acc*100:.2f}% | Kappa = {kappa:.4f}")
        results.append({'Subject': subject, 'Accuracy': acc, 'Kappa': kappa})
        
        plot_confusion_matrix(labels_all, preds, subject, acc*100)
        plot_tsne_model(model, test_loader, DEVICE, subject, stage_name="2. After SSCL-CSD")

    # RESULTS
    print("\n" + "="*60)
    df = pd.DataFrame(results)
    print(df)
    print(f"Avg Acc: {df['Accuracy'].mean()*100:.2f}%")
    df.to_csv("final_results_ssl_csd.csv", index=False)
