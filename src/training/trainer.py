import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 15,
    alpha_cap: float = 10.0,
    save_path: Optional[str] = None
) -> Tuple[nn.Module, Dict[str, list], float]:
    """
    Train temporal deep learning model with multi-task loss for RUL and SOH capacity.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    criterion_rul = nn.MSELoss()
    criterion_cap = nn.MSELoss()
    
    early_stopping = EarlyStopping(patience=patience)
    history = {'train_loss': [], 'val_loss': [], 'val_mae_rul': []}
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for X_b, y_rul_b, y_cap_b in train_loader:
            X_b = X_b.to(device)
            y_rul_b = y_rul_b.to(device)
            y_cap_b = y_cap_b.to(device)
            
            optimizer.zero_grad()
            pred_rul, pred_cap = model(X_b)
            
            loss_rul = criterion_rul(pred_rul, y_rul_b)
            loss_cap = criterion_cap(pred_cap, y_cap_b)
            loss = loss_rul + alpha_cap * loss_cap
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            
        model.eval()
        val_losses = []
        val_rul_errors = []
        with torch.no_grad():
            for X_v, y_rul_v, y_cap_v in val_loader:
                X_v = X_v.to(device)
                y_rul_v = y_rul_v.to(device)
                y_cap_v = y_cap_v.to(device)
                
                pred_rul_v, pred_cap_v = model(X_v)
                loss_rul_v = criterion_rul(pred_rul_v, y_rul_v)
                loss_cap_v = criterion_cap(pred_cap_v, y_cap_v)
                val_loss = loss_rul_v + alpha_cap * loss_cap_v
                
                val_losses.append(val_loss.item())
                val_rul_errors.extend(torch.abs(pred_rul_v - y_rul_v).cpu().numpy().flatten())
                
        epoch_train_loss = float(np.mean(train_losses))
        epoch_val_loss = float(np.mean(val_losses))
        epoch_val_mae = float(np.mean(val_rul_errors))
        
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['val_mae_rul'].append(epoch_val_mae)
        
        scheduler.step(epoch_val_loss)
        early_stopping(epoch_val_loss, model)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f'Epoch {epoch+1:03d}/{epochs:03d} - Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val RUL MAE: {epoch_val_mae:.2f} cycles')
            
        if early_stopping.early_stop:
            logger.info(f'Early stopping triggered at epoch {epoch+1}')
            break
            
    total_training_time = time.time() - start_time
    
    # Restore best weights
    if early_stopping.best_weights is not None:
        model.load_state_dict(early_stopping.best_weights)
        
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        logger.info(f'Saved best model checkpoint to {save_path}')
        
    return model, history, total_training_time
