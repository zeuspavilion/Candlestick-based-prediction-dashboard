import time
import logging
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support

from MarketPulse.config import (
    IMAGE_SIZE,
    LABELS,
    MANIFEST_DIR,
    MODEL_DIR,
    RANDOM_SEED,
    CLASS_TO_IDX,
)
from MarketPulse.ml.models import MODEL_BUILDERS
from MarketPulse.ml.dataset import CandlestickManifestDataset
from MarketPulse.database.connection import SessionLocal
from MarketPulse.database.models import ModelRun, PipelineLog

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("marketpulse.ml.training")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = len(LABELS)

# Setup Transforms (ImageNet standard for ViT/ResNet transfer learning)
imagenet_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Default Hyperparameters
TRAINING_CONFIG = {
    "batch_size": 32,
    "num_epochs": 10,
    "num_workers": 0,  # 0 is safest on Windows to prevent multi-processing leaks
    "early_stopping_patience": 3,
    "early_stopping_min_delta": 1e-4,
    "gradient_clip_norm": 1.0,
}

MODEL_HYPERPARAMS = {
    "vit_b_16": {
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "label_smoothing": 0.03,
        "freeze_pretrained_backbone": True,
    },
    "resnet18": {
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "label_smoothing": 0.03,
        "freeze_pretrained_backbone": False,
    },
    "custom_cnn": {
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "label_smoothing": 0.02,
        "freeze_pretrained_backbone": False,
    }
}

class ModelTrainer:
    def __init__(self, model_name: str, config: dict = None, db_session=None):
        self.model_name = model_name
        self.config = config or MODEL_HYPERPARAMS.get(model_name, MODEL_HYPERPARAMS["custom_cnn"])
        self.db = db_session or SessionLocal()
        
    def log_event(self, level: str, message: str, details: str = None):
        log = PipelineLog(module=f"ML.Train.{self.model_name}", level=level, message=message, details=details)
        self.db.add(log)
        self.db.commit()
        logger.info(message)

    def load_dataloaders(self) -> tuple:
        train_ds = CandlestickManifestDataset(MANIFEST_DIR / "train.csv", transform=imagenet_transform)
        val_ds = CandlestickManifestDataset(MANIFEST_DIR / "validation.csv", transform=imagenet_transform)
        test_ds = CandlestickManifestDataset(MANIFEST_DIR / "test.csv", transform=imagenet_transform)
        
        train_loader = DataLoader(
            train_ds,
            batch_size=TRAINING_CONFIG["batch_size"],
            shuffle=True,
            num_workers=TRAINING_CONFIG["num_workers"],
            pin_memory=torch.cuda.is_available()
        )
        
        val_loader = DataLoader(
            val_ds,
            batch_size=TRAINING_CONFIG["batch_size"],
            shuffle=False,
            num_workers=TRAINING_CONFIG["num_workers"],
            pin_memory=torch.cuda.is_available()
        )
        
        test_loader = DataLoader(
            test_ds,
            batch_size=TRAINING_CONFIG["batch_size"],
            shuffle=False,
            num_workers=TRAINING_CONFIG["num_workers"],
            pin_memory=torch.cuda.is_available()
        )
        
        return train_loader, val_loader, test_loader

    def compute_class_weights(self) -> torch.Tensor:
        train_manifest = pd.read_csv(MANIFEST_DIR / "train.csv")
        counts = train_manifest["label_id"].value_counts().sort_index()
        weights = len(train_manifest) / (NUM_CLASSES * counts)
        # Reindex to match target labels index size
        weights_arr = weights.reindex(range(NUM_CLASSES)).values
        return torch.tensor(weights_arr, dtype=torch.float32)

    def train(self, limit_epochs: int = None) -> tuple:
        epochs = limit_epochs or TRAINING_CONFIG["num_epochs"]
        self.log_event("INFO", f"Initializing model architecture: {self.model_name}")
        
        model_builder = MODEL_BUILDERS.get(self.model_name)
        if not model_builder:
            raise ValueError(f"Unknown model name: {self.model_name}")
            
        model, pretrained = model_builder(
            num_classes=NUM_CLASSES,
            freeze_pretrained_backbone=self.config.get("freeze_pretrained_backbone", False)
        )
        model = model.to(DEVICE)
        
        # Load dataloaders
        train_loader, val_loader, test_loader = self.load_dataloaders()
        
        # Criterion, Optimizer, Scheduler
        class_w = self.compute_class_weights().to(DEVICE)
        criterion = nn.CrossEntropyLoss(
            weight=class_w,
            label_smoothing=self.config.get("label_smoothing", 0.0)
        )
        
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(
            trainable_params,
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"]
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=self.config["learning_rate"] * 0.05
        )
        
        best_state = deepcopy(model.state_dict())
        best_val_macro_f1 = -1.0
        patience_counter = 0
        start_time = time.time()
        
        self.log_event("INFO", f"Starting training: {epochs} epochs, Device: {DEVICE}")
        
        for epoch in range(1, epochs + 1):
            # Train Phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for imgs, lbls in train_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                optimizer.zero_grad(set_to_none=True)
                logits = model(imgs)
                loss = criterion(logits, lbls)
                loss.backward()
                
                if TRAINING_CONFIG["gradient_clip_norm"] is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG["gradient_clip_norm"])
                    
                optimizer.step()
                
                train_loss += loss.item() * imgs.size(0)
                preds = logits.argmax(dim=1)
                train_correct += (preds == lbls).sum().item()
                train_total += imgs.size(0)
                
            scheduler.step()
            epoch_train_loss = train_loss / train_total
            epoch_train_acc = train_correct / train_total
            
            # Validation Phase
            model.eval()
            val_loss = 0.0
            val_total = 0
            val_targets = []
            val_preds = []
            
            with torch.no_grad():
                for imgs, lbls in val_loader:
                    imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                    logits = model(imgs)
                    loss = criterion(logits, lbls)
                    
                    val_loss += loss.item() * imgs.size(0)
                    preds = logits.argmax(dim=1)
                    val_total += imgs.size(0)
                    
                    val_targets.extend(lbls.cpu().numpy())
                    val_preds.extend(preds.cpu().numpy())
            
            epoch_val_loss = val_loss / val_total
            epoch_val_acc = accuracy_score(val_targets, val_preds)
            _, _, val_macro_f1, _ = precision_recall_fscore_support(
                val_targets, val_preds, average="macro", zero_division=0
            )
            
            self.log_event(
                "INFO",
                f"Epoch {epoch:02d}/{epochs:02d}: Train Loss={epoch_train_loss:.4f}, Train Acc={epoch_train_acc:.4f} | "
                f"Val Loss={epoch_val_loss:.4f}, Val Acc={epoch_val_acc:.4f}, Val Macro F1={val_macro_f1:.4f}"
            )
            
            # Early stopping check
            if val_macro_f1 > best_val_macro_f1 + TRAINING_CONFIG["early_stopping_min_delta"]:
                best_val_macro_f1 = val_macro_f1
                best_state = deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= TRAINING_CONFIG["early_stopping_patience"]:
                self.log_event("INFO", f"Early stopping triggered at epoch {epoch}")
                break
                
        # Load best model
        model.load_state_dict(best_state)
        
        # Test Phase Evaluation
        model.eval()
        test_loss = 0.0
        test_total = 0
        test_targets = []
        test_preds = []
        
        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                logits = model(imgs)
                loss = criterion(logits, lbls)
                
                test_loss += loss.item() * imgs.size(0)
                preds = logits.argmax(dim=1)
                test_total += imgs.size(0)
                
                test_targets.extend(lbls.cpu().numpy())
                test_preds.extend(preds.cpu().numpy())
                
        epoch_test_loss = test_loss / test_total
        test_acc = accuracy_score(test_targets, test_preds)
        test_bal_acc = balanced_accuracy_score(test_targets, test_preds)
        test_prec, test_rec, test_weighted_f1, _ = precision_recall_fscore_support(
            test_targets, test_preds, average="weighted", zero_division=0
        )
        _, _, test_macro_f1, _ = precision_recall_fscore_support(
            test_targets, test_preds, average="macro", zero_division=0
        )
        
        elapsed_minutes = (time.time() - start_time) / 60
        self.log_event(
            "INFO",
            f"Training finished in {elapsed_minutes:.2f} mins. "
            f"Test Accuracy: {test_acc:.4f}, Test Macro F1: {test_macro_f1:.4f}"
        )
        
        # Save model checkpoint
        checkpoint_path = MODEL_DIR / f"{self.model_name}_{self.config.get('name', 'head')}.pth"
        torch.save(
            {
                "model_name": self.model_name,
                "config": self.config,
                "class_to_idx": CLASS_TO_IDX,
                "state_dict": model.state_dict(),
                "test_metrics": {
                    "accuracy": test_acc,
                    "macro_f1": test_macro_f1,
                    "weighted_f1": test_weighted_f1,
                    "loss": epoch_test_loss,
                }
            },
            checkpoint_path
        )
        
        # Record ModelRun to database
        model_run = ModelRun(
            model_name=self.model_name,
            config_name=self.config.get("name", "head"),
            test_accuracy=float(test_acc),
            test_macro_f1=float(test_macro_f1),
            test_weighted_f1=float(test_weighted_f1),
            test_loss=float(epoch_test_loss),
            status="SUCCESS",
            checkpoint_path=str(checkpoint_path.relative_to(PROJECT_ROOT))
        )
        self.db.add(model_run)
        self.db.commit()
        
        return model, checkpoint_path

    def close(self):
        self.db.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a Candlestick Prediction Dashboard")
    parser.add_argument("--model", type=str, default="custom_cnn", choices=["custom_cnn", "resnet18", "vit_b_16"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=3, help="Max epochs to train")
    args = parser.parse_args()

    trainer = ModelTrainer(model_name=args.model)
    try:
        trainer.train(limit_epochs=args.epochs)
    finally:
        trainer.close()
