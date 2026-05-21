"""
HOPPER Trainer.
Handles training loop, evaluation, checkpointing,
temperature annealing, and metric logging.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from hopper.training.loss import HOPPERLoss
from hopper.evaluation.evaluator import HOPPEREvaluator
import json


class HOPPERTrainer:

    def __init__(self, model, config: dict, device: str = "cuda"):
        self.model   = model.to(device)
        self.config  = config
        self.device  = device

        self.loss_fn = HOPPERLoss(
            lambda_trans=config.get("lambda_trans", 0.1)
        )

        # Optimizer — AdamW with weight decay
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr           = config.get("learning_rate", 2e-5),
            weight_decay = config.get("weight_decay", 0.01),
        )

        # LR scheduler
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor = 1.0,
            end_factor   = 0.0,
            total_iters  = config.get("total_steps", 10000),
        )

        # FP16 scaler for P100
        self.scaler = GradScaler()

        self.evaluator   = HOPPEREvaluator(device)
        self.global_step = 0
        self.best_f1     = 0.0

        # Temperature annealing schedule
        self.tau_start = config.get("tau_start", 1.0)
        self.tau_end   = config.get("tau_end",   0.3)
        self.tau_steps = config.get("tau_steps", 5000)

        os.makedirs(config.get("output_dir", "outputs"), exist_ok=True)

    def _get_tau(self) -> float:
        """Linear temperature annealing."""
        progress = min(self.global_step / self.tau_steps, 1.0)
        return self.tau_start - progress * (self.tau_start - self.tau_end)

    def train_epoch(self, dataloader: DataLoader) -> dict:
        self.model.train()
        total_loss = total_span = total_trans = 0.0
        steps = 0

        pbar = tqdm(dataloader, desc="Training")
        for batch in pbar:
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            start_targets  = batch["start_positions"].to(self.device)
            end_targets    = batch["end_positions"].to(self.device)

            # Anneal temperature
            tau = self._get_tau()
            if hasattr(self.model, "set_temperature"):
                self.model.set_temperature(tau)

            self.optimizer.zero_grad()

            with autocast():
                start_logits, end_logits, hop_batches = self.model(
                    input_ids, attention_mask
                )
                losses = self.loss_fn(
                    start_logits, end_logits,
                    start_targets, end_targets,
                    hop_batches,
                )

            self.scaler.scale(losses["loss"]).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss  += losses["loss"].item()
            total_span  += losses["span_loss"].item()
            total_trans += losses["trans_loss"].item()
            steps       += 1
            self.global_step += 1

            pbar.set_postfix({
                "loss":  f"{losses['loss'].item():.4f}",
                "span":  f"{losses['span_loss'].item():.4f}",
                "trans": f"{losses['trans_loss'].item():.4f}",
                "tau":   f"{tau:.3f}",
            })

        return {
            "train_loss":       total_loss  / steps,
            "train_span_loss":  total_span  / steps,
            "train_trans_loss": total_trans / steps,
        }

    def evaluate(self, dataloader: DataLoader, dataset_name: str) -> dict:
        return self.evaluator.evaluate(self.model, dataloader, dataset_name)

    def save_checkpoint(self, path: str, metrics: dict):
        torch.save({
            "model_state":     self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "global_step":     self.global_step,
            "metrics":         metrics,
            "config":          self.config,
        }, path)
        print(f"Checkpoint saved: {path}")

    def train(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        dataset_name: str,
    ):
        num_epochs  = self.config.get("num_epochs", 3)
        output_dir  = self.config.get("output_dir", "outputs")
        all_metrics = []

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")

            train_metrics = self.train_epoch(train_loader)
            val_metrics   = self.evaluate(val_loader, dataset_name)

            metrics = {**train_metrics, **val_metrics, "epoch": epoch+1}
            all_metrics.append(metrics)

            print(f"  EM={val_metrics.get('em', 0):.4f}  "
                  f"F1={val_metrics.get('f1', 0):.4f}  "
                  f"SuppF1={val_metrics.get('supporting_fact_f1', 0):.4f}")

            # Save best checkpoint
            if val_metrics.get("f1", 0) > self.best_f1:
                self.best_f1 = val_metrics.get("f1", 0)
                self.save_checkpoint(
                    os.path.join(output_dir, "best_model.pt"), metrics
                )

        # Save all metrics
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(all_metrics, f, indent=2)

        return all_metrics