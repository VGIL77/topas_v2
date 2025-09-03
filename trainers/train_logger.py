"""
TOPAS Training Logger - Professional structured logging for ARC Prize runs
Provides both human-readable console output and machine-readable JSONL logs
"""

import json
import os
import datetime
from typing import Dict, Optional, Any

class TrainLogger:
    """
    Professional training logger for TOPAS Parent Brain
    
    Features:
    - Structured JSONL logging for analysis
    - Beautiful console output with all key metrics
    - Batch-by-batch tracking of losses, scheduler, dream, and memory
    """
    
    def __init__(self, log_path="logs/topas_train.jsonl", verbose=True):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_path = log_path
        self.verbose = verbose
        
        if self.verbose:
            print(f"🚀 TrainLogger initialized: {log_path}")

    def log_batch(self, batch_idx: int, losses: Dict[str, float], 
                  scheduler_info: Optional[Dict[str, Any]] = None,
                  dream_info: Optional[Dict[str, Any]] = None,
                  relmem_info: Optional[Dict[str, Any]] = None):
        """
        Log comprehensive batch information
        
        Args:
            batch_idx: Current batch number
            losses: dict -> {"total": x, "painter": y, "dsl": z, "ebr": q, "aux": w}
            scheduler_info: dict -> {"task_id":..., "bucket":..., "difficulty":..., "ucb":...}
            dream_info: dict -> {"ripple":..., "valence":..., "arousal":..., "motifs":...}
            relmem_info: dict -> {"inverse_loss":..., "hebbian_applied":..., "wta_applied":...}
        """
        record = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "batch": batch_idx,
            "losses": losses,
            "scheduler": scheduler_info or {},
            "dream": dream_info or {},
            "relmem": relmem_info or {}
        }

        # Write JSONL for analysis
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        # Pretty console output
        if self.verbose:
            msg = f"[Batch {batch_idx}] Total={losses.get('total',0):.3f}"
            if "painter" in losses: 
                msg += f" | Painter={losses['painter']:.3f}"
            if "dsl" in losses: 
                msg += f" DSL={losses['dsl']:.3f}"
            if "ebr" in losses: 
                msg += f" EBR={losses['ebr']:.3f}"
            if "aux" in losses: 
                msg += f" Aux={losses['aux']:.3f}"

            if scheduler_info:
                msg += f" | Task={scheduler_info.get('task_id','?')} bucket={scheduler_info.get('bucket','?')}"
                msg += f" diff={scheduler_info.get('difficulty','?'):.2f} ucb={scheduler_info.get('ucb','?'):.2f}"

            if dream_info:
                msg += f" | Dream ripple={dream_info.get('ripple')} val={dream_info.get('valence',0):.2f} aro={dream_info.get('arousal',0):.2f}"
                msg += f" motifs={dream_info.get('motifs',0)}"

            if relmem_info:
                msg += f" | RelMem inv_loss={relmem_info.get('inverse_loss',0):.3f}"
                if relmem_info.get("hebbian_applied"): 
                    msg += " HEBB"
                if relmem_info.get("wta_applied"): 
                    msg += " WTA"

            print(msg)
    
    def log_epoch(self, epoch: int, phase: str, avg_loss: float, 
                  eval_metrics: Optional[Dict[str, float]] = None):
        """Log epoch completion with evaluation metrics"""
        record = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "epoch": epoch,
            "phase": phase,
            "avg_loss": avg_loss,
            "eval_metrics": eval_metrics or {}
        }
        
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
            
        if self.verbose:
            msg = f"🎯 Epoch {epoch} ({phase}) Complete | Avg Loss: {avg_loss:.4f}"
            if eval_metrics:
                if "exact@1" in eval_metrics:
                    msg += f" | Exact@1: {eval_metrics['exact@1']:.1%}"
                if "iou" in eval_metrics:
                    msg += f" IoU: {eval_metrics['iou']:.3f}"
            print(msg)
    
    def log_milestone(self, milestone: str, data: Dict[str, Any]):
        """Log important training milestones"""
        record = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "milestone": milestone,
            "data": data
        }
        
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
            
        if self.verbose:
            print(f"🏆 Milestone: {milestone} | Data: {data}")
    
    def get_recent_losses(self, n=10) -> list:
        """Get recent loss values for trend analysis"""
        losses = []
        try:
            with open(self.log_path, "r") as f:
                for line in f:
                    record = json.loads(line.strip())
                    if "losses" in record and "total" in record["losses"]:
                        losses.append(record["losses"]["total"])
        except FileNotFoundError:
            pass
        return losses[-n:] if losses else []
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get comprehensive training statistics"""
        stats = {
            "total_batches": 0,
            "avg_loss": 0.0,
            "best_loss": float('inf'),
            "recent_trend": "stable"
        }
        
        try:
            losses = []
            with open(self.log_path, "r") as f:
                for line in f:
                    record = json.loads(line.strip())
                    if "losses" in record and "total" in record["losses"]:
                        loss = record["losses"]["total"]
                        losses.append(loss)
                        stats["total_batches"] += 1
            
            if losses:
                stats["avg_loss"] = sum(losses) / len(losses)
                stats["best_loss"] = min(losses)
                
                # Trend analysis (last 10 vs previous 10)
                if len(losses) >= 20:
                    recent = sum(losses[-10:]) / 10
                    previous = sum(losses[-20:-10]) / 10
                    if recent < previous * 0.9:
                        stats["recent_trend"] = "improving"
                    elif recent > previous * 1.1:
                        stats["recent_trend"] = "degrading"
                    else:
                        stats["recent_trend"] = "stable"
        except FileNotFoundError:
            pass
            
        return stats
    
    def print_summary(self):
        """Print a summary of training statistics"""
        stats = self.get_training_stats()
        print(f"[TrainLogger] Summary: Total batches: {stats['total_batches']}, "
              f"Avg loss: {stats['avg_loss']:.4f}, Best loss: {stats['best_loss']:.4f}, "
              f"Trend: {stats['recent_trend']}")
    
    def close(self):
        """Close the logger and finalize logs"""
        print(f"[TrainLogger] Closed. Logs written to {self.log_path}")
    
    def log(self, data: Dict[str, Any]):
        """Generic logging method for arbitrary data"""
        record = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            **data
        }
        
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        
        if self.verbose and "phase" in data:
            phase = data.get("phase", "?")
            epoch = data.get("epoch", "?")
            batch = data.get("batch", "?")
            loss = data.get("loss", 0.0)
            print(f"[Phase {phase}] Epoch {epoch}, Batch {batch}, Loss: {loss:.4f}")
    
    def log_error(self, message: str, error_data: Optional[Dict[str, Any]] = None):
        """Log an error message"""
        record = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "error": message,
            "details": error_data or {}
        }
        
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        
        if self.verbose:
            print(f"❌ ERROR: {message}")

# Example usage for integration
"""
from train_logger import TrainLogger
logger = TrainLogger(verbose=True)

for batch_idx, batch in enumerate(train_loader):
    # ... do forward, compute losses ...
    losses = {
        "total": total_loss.item(),
        "painter": painter_loss.item(),
        "dsl": dsl_loss.item() if dsl_loss else 0.0,
        "ebr": ebr_loss.item() if ebr_loss else 0.0,
        "aux": aux_loss.item() if aux_loss else 0.0,
    }
    scheduler_info = {
        "task_id": task_id,
        "bucket": bucket,
        "difficulty": diff_est,
        "ucb": ucb_score
    }
    dream_info = {
        "ripple": ripple_ctx.is_active if ripple_ctx else False,
        "valence": valence,
        "arousal": arousal,
        "motifs": len(motifs) if motifs else 0
    }
    relmem_info = {
        "inverse_loss": relmem.inverse_loss().item() if relmem else 0.0,
        "hebbian_applied": hebb_applied,
        "wta_applied": wta_applied
    }

    logger.log_batch(batch_idx, losses, scheduler_info, dream_info, relmem_info)
"""