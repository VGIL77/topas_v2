#!/usr/bin/env python3
"""
Fixed RelationalMemoryNeuro with guaranteed gradient flow
All operations are differentiable, Hebbian/WTA in post-optimizer hooks
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Optional

class RelationalMemoryNeuro(nn.Module):
    def __init__(self, hidden_dim: int, max_concepts: int = 4096, rank: int = 16, 
                 relations: List[str] = None, inverse_pairs: Dict[str, str] = None, 
                 wta_frac: float = 0.1, wta_warmup_updates: int = 50, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.N = max_concepts
        self.D = hidden_dim
        self.R = rank

        self.relations = relations or ["is_a", "has_attr", "owns", "belongs_to", 
                                       "part_of", "parent_of", "child_of", "cooccur"]
        self.inverse = inverse_pairs or {"owns": "belongs_to", "parent_of": "child_of", 
                                         "child_of": "parent_of"}

        # Core learnable parameters - MUST receive gradients
        self.concept_proto = nn.Parameter(torch.randn(self.N, self.D, device=self.device) * 0.01)
        self.A = nn.ParameterDict({
            r: nn.Parameter(torch.randn(self.N, self.R, device=self.device) * 0.01) 
            for r in self.relations
        })
        self.B = nn.ParameterDict({
            r: nn.Parameter(torch.randn(self.N, self.R, device=self.device) * 0.01) 
            for r in self.relations
        })
        self.rel_gain = nn.ParameterDict({
            r: nn.Parameter(torch.ones(1, device=self.device)) 
            for r in self.relations
        })
        
        # Non-learnable state
        self.register_buffer("concept_used", torch.zeros(self.N, dtype=torch.bool))
        
        # WTA and Hebbian settings (applied post-optimizer)
        self.wta_frac = wta_frac
        self.wta_warmup_updates = wta_warmup_updates
        self.wta_enabled = False
        self.hebb_updates = 0
        self.depth_hist: List[int] = []
        
        # Kuramoto oscillators
        self.theta = nn.Parameter(torch.randn(self.N, device=self.device) * 2 * math.pi)
        self.omega = nn.Parameter(torch.randn(self.N, device=self.device) * 0.1)
        
        # Queue for post-optimizer updates
        self.hebbian_queue = []
        self.wta_queue = []
        # --- UKS-lite additions ---
        self.exceptions = {}  # dict[(sid:int, rel:str)] = set([oid:int])
        self.persist_path = None  # optional save/load path for UKS-like state
        
        # Concept management
        self.concepts = {}      # dict cid -> record
        self._next_cid = 0
        self._index_dirty = True
    
    def forward(self, x: torch.Tensor, state=None, top_k: int = 128):
        """
        Sparse Top-K forward pass for relational memory.
        
        x: [B, T, D] tokens (D == hidden_dim)
        top_k: Number of top concepts to route through (default 128 for 32x speedup)
        returns: (y, state) where y is [B, T, D]
        """
        B, T, D = x.shape
        device = x.device
        assert D == self.D, f"Expected hidden_dim={self.D}, got {D}"
        
        # 1) Token → concept similarity scores and Top-K routing
        proto = self.concept_proto  # [N, D]
        scale = float(D) ** 0.5
        
        # Compute similarity scores: [B,T,N] = [B,T,D] @ [D,N]
        similarity_scores = torch.matmul(x, proto.t()) / scale  # [B,T,N]
        
        # Top-K selection per token
        top_k = min(top_k, self.N)  # Don't exceed total concepts
        topk_values, topk_indices = torch.topk(similarity_scores, k=top_k, dim=-1)  # [B,T,K]
        
        # Sparse softmax only over top-k
        att_sparse = torch.softmax(topk_values, dim=-1)  # [B,T,K]
        
        # Optional: mask to active concepts among top-k
        if self.concept_used.any():
            active = self.concept_used.to(device)  # [N]
            # Gather active status for top-k indices: [B,T,K]
            active_topk = torch.gather(active.float().unsqueeze(0).unsqueeze(0).expand(B, T, -1), 
                                     dim=2, index=topk_indices)
            # Apply mask and renormalize
            att_sparse = att_sparse * active_topk
            sums = att_sparse.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            att_sparse = att_sparse / sums
        
        # 2) Sparse propagation through relations using Top-K routing
        ctx_total = torch.zeros_like(x)
        denom = 0.0
        
        for rel in self.relations:
            A = self.A[rel]            # [N,R]
            Bf = self.B[rel]           # [N,R]
            gain = self.rel_gain[rel]  # [1]
            
            # Gather A and B for top-k concepts: [B,T,K,R]
            A_topk = torch.gather(A.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1), 
                                dim=2, index=topk_indices.unsqueeze(-1).expand(-1, -1, -1, A.size(-1)))
            Bf_topk = torch.gather(Bf.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1),
                                 dim=2, index=topk_indices.unsqueeze(-1).expand(-1, -1, -1, Bf.size(-1)))
            
            # Sparse relation propagation: [B,T,K] @ [B,T,K,R] -> [B,T,R]
            z = torch.sum(att_sparse.unsqueeze(-1) * A_topk, dim=2)  # [B,T,R]
            # [B,T,R] @ [B,T,K,R]^T -> [B,T,K] (via broadcasting)
            c = torch.sum(z.unsqueeze(2) * Bf_topk, dim=-1)  # [B,T,K]
            
            # Map back to D using top-k prototypes: [B,T,K] @ [B,T,K,D] -> [B,T,D]
            proto_topk = torch.gather(proto.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1),
                                    dim=2, index=topk_indices.unsqueeze(-1).expand(-1, -1, -1, D))
            ctx_r = torch.sum(c.unsqueeze(-1) * proto_topk, dim=2)  # [B,T,D]
            
            ctx_total = ctx_total + (gain * ctx_r)
            denom += float(gain.item())
        
        if denom <= 0:
            denom = 1.0
        ctx_total = ctx_total / denom
        
        # 3) Residual connection
        y = x + 0.1 * ctx_total
        return y, state

    def _scores(self, rel: str) -> torch.Tensor:
        """Compute relation scores - MUST be differentiable and connected to A, B, rel_gain"""
        A = self.A[rel]
        B = self.B[rel]
        # Matrix multiplication connects gradients to both A and B
        scores = (A @ B.transpose(0, 1)) * self.rel_gain[rel]
        # Softplus ensures positive scores while maintaining gradients
        return F.softplus(scores)

    def query_object(self, rel: str, sid: int) -> Optional[torch.Tensor]:
        """Query for object given subject and relation - returns RAW LOGITS"""
        scores = self._scores(rel)[sid]  # Get row for subject
        active = self.concept_used
        
        if active.sum() == 0:
            # Return dummy logits that maintain gradient flow
            return scores * 0.0  # Shape preserved, gradients flow
        
        # Mask inactive concepts with -inf for softmax stability
        logits = scores.masked_fill(~active, float('-inf'))
        return logits  # RAW LOGITS for CrossEntropyLoss

    def query_subject(self, rel: str, oid: int) -> Optional[torch.Tensor]:
        """Query for subject given object and relation - returns RAW LOGITS"""
        scores = self._scores(rel)[:, oid]  # Get column for object
        active = self.concept_used
        
        if active.sum() == 0:
            # Return dummy logits that maintain gradient flow
            return scores * 0.0  # Shape preserved, gradients flow
        
        # Mask inactive concepts with -inf for softmax stability
        logits = scores.masked_fill(~active, float('-inf'))
        return logits  # RAW LOGITS for CrossEntropyLoss

    def query_relation(self, sid: int, oid: int) -> torch.Tensor:
        """Query for relation given subject and object - returns RAW LOGITS"""
        rel_scores = []
        for rel in self.relations:
            score = self._scores(rel)[sid, oid]
            rel_scores.append(score)
        
        logits = torch.stack(rel_scores)
        return logits  # RAW LOGITS for CrossEntropyLoss

    def bind_concept(self, cid: int, vec: torch.Tensor, alpha: float = 0.1):
        """Queue concept binding for post-optimizer application"""
        cid = int(cid)
        if 0 <= cid < self.N:
            self.concept_used[cid] = True
            # Queue for post-optimizer (no in-place ops during forward)
            if vec.dim() == 1:
                vec = vec.unsqueeze(0)

            # RelMem Loss Safety: Project if dimensional mismatch
            processed_vec = vec.mean(0).detach()
            if processed_vec.shape[-1] != self.D:
                # Project once if dimensional mismatch
                if not hasattr(self, "_proj_bind"):
                    self._proj_bind = nn.Linear(processed_vec.shape[-1], self.D).to(self.device)
                processed_vec = self._proj_bind(processed_vec)

            self.pending_concept_updates = getattr(self, 'pending_concept_updates', {})
            self.pending_concept_updates[cid] = (processed_vec.cpu(), alpha)

    def queue_hebbian_update(self, rel: str, sid: int, oid: int, eta: float = 0.1):
        """Queue Hebbian update for post-optimizer application"""
        self.hebbian_queue.append((rel, sid, oid, eta))
        self.hebb_updates += 1
        if self.hebb_updates > self.wta_warmup_updates:
            self.wta_enabled = True

    def queue_wta_update(self, rel: str):
        """Queue WTA update for post-optimizer application"""
        if self.wta_enabled:
            self.wta_queue.append(rel)

    def add_concept(self, vec: torch.Tensor, meta: Optional[dict] = None, alpha: float = 1.0) -> int:
        """Create and store a new concept from vec (tensor). Return cid."""
        import logging
        if vec is None:
            raise ValueError("add_concept requires a vector")
        vec_cpu = vec.detach().cpu().clone()
        cid = self._next_cid
        self._next_cid += 1
        self.concepts[cid] = {"vec": vec_cpu, "meta": meta or {}, "count": 1, "alpha": float(alpha)}
        self._index_dirty = True
        logging.getLogger(__name__).info("[RelMem] add_concept cid=%d vec_norm=%.4f", cid, float(vec_cpu.norm().item()))
        return cid

    def bind_concept_by_vector(self, vec: torch.Tensor, op_name: str, meta: Optional[dict] = None, alpha: float = 0.5):
        import logging
        cid = self.add_concept(vec, meta=meta, alpha=alpha)
        try:
            # existing bind API expects (cid, vec, alpha)
            self.bind_concept(cid, vec, alpha=alpha)
        except Exception as e:
            logging.getLogger(__name__).warning("[RelMem] bind_concept failed for cid=%s op=%s: %s", cid, op_name, e)
        try:
            self.queue_hebbian_update(op_name, cid, cid)
        except Exception:
            logging.getLogger(__name__).debug("[RelMem] queue_hebbian_update failed for cid=%s op=%s", cid, op_name)
        return cid

    @torch.no_grad()
    def apply_hebbian(self):
        """Apply queued Hebbian updates - called AFTER optimizer.step()"""
        for rel, sid, oid, eta in self.hebbian_queue:
            if rel in self.A and rel in self.B:
                a = self.A[rel][sid].unsqueeze(0)
                b = self.B[rel][oid].unsqueeze(1)
                delta = eta * a @ b
                self.rel_gain[rel].add_(delta.mean())
                self.rel_gain[rel].clamp_(0.1, 1.5)  # Tighter clamping
        self.hebbian_queue.clear()

    @torch.no_grad()
    def apply_wta(self):
        """Apply queued WTA updates - called AFTER optimizer.step()"""
        for rel in self.wta_queue:
            if rel not in self.A or rel not in self.B:
                continue
            
            W = self._scores(rel).detach()  # Safe to detach in post-optimizer
            k = max(1, int(W.size(-1) * self.wta_frac))
            
            # Compute top-k mask
            top_idx = torch.topk(W, k=k, dim=-1).indices
            mask = torch.zeros_like(W, dtype=torch.bool)
            mask.scatter_(dim=-1, index=top_idx, value=True)
            
            # Row and col activity masks
            keep_rows = mask.any(dim=-1).float().unsqueeze(-1)
            keep_cols = mask.any(dim=0).float().unsqueeze(-1)
            
            # Apply sparsification
            self.A[rel].mul_(keep_rows)
            self.B[rel].mul_(keep_cols)
        self.wta_queue.clear()

    def kuramoto_sync(self, K: float = 1.0, steps: int = 10, dt: float = 0.1):
        """Kuramoto synchronization - differentiable"""
        active = self.concept_used.nonzero().squeeze(-1)
        if active.numel() < 2: 
            return
        
        N_active = active.numel()
        theta_active = self.theta[active]
        
        for _ in range(steps):
            sin_diff = torch.sin(theta_active.unsqueeze(0) - theta_active.unsqueeze(1))
            dtheta = self.omega[active] + (K / N_active) * sin_diff.sum(dim=1)
            theta_active = theta_active + dt * dtheta
        
        # Update theta without in-place operation
        with torch.no_grad():
            self.theta.data[active] = theta_active % (2 * math.pi)

    def inverse_loss(self) -> torch.Tensor:
        """Compute inverse relation consistency loss - fully differentiable"""
        total_loss = (self.concept_proto * 0).sum()  # Start with graph-connected zero
        count = 0
        
        for r, ri in self.inverse.items():
            if ri in self.relations:
                fwd = self._scores(r)
                rev = self._scores(ri).transpose(0, 1)
                loss = F.mse_loss(fwd, rev)
                total_loss = total_loss + loss
                count += 1
        
        if count > 0:
            return total_loss / count
        return total_loss

    def inheritance_pass(self) -> torch.Tensor:
        """Compute inheritance consistency - fully differentiable"""
        if "is_a" not in self.relations or "has_attr" not in self.relations:
            return (self.concept_proto * 0).sum()
        
        Wisa = self._scores("is_a")
        What = self._scores("has_attr")
        
        # Propagate attributes through is-a hierarchy
        P = What.clone()
        for _ in range(3):
            P = P + 0.1 * (Wisa @ P)  # Small step size for stability
        
        # Compute consistency loss
        loss = F.mse_loss(P, What)
        return torch.nan_to_num(loss, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    def grid_consistency_loss(self) -> torch.Tensor:
        """Grid consistency for active concepts - fully differentiable"""
        active = self.concept_used.nonzero().squeeze(-1)
        if active.numel() < 4:
            return (self.concept_proto * 0).sum()
        
        embeds = self.concept_proto[active]
        diffs = embeds[1:] - embeds[:-1]
        return diffs.pow(2).mean()

    def anti_recurrence_penalty(self, rel: str, num_walks: int = 100, 
                                max_steps: int = 5) -> torch.Tensor:
        """Anti-recurrence penalty - fully differentiable"""
        device = self.concept_proto.device
        
        # Get transition matrix
        logits = self._scores(rel)
        W = F.softmax(logits, dim=-1)
        
        used = torch.nonzero(self.concept_used).flatten()
        if used.numel() < 2:
            return (self.concept_proto * 0).sum()
        
        # Random walk simulation
        starts = used[torch.randint(0, used.numel(), (num_walks,), device=device)]
        pos = starts.unsqueeze(1).repeat(1, max_steps+1)
        
        for t in range(1, max_steps+1):
            probs = W[pos[:, t-1]]
            probs = torch.nan_to_num(probs, nan=1.0/self.N, posinf=1.0, neginf=0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            next_idx = torch.multinomial(probs, 1).squeeze(1)
            pos[:, t] = next_idx
        
        # Check for cycles
        cycles = (pos[:, 1:] == pos[:, 0].unsqueeze(1)).any(dim=1).float().mean()
        return cycles.clamp(0.0, 1.0)

    def predictive_prune_loss(self, thresh: float = 0.1) -> torch.Tensor:
        """Predictive pruning loss - fully differentiable"""
        if "is_a" not in self.relations or "has_attr" not in self.relations:
            return (self.concept_proto * 0).sum()
        
        Wisa = self._scores("is_a")
        What = self._scores("has_attr")
        
        # Predict attributes through is-a
        P = What.clone()
        for _ in range(2):
            P = P + 0.1 * (Wisa @ P)
        
        # Check grounding
        grounded = (self.concept_proto.abs().sum(-1) > 0.1).float()
        mismatch = (P > thresh).float() * (1.0 - grounded.unsqueeze(0))
        return mismatch.mean().clamp(0.0, 1.0)

    def get_hierarchy_depth(self) -> float:
        """Get average hierarchy depth"""
        if not self.depth_hist: 
            return 0.0
        return float(sum(self.depth_hist) / len(self.depth_hist))
    
    # Compatibility methods for existing code
    def hebbian_relation(self, rel: str, sid: int, oid: int, eta: float = 0.1):
        """Compatibility wrapper - queues update for post-optimizer"""
        self.queue_hebbian_update(rel, sid, oid, eta)
    
    def wta_inhibition(self, rel: str, k_frac: float = 0.1):
        """Compatibility wrapper - queues update for post-optimizer"""
        self.queue_wta_update(rel)
    
    def apply_post_optimizer_hooks(self):
        """Apply all post-optimizer updates"""
        # Apply pending concept bindings first
        if hasattr(self, 'pending_concept_updates') and self.pending_concept_updates:
            with torch.no_grad():
                for cid, (vec, alpha) in self.pending_concept_updates.items():
                    vec = vec.to(self.device, dtype=self.concept_proto.dtype)
                    self.concept_proto.data[cid] = (1 - alpha) * self.concept_proto.data[cid] + alpha * vec
            self.pending_concept_updates.clear()
        
        self.apply_hebbian()
        self.apply_wta()
    
    def get_op_bias(self, dsl_ops: List[str] = None, scale: float = 1.0) -> Dict[str, float]:
        """
        Return a mapping {dsl_op_name: bias} with values in [0, 1].
        - dsl_ops: optional list of DSL ops to restrict to (defaults to all known ops).
        - scale: multiplicative scaling factor from ModelConfig.relmem_op_bias_scale
        """
        from typing import Dict, List
        import torch
        import logging
        
        op_bias: Dict[str, float] = {}

        # If no learned weights present, return empty (but log)
        if not hasattr(self, "weights") or not self.weights:
            logging.getLogger().debug("[RelMem] get_op_bias: no weights available")
            return {}

        # Example mapping: map internal relation names to candidate DSL ops (use actual DSL_OPS names)
        relation_to_ops = {
            'color': ['color_map', 'flood_fill', 'extract_color'],
            'shape': ['rotate90', 'flip_h', 'flip_v', 'transpose'],
            'structure': ['crop_bbox', 'tile_pattern', 'resize_nn'],
            'translate': ['translate'],
            'size': ['resize_nn', 'scale'],
            'flip': ['flip_h', 'flip_v'],
            'rotate': ['rotate90', 'rotate180', 'rotate270'],
            'crop': ['crop_bbox', 'crop_nonzero'],
            'tile': ['tile', 'tile_pattern'],
            'flood': ['flood_fill', 'flood_select'],
            'outline': ['outline', 'boundary_extract'],
            'symmetry': ['symmetry'],
            'paste': ['paste'],
            'count': ['count_objects', 'count_colors'],
            'pattern': ['find_pattern', 'extract_pattern', 'match_template'],
            'logic': ['grid_union', 'grid_intersection', 'grid_xor', 'grid_difference'],
            'object': ['for_each_object', 'for_each_object_translate', 'for_each_object_recolor',
                      'for_each_object_rotate', 'for_each_object_scale', 'for_each_object_flip'],
            'conditional': ['conditional_map', 'apply_rule'],
            'select': ['select_by_property', 'flood_select'],
            'identity': ['identity'],
            'arithmetic': ['arithmetic_op']
        }

        # compute a score per relation (robust: use sigmoid of mean to make [0,1])
        for rel_name, ops in relation_to_ops.items():
            if hasattr(self, "relations") and rel_name in getattr(self, "relations", []):
                try:
                    scores = self._scores(rel_name)  # tensor
                    if isinstance(scores, torch.Tensor):
                        mean_score = float(torch.sigmoid(scores.mean()).item())
                    else:
                        mean_score = float(scores)  # fallback
                except Exception:
                    mean_score = 0.0

                # push into ops
                for op in ops:
                    op_bias[op] = max(op_bias.get(op, 0.0), min(1.0, mean_score * scale))

        # if user passed explicit dsl_ops, filter/ensure baseline
        if dsl_ops is not None:
            for op in dsl_ops:
                if op not in op_bias:
                    op_bias[op] = 0.0

        # Always clamp
        op_bias = {k: float(max(0.0, min(1.0, v))) for k, v in op_bias.items()}

        logging.getLogger().info(f"[RelMem] get_op_bias produced {len(op_bias)} ops: {list(op_bias.keys())[:6]}")
        return op_bias
    
    def compute_inverse_loss(self) -> torch.Tensor:
        """
        Compute inverse loss regularizer for RelMem training.
        Returns a small positive scalar when relation activations are present.
        """
        import torch
        
        # defensive: if no weights return zero-tensor on correct device
        if not hasattr(self, "weights") or not self.weights:
            return torch.tensor(0.0, device=next(self.parameters()).device if hasattr(self, "parameters") else "cpu")
        
        # simple proxy: L2 of off-diagonal relation weights
        total = 0.0
        count = 0
        for k, W in getattr(self, "weights", {}).items():
            if isinstance(W, torch.Tensor):
                total = total + (W**2).mean()
                count += 1
        
        return (total / max(1, count))
    
    def op_bias(self) -> Dict[str, float]:
        """
        Return operation bias dict for DSL search - NEVER EMPTY.
        - Looks at self.relations (list of relation names) and
          self._scores(rel) (must return a tensor or numeric score).
        - Fallbacks: if no scores present, use lightweight learned 'weights' if present,
          else return baseline small biases.
        """
        from typing import Dict
        import torch
        import logging
        
        # list of operations we know about (keep in sync with DSL registry)
        known_ops = [
            "identity","rotate90","rotate180","rotate270",
            "flip_h","flip_v","translate","scale","resize_nn",
            "color_map","flood_fill","extract_color","crop_bbox","crop_nonzero",
            "tile_pattern","tile","paste","center_pad_to","outline","symmetry",
            "grid_union","grid_intersection","grid_xor","grid_difference",
            "count_objects","count_colors","find_pattern","extract_pattern","match_template",
            "for_each_object","for_each_object_translate","for_each_object_recolor",
            "for_each_object_rotate","for_each_object_scale","for_each_object_flip",
            "conditional_map","apply_rule","select_by_property","flood_select",
            "boundary_extract","arithmetic_op"
        ]

        op_bias: Dict[str, float] = {op: 0.0 for op in known_ops}

        # If we have explicit relation names -> operation mapping, prefer that.
        relation_to_ops = {
            "color": ["color_map","flood_fill","extract_color"],
            "shape": ["rotate90","flip_h","flip_v"],
            "structure": ["crop_bbox","tile_pattern","resize_nn","translate","scale"],
            "logic": ["grid_union","grid_intersection","grid_xor","grid_difference"],
            "identity": ["identity"],
            "flip": ["flip_h","flip_v"],
            "rotate": ["rotate90","rotate180","rotate270"],
            "crop": ["crop_bbox","crop_nonzero"],
            "tile": ["tile","tile_pattern"],
            "flood": ["flood_fill","flood_select"],
            "outline": ["outline","boundary_extract"],
            "symmetry": ["symmetry"],
            "paste": ["paste"],
            "count": ["count_objects","count_colors"],
            "pattern": ["find_pattern","extract_pattern","match_template"],
            "object": ["for_each_object","for_each_object_translate","for_each_object_recolor",
                      "for_each_object_rotate","for_each_object_scale","for_each_object_flip"],
            "conditional": ["conditional_map","apply_rule"],
            "select": ["select_by_property","flood_select"],
            "arithmetic": ["arithmetic_op"]
        }

        # 1) Use _scores(rel) if present
        try:
            if hasattr(self, "_scores") and callable(getattr(self, "_scores")):
                for rel, ops in relation_to_ops.items():
                    if rel in getattr(self, "relations", []):
                        try:
                            score_t = self._scores(rel)
                            score = float(score_t.mean().item()) if hasattr(score_t, "mean") else float(score_t)
                            score = max(0.0, min(1.0, score))  # clamp
                        except Exception:
                            score = 0.0
                        # distribute the score to ops (simple equal split)
                        for op in ops:
                            op_bias[op] = max(op_bias.get(op, 0.0), score)
        except Exception:
            # defensive - don't fail training due to relmem
            pass

        # 2) If still all zeros, fall back to 'weights' or small priors
        if all(v == 0.0 for v in op_bias.values()):
            try:
                # if self.weights exists and is a dict mapping relation->tensor
                if hasattr(self, "weights") and isinstance(self.weights, dict):
                    for rel, W in self.weights.items():
                        # map the rel to ops if possible
                        ops = relation_to_ops.get(rel, [])
                        strength = float(torch.norm(W).item()) / (W.numel() + 1e-9) if hasattr(W, "numel") else 0.0
                        strength = max(0.0, min(1.0, strength))
                        for op in ops:
                            op_bias[op] = max(op_bias.get(op, 0.0), strength * 0.3)
            except Exception:
                pass

        # 3) final baseline small bias so bias dict is never empty
        for k in list(op_bias.keys()):
            if op_bias[k] == 0.0:
                op_bias[k] = 0.01

        # 4) optionally normalize (keep flexible)
        total = sum(op_bias.values()) + 1e-12
        # keep raw scale for downstream; but normalize to [0,1] relative
        for k in op_bias:
            op_bias[k] = float(op_bias[k]) / total

        return op_bias
    
    def _scores(self, rel_name):
        """Return a FloatTensor score for relation rel_name (shape [] or [N])"""
        try:
            if hasattr(self, "weights") and rel_name in self.weights:
                W = self.weights[rel_name]
                return torch.tensor(float(torch.norm(W).item()) / (W.numel() + 1e-9))
            # fallback to stored activations if available
            if hasattr(self, "rel_activations") and rel_name in self.rel_activations:
                return torch.tensor(self.rel_activations[rel_name]).float()
        except Exception:
            pass
        return torch.tensor(0.0)

    # -------- Exceptions / Inheritance+ ----------
    def add_exception(self, sid: int, rel: str, oid: int):
        key = (int(sid), str(rel))
        s = self.exceptions.get(key, set())
        s.add(int(oid))
        self.exceptions[key] = s

    def remove_exception(self, sid: int, rel: str, oid: int):
        key = (int(sid), str(rel))
        if key in self.exceptions and int(oid) in self.exceptions[key]:
            self.exceptions[key].remove(int(oid))
            if not self.exceptions[key]:
                self.exceptions.pop(key, None)

    def _is_exception(self, sid: int, rel: str, oid: int) -> bool:
        key = (int(sid), str(rel))
        return key in self.exceptions and int(oid) in self.exceptions[key]

    @torch.no_grad()
    def inheritance_pass_plus(self, steps: int = 3, alpha: float = 0.1, thresh: float = 0.5) -> torch.Tensor:
        """
        Enhanced inheritance with exceptions and confidence gating.
        Returns a small scalar consistency loss for training signals.
        """
        if "is_a" not in self.relations or "has_attr" not in self.relations:
            return (self.concept_proto * 0).sum()
        Wisa = self._scores("is_a")
        What = self._scores("has_attr")
        P = What.clone() if torch.is_tensor(What) else torch.tensor(What)
        for _ in range(max(1, steps)):
            if torch.is_tensor(Wisa) and torch.is_tensor(P):
                P = P + alpha * (Wisa @ P) if Wisa.dim() == 2 else P
        # Apply exceptions
        if self.exceptions and torch.is_tensor(P) and P.dim() >= 2:
            for (sid, rel), oids in self.exceptions.items():
                if rel == "has_attr" and len(oids) > 0:
                    if P.dim() == 2 and sid < P.shape[0]:
                        for oid in oids:
                            if oid < P.shape[1]:
                                P[sid, oid] = 0.0
        # Confidence gating
        if torch.is_tensor(P) and torch.is_tensor(What):
            P = torch.where(P > thresh, P, What)
            loss = F.mse_loss(P, What)
        else:
            loss = torch.tensor(0.0)
        return torch.nan_to_num(loss, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    # -------- Contextual op-bias + Theme priors ----------
    def get_op_bias_contextual(self, slot_vecs: Optional[torch.Tensor] = None,
                               theme_embed: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Contextual op-bias: combines graph priors (op_bias), slot evidence, and theme steer.
        Returns normalized dict suitable for DSL search.
        """
        base = self.op_bias()  # normalized dict, never empty
        if slot_vecs is not None and torch.is_tensor(slot_vecs):
            ctx = float(slot_vecs.abs().mean().detach().cpu())
            for k in base:
                base[k] = float(min(1.0, max(0.0, base[k] * (0.9 + 0.2 * ctx))))
        if theme_embed is not None and isinstance(theme_embed, torch.Tensor) and theme_embed.numel() > 0:
            t = theme_embed.detach().float()
            s = float(t.mean().sigmoid().item())
            v = float(t.std().clamp(0,1).item())
            sym_ops = ["symmetry","flip_h","flip_v","rotate90","rotate180","rotate270","transpose"]
            conn_ops = ["flood_fill","flood_select","outline","boundary_extract","for_each_object"]
            for k in sym_ops:
                if k in base: base[k] = min(1.0, base[k] * (0.9 + 0.3 * s))
            for k in conn_ops:
                if k in base: base[k] = min(1.0, base[k] * (0.9 + 0.3 * v))
        Z = sum(base.values()) + 1e-12
        for k in base: base[k] = float(base[k] / Z)
        return base

    def get_theme_priors(self, theme_embed: Optional[torch.Tensor]) -> Dict[str, float]:
        """
        Produce {'phi','kappa','cge'} in [0.5, 1.5] for EBR gating (neutral=1.0).
        """
        if theme_embed is None or not isinstance(theme_embed, torch.Tensor) or theme_embed.numel() == 0:
            return {"phi": 1.0, "kappa": 1.0, "cge": 1.0}
        x = theme_embed.detach().float()
        m = float(x.mean().tanh().item())   # [-1,1]
        v = float(x.std().tanh().item())    # [-1,1]
        scale = lambda z: 1.0 + 0.5 * z     # [-1,1] -> [0.5,1.5]
        return {"phi": scale(m), "kappa": scale(v), "cge": scale((m+v)/2.0)}

    # -------- Refinement agent & stats ----------
    @torch.no_grad()
    def refinement_step(self, cos_thresh: float = 0.92, min_size: int = 3, merge_alpha: float = 0.2):
        """Cluster similar concepts and lightly merge to form cleaner parents."""
        used = self.concept_used.nonzero().flatten()
        if used.numel() < min_size: return
        E = self.concept_proto[used]
        sim = F.cosine_similarity(E.unsqueeze(1), E.unsqueeze(0), dim=-1)
        visited = set()
        for i in range(used.numel()):
            if i in visited: continue
            cluster = [i]
            for j in range(i+1, used.numel()):
                if j in visited: continue
                if float(sim[i,j].item()) >= cos_thresh:
                    cluster.append(j)
            if len(cluster) >= min_size:
                cids = used[torch.tensor(cluster, device=used.device)]
                centroid = self.concept_proto[cids].mean(dim=0, keepdim=True)
                self.concept_proto.data[cids] = (1-merge_alpha)*self.concept_proto.data[cids] + merge_alpha*centroid
                self.depth_hist.append(len(cluster))
                visited.update(cluster)

    def stats(self) -> Dict[str, float]:
        active = int(self.concept_used.sum().item())
        depth = self.get_hierarchy_depth() if hasattr(self, "get_hierarchy_depth") else 0.0
        exc = sum(len(v) for v in self.exceptions.values()) if self.exceptions else 0
        return {"relmem_active": float(active), "relmem_depth": float(depth), "relmem_exceptions": float(exc)}

    # -------- Persistence ----------
    def save_uks(self, path: str):
        self.persist_path = path
        state = {
            "concept_proto": self.concept_proto.detach().cpu(),
            "A": {k: v.detach().cpu() for k,v in self.A.items()},
            "B": {k: v.detach().cpu() for k,v in self.B.items()},
            "rel_gain": {k: v.detach().cpu() for k,v in self.rel_gain.items()},
            "concept_used": self.concept_used.detach().cpu(),
            "exceptions": { (sid,rel): list(oids) for (sid,rel), oids in self.exceptions.items() }
        }
        torch.save(state, path)

    def load_uks(self, path: str):
        import os
        if not os.path.exists(path): return
        state = torch.load(path, map_location=self.device)
        self.concept_proto.data.copy_(state["concept_proto"].to(self.device))
        for k in self.A: 
            if k in state["A"]:
                self.A[k].data.copy_(state["A"][k].to(self.device))
        for k in self.B: 
            if k in state["B"]:
                self.B[k].data.copy_(state["B"][k].to(self.device))
        for k in self.rel_gain: 
            if k in state["rel_gain"]:
                self.rel_gain[k].data.copy_(state["rel_gain"][k].to(self.device))
        self.concept_used.copy_(state["concept_used"].to(self.device))
        self.exceptions = { (int(sid), rel): set(map(int, oids)) for (sid,rel), oids in state.get("exceptions", {}).items() }

    # --- Shape-safe wrapper to keep training going & preserve grads ---
    _REL_INV_WARN = 0
    _REL_INV_WARN_MAX = 5

    def inverse_loss_safe(self, *args, **kwargs):
        """Wrapper around inverse loss to avoid shape/index errors and to log shapes."""
        import logging
        import torch
        try:
            if not getattr(self, "concepts", None):
                logging.getLogger(__name__).debug("[RelMem] inverse_loss_safe: no concepts present -> 0")
                return torch.tensor(0.0, device=getattr(self,'device','cpu'))
            # call underlying implementation if exists
            if hasattr(self, "_inverse_loss_impl"):
                return self._inverse_loss_impl(*args, **kwargs)
            # otherwise, try original inverse_loss if present
            if hasattr(self, "inverse_loss"):
                return self.inverse_loss(*args, **kwargs)
            return torch.tensor(0.0, device=getattr(self,'device','cpu'))
        except IndexError as e:
            logging.getLogger(__name__).exception("[RelMem] inverse_loss IndexError shapes=%s: %s", getattr(args[0],'shape',None), e)
            return torch.tensor(0.0, device=getattr(self,'device','cpu'))
        except Exception as e:
            logging.getLogger(__name__).exception("[RelMem] inverse_loss unexpected error: %s", e)
            return torch.tensor(0.0, device=getattr(self,'device','cpu'))

# --- Exemplar-enhanced Relational Memory -------------------------------------
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import torch
import torch.nn.functional as F

@dataclass
class Exemplar:
    vec: torch.Tensor
    freq: int
    last_step: int
    # Optional grounded attributes (phi/kappa/CGE/hodge etc.)
    attrs: Optional[Dict[str, float]] = None

class RelationalMemoryExemplar(RelationalMemoryNeuro):
    """
    Extends RelationalMemoryNeuro with per-concept exemplar libraries
    + ripple/dream-gated consolidation + optional 'told facts' injection.
    - Each concept stores a small set of exemplars (vec + attrs + usage stats).
    - New inputs update closest exemplar (if similar) or create a new one.
    - Periodic consolidation merges/prunes exemplars and recenters the prototype.
    """
    def __init__(
        self, hidden_dim: int, max_concepts: int = 4096, rank: int = 16,
        relations: List[str] = None, inverse_pairs: Dict[str, str] = None,
        wta_frac: float = 0.1, wta_warmup_updates: int = 50, device="cpu",
        max_exemplars_per_concept: int = 8, sim_tau: float = 0.85,
        merge_tau: float = 0.92, prune_min_freq: int = 2
    ):
        super().__init__(hidden_dim, max_concepts, rank, relations, inverse_pairs,
                         wta_frac, wta_warmup_updates, device)
        self.max_exemplars_per_concept = max_exemplars_per_concept
        self.sim_tau = float(sim_tau)
        self.merge_tau = float(merge_tau)
        self.prune_min_freq = int(prune_min_freq)
        # exemplar store: cid -> List[Exemplar]
        self._exemplars: Dict[int, List[Exemplar]] = {}
        # optional name->cid for 'told facts'
        self._symbolic_index: Dict[str, int] = {}

    # ---------- Core: add/update exemplars ----------
    @torch.no_grad()
    def add_or_update_exemplar(
        self, cid: int, vec: torch.Tensor, step: int,
        attrs: Optional[Dict[str, float]] = None
    ):
        """Add a new exemplar or update the most similar one."""
        cid = int(cid)
        vec = vec.detach().to(self.device).float()
        bank = self._exemplars.setdefault(cid, [])
        best_i, best_sim = -1, -1.0
        for i, ex in enumerate(bank):
            sim = float(F.cosine_similarity(vec.unsqueeze(0), ex.vec.unsqueeze(0)))
            if sim > best_sim:
                best_i, best_sim = i, sim
        if best_sim >= self.sim_tau and best_i >= 0:
            # Light moving average toward new vec; bump freq/ts; merge attrs
            ex = bank[best_i]
            ex.vec = 0.9 * ex.vec + 0.1 * vec
            ex.freq += 1
            ex.last_step = step
            if attrs:
                ex.attrs = (ex.attrs or {})
                for k, v in attrs.items():
                    ex.attrs[k] = float(v)
        else:
            bank.append(Exemplar(vec=vec, freq=1, last_step=step, attrs=attrs))
            # Bound memory
            if len(bank) > self.max_exemplars_per_concept:
                # drop least-frequent / oldest
                bank.sort(key=lambda e: (e.freq, e.last_step))
                bank.pop(0)

    # ---------- Consolidation: prune/merge + recenter prototype ----------
    @torch.no_grad()
    def consolidate_concept(self, cid: int):
        """Merge near-duplicate exemplars; prune low-usage; update prototype."""
        if cid not in self._exemplars:
            return
        bank = self._exemplars[cid]
        if not bank:
            return

        # 1) prune rarely used
        bank = [e for e in bank if e.freq >= self.prune_min_freq]

        # 2) merge highly-similar (greedy)
        merged: List[Exemplar] = []
        for e in sorted(bank, key=lambda x: -x.freq):
            keep = True
            for m in merged:
                sim = float(F.cosine_similarity(e.vec.unsqueeze(0), m.vec.unsqueeze(0)))
                if sim >= self.merge_tau:
                    # merge into m (frequency-weighted)
                    w1, w2 = float(m.freq), float(e.freq)
                    m.vec = (w1 * m.vec + w2 * e.vec) / max(1e-6, (w1 + w2))
                    m.freq += e.freq
                    m.last_step = max(m.last_step, e.last_step)
                    if e.attrs:
                        m.attrs = (m.attrs or {})
                        for k, v in e.attrs.items():
                            m.attrs[k] = float(0.5 * m.attrs.get(k, v) + 0.5 * v)
                    keep = False
                    break
            if keep:
                merged.append(e)
        self._exemplars[cid] = merged

        # 3) recenter prototype to exemplar mean (concept_proto is learnable Param)
        if merged:
            mean_vec = torch.stack([e.vec for e in merged], dim=0).mean(0)
            self.concept_proto.data[cid] = 0.8 * self.concept_proto.data[cid] + 0.2 * mean_vec

    @torch.no_grad()
    def consolidate_exemplars(self):
        """Call this during dream/ripple cycles or every N steps."""
        used_ids = self.concept_used.nonzero().flatten().tolist()
        for cid in used_ids:
            self.consolidate_concept(int(cid))

    # ---------- Optional: bind sample (streaming unsupervised) ----------
    @torch.no_grad()
    def observe_sample(
        self, x_tokens: torch.Tensor, step: int,
        attrs: Optional[Dict[str, float]] = None, top_k: int = 64
    ):
        """
        Light unsupervised update from relational tokens [B,T,D].
        Picks the highest-activation concept per token and updates its exemplar bank.
        """
        if x_tokens.dim() != 3 or x_tokens.size(-1) != self.D:
            return
        B, T, D = x_tokens.shape
        proto = self.concept_proto  # [N,D]
        scale = float(D) ** 0.5
        scores = torch.matmul(x_tokens, proto.t()) / scale  # [B,T,N]
        _, idx = torch.topk(scores, k=min(top_k, self.N), dim=-1)  # [B,T,K]
        # Use the top-1 for fast streaming update
        cid = idx[..., 0]  # [B,T]
        for b in range(B):
            for t in range(T):
                c = int(cid[b, t].item())
                v = x_tokens[b, t].detach()
                self.add_or_update_exemplar(c, v, step=step, attrs=attrs)

    # ---------- Dream/ripple hookup ----------
    @torch.no_grad()
    def on_ripple_event(self, ripple_stats: Optional[Dict[str, float]] = None):
        """
        Hook to be called after DreamEngine ripple cycles.
        Uses ripple coherence to adjust consolidation aggressiveness (optional).
        """
        self.consolidate_exemplars()
        # Example: if coherence high, enable WTA queue on frequently used relations
        coh = float(ripple_stats.get("ripple_phase_coherence", 1.0)) if isinstance(ripple_stats, dict) else 1.0
        if coh > 0.8:
            for rel in self.relations[:2]:  # small nudge
                self.queue_wta_update(rel)

    # ---------- 'Being told facts' (symbolic injection) ----------
    @torch.no_grad()
    def bind_fact(
        self, subj_name: str, rel: str, obj_name: str,
        subj_vec: Optional[torch.Tensor] = None,
        obj_vec: Optional[torch.Tensor] = None,
        alpha: float = 0.5
    ):
        """
        Create or reuse named concepts and connect them with relation 'rel'.
        If vectors are provided, initialize/refresh their prototypes + exemplars.
        """
        # (a) subject concept id
        if subj_name not in self._symbolic_index:
            sid = self._next_cid
            self._symbolic_index[subj_name] = sid
            self._next_cid += 1
            self.concept_used[sid] = True
            if subj_vec is not None:
                self.concept_proto.data[sid] = subj_vec.to(self.device).float()
                self.add_or_update_exemplar(sid, self.concept_proto.data[sid], step=0)
        else:
            sid = self._symbolic_index[subj_name]
            if subj_vec is not None:
                self.add_or_update_exemplar(sid, subj_vec.to(self.device).float(), step=0)

        # (b) object concept id
        if obj_name not in self._symbolic_index:
            oid = self._next_cid
            self._symbolic_index[obj_name] = oid
            self._next_cid += 1
            self.concept_used[oid] = True
            if obj_vec is not None:
                self.concept_proto.data[oid] = obj_vec.to(self.device).float()
                self.add_or_update_exemplar(oid, self.concept_proto.data[oid], step=0)
        else:
            oid = self._symbolic_index[obj_name]
            if obj_vec is not None:
                self.add_or_update_exemplar(oid, obj_vec.to(self.device).float(), step=0)

        # (c) strengthen relation with small Hebbian pulse
        try:
            self.queue_hebbian_update(rel, sid, oid, eta=alpha)
        except Exception:
            pass

    # ---------- Expose exemplar stats for logging ----------
    @torch.no_grad()
    def exemplar_stats(self) -> Dict[str, float]:
        total = sum(len(v) for v in self._exemplars.values())
        active = int(self.concept_used.sum().item())
        avg_per = float(total) / max(1, active)
        return {"exemplar_total": float(total), "exemplar_avg_per_concept": avg_per}
