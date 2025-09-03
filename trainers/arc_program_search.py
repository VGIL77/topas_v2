
import numpy as np
import torch
from typing import List, Tuple, Dict, Any
from models.dsl_search import DSLProgram, apply_program, CORE_OPS, DSLProgram
from arc_constraints import ARCGridConstraints
from energy_refinement import EnergyRefiner
from phi_metrics_neuro import phi_synergy_features, kappa_floor, cge_boost
from size_oracle import predict_size

def grid_to_logits(grid: np.ndarray, num_colors:int, deterministic_logits: bool = False) -> torch.Tensor:
    H, W = grid.shape
    C = num_colors
    onehot = np.zeros((C, H, W), dtype=np.float32)
    for c in range(C):
        onehot[c] = (grid == c).astype(np.float32)
    logits = torch.from_numpy(onehot).unsqueeze(0) * 4.0
    if not deterministic_logits:
        logits += 0.01 * torch.randn_like(logits)
    return logits

def logits_to_grid(logits: torch.Tensor) -> np.ndarray:
    probs = torch.softmax(logits, dim=1)[0]
    pred = torch.argmax(probs, dim=0).cpu().numpy().astype(np.int32)
    return pred

class WeightedCollapse:
    @staticmethod
    def collapse(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not candidates:
            raise ValueError("No candidates provided")
        losses = np.array([c['loss'] for c in candidates], dtype=np.float32)
        L = losses.max() - losses.min() + 1e-6
        norm_loss = 1.0 - (losses - losses.min())/L
        simp = np.array([c.get('simplicity',0.5) for c in candidates], dtype=np.float32)
        symm = np.array([c.get('symmetry',0.5) for c in candidates], dtype=np.float32)
        size_compat = np.array([c.get('size_compat',0.0) for c in candidates], dtype=np.float32)
        
        # Adjust weights to incorporate size compatibility (bias towards size-compatible programs)
        scores = 0.4*norm_loss + 0.2*simp + 0.1*symm + 0.3*size_compat
        idx = int(np.argmax(scores))
        return candidates[idx]

class ARCSearchManager:
    def __init__(self, num_colors:int=10, max_len:int=4, candidates:int=12, seed:int=0, use_ebr:bool=True, deterministic_logits: bool=False):
        self.num_colors = num_colors
        self.max_len = max_len
        self.candidates = candidates
        self.rng = np.random.default_rng(seed)
        self.use_ebr = use_ebr
        self.deterministic_logits = deterministic_logits
        
        # Initialize canonical DSL
        # MASTER'S ZOMBIE SLAYING: Replace DSLHead with dummy stub
        class DummyDSL:
            def apply_program(self, grid, prog): return grid
            def random_program(self, *args, **kwargs): return None
        
        self.dsl = DummyDSL()
        
        if use_ebr:
            self.ebr = EnergyRefiner(min_steps=3, max_steps=5, step_size=0.25, noise=0.0, lambda_prior=1.0, w_phi=2.0, w_kappa=2.0, w_cge=2.0)
    
    def execute_program(self, grid: np.ndarray, program: List[Tuple[str, Dict]]) -> np.ndarray:
        """Execute a program using DSLHead - bridge function to maintain compatibility"""
        # Convert numpy grid to torch tensor
        torch_grid = torch.from_numpy(grid).long()
        
        # Convert program list to DSLProgram
        ops = [op for op, _ in program]
        params = [param for _, param in program]
        dsl_prog = DSLProgram(ops=ops, params=params)
        
        # Apply program
        result = self.dsl.apply_program(torch_grid, dsl_prog)
        
        # Convert back to numpy
        return result.detach().cpu().numpy().astype(np.int32)
    
    def _generate_size_aware_program(self, H_in: int, W_in: int, H_out: int, W_out: int, reason: str):
        """Generate program with bias towards operations likely to produce target size"""
        # If reason suggests specific operations, bias towards them
        if "affine_int" in reason or "axis_ratio" in reason:
            # Likely scale operation
            if H_out == 2 * H_in and W_out == 2 * W_in:
                # Try uniform scale 2x
                if self.rng.random() < 0.7:  # 70% chance
                    return [("scale", {"fy": 2, "fx": 2})]
            elif H_out == 3 * H_in and W_out == 3 * W_in:
                # Try uniform scale 3x
                if self.rng.random() < 0.7:
                    return [("scale", {"fy": 3, "fx": 3})]
            elif "axis_ratio" in reason:
                # Try different H/W ratios
                if self.rng.random() < 0.6:
                    ratio_h = max(1, H_out // H_in) if H_in > 0 else 2
                    ratio_w = max(1, W_out // W_in) if W_in > 0 else 3
                    return [("scale", {"fy": ratio_h, "fx": ratio_w})]
                    
        elif "constant_from_demos" in reason:
            # Likely pad_to operation
            if self.rng.random() < 0.8:  # 80% chance
                return [("pad_to", {"H": H_out, "W": W_out, "pad_value": 0})]
        
        elif "bbox" in reason:
            # Likely crop_bbox then maybe scale or pad
            if self.rng.random() < 0.5:
                prog = [("crop_bbox", {})]
                # Maybe add scaling if needed
                if self.rng.random() < 0.3:
                    prog.append(("scale", {"fy": 2, "fx": 2}))
                return prog
        
        # Fallback to regular random program with some size-aware bias
        return self._generate_biased_random_program(H_in, W_in, H_out, W_out)
    
    def _generate_biased_random_program(self, H_in: int, W_in: int, H_out: int, W_out: int):
        """Generate random program with light bias towards size-changing operations"""
        prog_len = int(self.rng.integers(1, self.max_len + 1))
        prog = []
        
        for i in range(prog_len):
            # For first operation, bias towards size-changing ops if size differs
            if i == 0 and (H_in != H_out or W_in != W_out):
                if self.rng.random() < 0.4:  # 40% chance to use size-changing op
                    op_choices = ["scale", "pad_to", "crop_bbox"]
                    op = self.rng.choice(op_choices)
                    if op == "scale":
                        fy = int(self.rng.choice([2, 3]))
                        fx = int(self.rng.choice([2, 3]))
                        prog.append((op, {"fy": fy, "fx": fx}))
                    elif op == "pad_to":
                        # Random target size around expected output
                        target_h = int(self.rng.integers(max(H_out-2, 5), H_out+3))
                        target_w = int(self.rng.integers(max(W_out-2, 5), W_out+3))
                        prog.append((op, {"H": target_h, "W": target_w, "pad_value": 0}))
                    else:  # crop_bbox
                        prog.append((op, {}))
                    continue
            
            # Otherwise use regular random program generation
            regular_prog = self.dsl.random_program(max_len=1, num_colors=self.num_colors, 
                                        H=H_in, W=W_in)
            if regular_prog:
                # Convert DSLProgram to list of tuples
                prog.extend(list(zip(regular_prog.ops, regular_prog.params)))
        
        return prog if prog else [("rotate90", {})]  # Fallback

    def run(self, examples: List[Tuple[np.ndarray, np.ndarray]], test_input: np.ndarray, expect_symmetry=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        constr = ARCGridConstraints(expect_symmetry=expect_symmetry)

        # Call size oracle before candidate sampling
        H_out, W_out, reason = predict_size(examples, test_input)
        if hasattr(self, 'verbose') and self.verbose:
            print(f"[SIZE_ORACLE] Predicted output size: {H_out}x{W_out}, reason: {reason}")

        cands = []
        candidates_checked = 0
        shape_feasible_count = 0
        first_feasible_at = None
        
        # Bias program generation towards size-compatible operations
        H_in, W_in = test_input.shape
        
        for _ in range(self.candidates):
            candidates_checked += 1
            
            # Generate size-aware program with bias towards operations that could produce target size
            prog = self._generate_size_aware_program(H_in, W_in, H_out, W_out, reason)
            
            # Check if program produces expected output size on test input
            test_pred = self.execute_program(test_input, prog)
            if test_pred.shape == (H_out, W_out):
                shape_feasible_count += 1
                if first_feasible_at is None:
                    first_feasible_at = candidates_checked
                    if hasattr(self, 'verbose') and self.verbose:
                        print(f"[SHAPE_FEASIBLE] Found first shape-feasible candidate after {candidates_checked} attempts")
            
            losses = []
            sym_scores = []
            for (gin, gout) in examples:
                pred = self.execute_program(gin, prog)
                if pred.shape != gout.shape:
                    loss = 1e6
                else:
                    loss = float((pred != gout).sum())
                losses.append(loss)
                if expect_symmetry is not None:
                    if expect_symmetry == 'h':
                        sym = float((pred == pred[:, ::-1]).mean())
                    elif expect_symmetry == 'v':
                        sym = float((pred == pred[::-1, :]).mean())
                    else:
                        sym = 0.5
                else:
                    sym = 0.5
                sym_scores.append(sym)

            mean_loss = float(np.mean(losses))
            simp = 1.0 / (1 + len(prog))
            sym  = float(np.mean(sym_scores))
            
            # Add size compatibility score
            size_compat = 1.0 if test_pred.shape == (H_out, W_out) else 0.0

            cands.append({"program": prog, "loss": mean_loss, "simplicity": simp, "symmetry": sym, "size_compat": size_compat})

        winner = WeightedCollapse.collapse(cands)
        pred = self.execute_program(test_input, winner["program"])

        if self.use_ebr:
            logits = grid_to_logits(pred, self.num_colors, self.deterministic_logits)
            constr2 = ARCGridConstraints(expect_symmetry=expect_symmetry)
            
            # Convert logits to proper tokens for phi metrics: [B, C, H, W] -> [B, H*W, C]
            B, C, H, W = logits.shape
            soft_tokens = torch.softmax(logits, dim=1).permute(0,2,3,1).reshape(B, H*W, C)
            
            # Compute priors using one-hot channel representation  
            phi = phi_synergy_features(soft_tokens, parts=2)
            kappa = kappa_floor(soft_tokens, H, W)
            cge = cge_boost(soft_tokens)
            priors = {"phi": float(phi), "kappa": float(kappa), "cge": float(cge), "hodge": 0.0}
            
            if hasattr(self, 'verbose') and self.verbose:
                print(f"[PRIORS] phi={float(phi):.4f}, kappa={float(kappa):.4f}, cge={float(cge):.4f}")
            
            logits_ref = self.ebr(logits, constr2, priors)
            pred = logits_to_grid(logits_ref)

        info = {
            "winner": winner, 
            "num_candidates": len(cands), 
            "candidates": cands,
            "predicted_size": (H_out, W_out),
            "size_reason": reason,
            "candidates_to_first_feasible": first_feasible_at if first_feasible_at is not None else candidates_checked,
            "total_shape_feasible": shape_feasible_count
        }
        return pred, info
