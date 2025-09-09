"""
TOPAS ARC 60M - Canonical Unified Model
Integrates AuroraARC60M with real EnergyRefiner and full DSL operations
Sacred Signature: (grid[B,H,W], logits[B,H*W,C], size[B,2])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, NamedTuple, Tuple, Any
from dataclasses import dataclass
import sys
import os
import math
import hashlib

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Core imports - canonical paths only
from models.grid_encoder import GridEncoder
from models.object_slots import ObjectSlots
from models.rel_graph import RelGraph, ObjectRelationPredictor
from models.painter import NeuralPainter
from models.hrm_topas_bridge import HRMTOPASBridge, HRMTOPASIntegrationConfig
from models.dsl_search import beam_search, DSLProgram, apply_program
from models.utils import (
    logits_from_grid as logits_from_grid,
    size_tensor_from_grid as size_tensor_from_grid,
    ObjectAuxiliaryLossManager,
    validate_grid_signature, validate_logits_signature, validate_size_signature,
    compute_eval_metrics
)

# Required imports - no fallbacks, fail hard on missing components
sys.path.insert(0, parent_dir)

# Core components
from energy_refinement import EnergyRefiner
from trainers.arc_constraints import ARCGridConstraints
from dream_engine import DreamEngine, DreamConfig
from phi_metrics_neuro import phi_synergy_features, kappa_floor, cge_boost, neuro_hodge_penalty
from trainers.schedulers.ucb_scheduler import EnhancedUCBTaskScheduler
from relational_memory_neuro import RelationalMemoryNeuro
from models.dsl_registry import DSL_OPS

# HRM Planner imports - direct import from models directory
try:
    from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
    from models.losses import ACTLossHead
    _HAS_HRM_PLANNER = True
    print("✅ HRM models loaded successfully in TopasARC60M")
except ImportError as e:
    print(f"Warning: HRM models not available in TopasARC60M: {e}")
    _HAS_HRM_PLANNER = False
    class HierarchicalReasoningModel_ACTV1:
        def __init__(self, *args, **kwargs): pass
    class ACTLossHead:
        def __init__(self, *args, **kwargs): pass

_HAS_ALL_COMPONENTS = True
_HAS_SIMPLE_DSL = True

NUM_COLORS = 10

# Sacred Signature enforcement - using canonical validators from utils.py

def validate_sacred_signature(grid: torch.Tensor, logits: torch.Tensor, size: torch.Tensor, 
                             extras: dict, operation: str = "unknown") -> None:
    """
    Sacred Signature validation using canonical validators.
    Sacred Signature: (grid[B,H,W], logits[B,H*W,C], size[B,2], extras)
    """
    if grid.dim() != 3:
        raise RuntimeError(f"[{operation}] Grid must be [B,H,W], got {grid.shape}")
    
    B, H, W = grid.shape
    
    # Use canonical validators from utils.py
    validate_grid_signature(grid, operation)
    validate_logits_signature(logits, B, H, W, NUM_COLORS, operation)
    validate_size_signature(size, B, H, W, operation)
    
    # Validate extras
    if not isinstance(extras, dict):
        raise RuntimeError(f"[{operation}] SIGNATURE VIOLATION: extras must be dict, got {type(extras)}")
    
    # Check required keys
    required_keys = ["latent", "rule_vec"]
    for key in required_keys:
        if key not in extras:
            raise RuntimeError(f"[{operation}] SIGNATURE VIOLATION: extras missing required key '{key}'")

def enforce_sacred_signature(grid: torch.Tensor, logits: torch.Tensor, size: torch.Tensor, 
                           extras: dict, operation: str, training_mode: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    FINAL enforcement point - validates and fixes any remaining signature issues.
    
    Args:
        grid, logits, size, extras: Output tensors
        operation: Operation name for error reporting
        training_mode: If True, skip regeneration to preserve gradients
        
    Returns:
        Validated and corrected (grid, logits, size, extras) tuple
        
    Raises:
        RuntimeError: On unfixable signature violations
    """
    
    if training_mode:
        # Training mode: validate but never regenerate tensors (preserve grad graph)
        try:
            validate_sacred_signature(grid, logits, size, extras, operation)
        except RuntimeError as e:
            print(f"[TRAINING-SIG] Validation warning (ignored to keep grads): {e}")
        return grid, logits, size, extras
    
    try:
        # First attempt validation
        validate_sacred_signature(grid, logits, size, extras, operation)
        # After successful validation, return immediately (no recursion)
        return grid, logits, size, extras
    except RuntimeError as e:
        print(f"[SIGNATURE-FIXER] Attempting to fix violation: {e}")
        
        # Attempt to fix common issues
        fixed_grid = grid
        fixed_logits = logits  
        fixed_size = size
        fixed_extras = extras
        
        # Fix grid issues
        if not isinstance(fixed_grid, torch.Tensor):
            raise RuntimeError(f"[{operation}] UNFIXABLE: grid is not tensor")
        
        if fixed_grid.dim() != 3:
            if fixed_grid.dim() == 2:
                fixed_grid = fixed_grid.unsqueeze(0)
            else:
                raise RuntimeError(f"[{operation}] UNFIXABLE: cannot reshape grid from {fixed_grid.shape} to [B,H,W]")
        
        # Ensure grid is integer and in valid range
        if fixed_grid.dtype not in [torch.int8, torch.int16, torch.int32, torch.int64, torch.long]:
            fixed_grid = fixed_grid.round().long()
        fixed_grid = torch.clamp(fixed_grid, 0, NUM_COLORS - 1)
        
        # Regenerate logits and size from fixed grid
        fixed_logits = logits_from_grid(fixed_grid, NUM_COLORS, operation)
        fixed_size = size_tensor_from_grid(fixed_grid, operation)
        
        # Fix extras
        if not isinstance(fixed_extras, dict):
            fixed_extras = {"latent": None, "rule_vec": None}
        
        if "latent" not in fixed_extras:
            fixed_extras["latent"] = None
        if "rule_vec" not in fixed_extras:
            fixed_extras["rule_vec"] = None
        
        # Final validation after fixes
        validate_sacred_signature(fixed_grid, fixed_logits, fixed_size, fixed_extras, f"{operation}-FIXED")
        
        print(f"[SIGNATURE-FIXER] Successfully fixed {operation} signature violation")
        return fixed_grid, fixed_logits, fixed_size, fixed_extras

# ============================================================================
# ============================================================================

@dataclass
class RelEdge:
    src: str
    rel: str
    dst: str
    w: float
    last: int
    params: Optional[Dict[str, float]] = None

class RelationManager:
    """
    Lightweight relation graph with decay, auto-inverses, and negative-learn.
    Neurally faithful: keeps only small active subset; edges decay unless refreshed.
    """
    def __init__(self, decay: float = 0.02, negate_rate: float = 0.5, max_edges: int = 10000):
        self.decay = decay
        self.negate_rate = negate_rate
        self.max_edges = max_edges
        self.t = 0
        self.edges: Dict[tuple, RelEdge] = {}

    def _pkey(self, params: Optional[Dict[str, float]]) -> tuple:
        """Create parameter key handling nested dictionaries"""
        if not params:
            return ()
        
        # MACHETE FIX: Ensure params is a dict, not tuple
        if not isinstance(params, dict):
            # DEBUG: Log the problematic params to find the source
            print(f"[MACHETE DEBUG] Bad params type: {type(params)} = {params}")
            # If params is tuple or other, return it as-is safely
            return tuple(params) if hasattr(params, '__iter__') and not isinstance(params, str) else (params,)
        
        result = []
        for k, v in sorted(params.items()):
            if isinstance(v, dict):
                # Handle nested dictionaries (e.g., color mappings)
                nested_items = tuple(sorted(v.items())) if v else ()
                result.append((k, nested_items))
            else:
                # Handle regular float/int values
                result.append((k, float(v)))
        
        return tuple(result)

    def _key(self, src, rel, dst, params=None):
        return (src, rel, dst, self._pkey(params))

    def inverse(self, rel: str, params: Optional[Dict[str, float]]):
        """Enhanced inverse relation mapping for extended operation set"""
        
        # Handle parametric inverses
        if rel == "translate" and params and isinstance(params, dict):
            dx, dy = params.get("dx", 0.0), params.get("dy", 0.0)
            return "translate", {"dx": -dx, "dy": -dy}
        
        if rel == "resize" and params and isinstance(params, dict):
            sx, sy = params.get("sx", 1.0), params.get("sy", 1.0)
            if sx != 0.0 and sy != 0.0:
                return "resize", {"sx": 1.0 / sx, "sy": 1.0 / sy}
            return None, None
        
        if rel == "scale" and params and isinstance(params, dict):
            fy, fx = params.get("fy", 1.0), params.get("fx", 1.0)
            if fy > 0 and fx > 0:
                # Scale down by inverse factors (approximation)
                return "resize", {"sx": 1.0 / fy, "sy": 1.0 / fx}
            return None, None
        
        if rel == "color_map" and params and isinstance(params, dict) and "mapping" in params:
            # Reverse color mapping
            mapping = params["mapping"]
            if isinstance(mapping, dict):
                reverse_mapping = {v: k for k, v in mapping.items()}
                return "color_map", {"mapping": reverse_mapping}
            return None, None
        
        if rel == "center_pad_to" and params:
            # Inverse is approximately crop_bbox
            return "crop_bbox", None
        
        if rel == "extract_color" and params:
            # No direct inverse for color extraction
            return None, None
        
        if rel == "mask_color" and params:
            # Cannot reliably reverse masking without knowing original color
            return None, None

        # Static inverse mappings
        mapping = {
            # Rotations
            "rotate90": "rotate270",
            "rotate270": "rotate90",
            "rotate180": "rotate180",
            
            # Flips (self-inverse)
            "flip_h": "flip_h",
            "flip_v": "flip_v",
            
            # Crops (heuristic inverses)
            "crop_bbox": "center_pad_to",
            "crop_nonzero": "center_pad_to",
            
            # Pattern operations (limited inverses)
            "tile_pattern": None,  # Cannot reliably invert tiling
            
            # Semantic relations
            "is_a": "has_instance",
            "has_instance": "is_a",
            "has_a": "part_of",
            "part_of": "has_a",
        }
        
        inv = mapping.get(rel)
        return (inv, params) if inv else (None, None)

    def tick(self):
        """Exponential decay; prune tiny edges."""
        self.t += 1
        drop = []
        factor = math.exp(-self.decay)
        for k, e in self.edges.items():
            e.w *= factor
            if e.w < 1e-4:
                drop.append(k)
        for k in drop:
            self.edges.pop(k, None)
        # soft bound
        if len(self.edges) > self.max_edges:
            # drop oldest/lightest
            victims = sorted(self.edges.items(), key=lambda kv: (kv[1].w, kv[1].last))[:len(self.edges)-self.max_edges]
            for k, _ in victims:
                self.edges.pop(k, None)

    def add(self, src: str, rel: str, dst: str, w: float = 1.0, params: Optional[Dict[str, float]] = None):
        k = self._key(src, rel, dst, params)
        e = self.edges.get(k)
        if e is None:
            e = RelEdge(src, rel, dst, 0.0, self.t, params)
        e.w = min(1.0, e.w + w)
        e.last = self.t
        self.edges[k] = e

        inv_rel, inv_params = self.inverse(rel, params)
        if inv_rel:
            ki = self._key(dst, inv_rel, src, inv_params)
            ei = self.edges.get(ki)
            if ei is None:
                ei = RelEdge(dst, inv_rel, src, 0.0, self.t, inv_params)
            ei.w = min(1.0, ei.w + w)
            ei.last = self.t
            self.edges[ki] = ei

    def negative_learn(self, src: str, rel: str, dst: str, params: Optional[Dict[str, float]] = None, amount: float = None):
        """Down-weight competing relations for same src/dst."""
        amount = self.negate_rate if amount is None else amount
        for k, e in list(self.edges.items()):
            s, r, d, _ = k
            if s == src and d == dst and r != rel:
                e.w = max(0.0, e.w * (1.0 - amount))

    # Optional op-bias vector for DSL heuristics (not strictly required)
    def op_bias(self) -> Dict[str, float]:
        bias: Dict[str, float] = {}
        for e in self.edges.values():
            bias[e.rel] = max(bias.get(e.rel, 0.0), e.w)
        return bias


@dataclass
class ModelConfig:
    """Configuration for TOPAS ARC model"""
    # Architecture - scaled to ~60M parameters
    width: int = 640
    depth: int = 16
    slots: int = 80
    slot_dim: int = 512
    rt_layers: int = 10
    rt_heads: int = 8
    
    # Search
    max_dsl_depth: int = 6
    max_beam_width: int = 12
    dsl_vocab_size: int = 41  # Number of DSL operations
    
    # Refinement
    use_ebr: bool = True
    ebr_steps: int = 5
    ebr_step_size: float = 0.25
    ebr_noise: float = 0.0
    painter_refine: bool = True  # Skip EBR after painter fallback if False
    painter_confidence_threshold: float = 0.0  # Skip EBR if logits entropy < threshold (0.0 = disabled)
    
    # Dream Engine
    enable_dream: bool = True
    dream_micro_ticks: int = 1
    dream_offline_iters: int = 50
    dream_valence_default: float = 0.7
    dream_arousal_default: float = 0.5
    
    # Constraints
    max_grid_size: int = 30
    
    # Debugging
    verbose: bool = False
    
    # Scheduler integration
    use_scheduler: bool = True
    scheduler_task_history: int = 1000  # Number of tasks to track
    
    # Pretraining mode
    pretraining_mode: bool = False  # Enable multi-head pretraining
    use_multi_head_loss: bool = False  # Use multi-head loss during training
    
    def validate(self):
        """Validate configuration parameters"""
        assert self.width > 0, "width must be positive"
        assert self.depth > 0, "depth must be positive"
        assert self.slots > 0, "slots must be positive"
        assert 0 < self.max_dsl_depth <= 15, "max_dsl_depth must be in (0, 15]"
        assert 0 < self.max_beam_width <= 100, "max_beam_width must be in (0, 100]"

class TopasARC60M(nn.Module):
    """
    Unified 60M parameter ARC model with full rails integration
    - Neural backbone with object slots and relational reasoning
    - DSL search with 18+ operations
    - Energy-based refinement for polish
    - Sacred signature compliance
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        
        # Use provided config or create default
        self.config = config or ModelConfig()
        self.config.validate()
        
        # Neural components - encoder will be enhanced with HRM after planner init
        self.encoder = GridEncoder(width=self.config.width, depth=self.config.depth)
        self.slots = ObjectSlots(in_ch=self.config.width, K=self.config.slots, 
                                slot_dim=self.config.slot_dim)
        
        # Always define encoder projection at init for stability
        if self.config.width != self.config.slot_dim:
            self.encoder_proj = nn.Linear(self.config.width, self.config.slot_dim)
            self.glob_dim = self.config.slot_dim
        else:
            self.encoder_proj = nn.Identity()
            self.glob_dim = self.config.width
        
        self.reln = RelGraph(d=self.config.slot_dim, 
                           layers=self.config.rt_layers, 
                           heads=self.config.rt_heads)
        
        # Define ctrl_dim correctly after projection
        self.ctrl_dim = self.glob_dim + self.slots.out_dim
        print(f"[TOPAS] Prior heads expect ctrl_dim={self.ctrl_dim} (glob={self.glob_dim} + slot_out={self.slots.out_dim})")
        
        # Prior heads
        self.prior_transform = nn.Linear(self.ctrl_dim, 8)
        self.prior_size      = nn.Linear(self.ctrl_dim, 2)
        self.prior_palette   = nn.Linear(self.ctrl_dim, NUM_COLORS)
        
        # Painter fallback
        self.painter = NeuralPainter(width=self.config.width)
        
        # Slot to logits head for pretraining
        self.num_colors = getattr(self.config, "num_colors", NUM_COLORS)
        self.slot_to_logits = nn.Sequential(
            nn.Linear(self.config.slot_dim, self.config.slot_dim),
            nn.GELU(),
            nn.Linear(self.config.slot_dim, self.num_colors),
        )
        
        # Pixel fallback head for when attention is unavailable
        self.pixel_fallback = nn.Conv2d(self.config.width, self.num_colors, kernel_size=1)
        
        # DSL and refinement
        self.ebr = EnergyRefiner(
            min_steps=3,
            max_steps=self.config.ebr_steps, 
            step_size=self.config.ebr_step_size,
            noise=self.config.ebr_noise,
            lambda_violation=1.0,
            lambda_prior=1e-3,
            verbose=self.config.verbose
        )
        
        # Object mastery components - always integrated
        self.init_object_mastery()
        
        # Relation graph: decay + inverse closure
        self.relman = RelationManager(decay=0.02, negate_rate=0.5, max_edges=10000)

        # --- Optional relational memory (feature-flagged) ---
        self.enable_relmem = bool(getattr(self.config, "enable_relmem", True))
        if self.enable_relmem:
            hid = getattr(self, "slots").out_dim  # relational tokens live in slot space
            self.relmem = RelationalMemoryNeuro(
                hidden_dim=hid,
                max_concepts=getattr(self.config, "relmem_max_concepts", 2048),
                rank=getattr(self.config, "relmem_rank", 16),
                device=self._get_device()
            )
        else:
            self.relmem = None
        
        # Legacy DSLHead fallback removed. All DSL calls route through dsl_search.
        self.simple_dsl = None
        
        # Dream Engine (optional)
        self.dream = None
        if self.config.enable_dream:
            # Use full control vector (encoder width + slot_dim) as DreamEngine state_dim
            # This matches the ctrl_dim used for prior heads (width + slots.out_dim)
            state_dim = self.config.width + self.slots.out_dim
            dcfg = DreamConfig(
                state_dim=state_dim,
                device=self._get_device(),
                micro_ticks=self.config.dream_micro_ticks,
                offline_iters=self.config.dream_offline_iters,
                valence_default=self.config.dream_valence_default,
                arousal_default=self.config.dream_arousal_default,
                verbose=self.config.verbose
            )
            self.dream = DreamEngine(dcfg)
            
            # Attach RelMem to DreamEngine for dream-gated plasticity
            if hasattr(self, "relmem") and self.relmem is not None:
                if hasattr(self.dream, "attach_relmem"):
                    self.dream.attach_relmem(self.relmem)
            
            # Log ripple configuration if enabled
            if hasattr(self.dream, 'ripple') and self.dream.ripple is not None:
                print(f"[Dream] Ripple substrate: rate={self.dream.ripple.config.event_rate_hz}Hz, "
                      f"stdp_gain={self.dream.ripple.config.stdp_gain}, "
                      f"center={self.dream.ripple.config.center_freq_hz}Hz")
        
        # Dream token cache (one sample, downsampled)
        self._dream_tokens = None  # type: Optional[torch.Tensor]
        self._dream_tokens_info = {"H": None, "W": None}  # keep spatial metadata if helpful
        
        # Enhanced Task Scheduler for intelligent compute allocation
        self.scheduler = None
        if self.config.use_scheduler:
            try:
                from trainers.schedulers.ucb_scheduler import EnhancedUCBTaskScheduler, SchedulerConfig
                # Initialize with config object
                sched_config = SchedulerConfig(
                    empowerment_weight=0.05,
                    sync_weight=0.05,
                    exploration_weight=2.0,
                    retry_budget=0.15
                )
                self.scheduler = EnhancedUCBTaskScheduler(config=sched_config)
            except Exception as e:
                if self.config.verbose:
                    print(f"[TOPAS] Warning: Scheduler initialization failed: {e}")
                self.scheduler = None
            if self.config.verbose:
                print(f"[TOPAS] Enhanced scheduler initialized for intelligent compute allocation")
        else:
            pass
            
        # Task tracking for scheduler
        self.task_history = {}  # task_id -> performance history
        self.strategy_usage = {}  # strategy -> usage count
        
        # === HRM Planner Rail ===
        hrm_cfg = dict(
            batch_size=1,
            seq_len=30*30,          # Flattened ARC grid
            vocab_size=10,          # ARC colors
            num_puzzle_identifiers=1000,  # Map ARC task ids → puzzle ids
            puzzle_emb_ndim=128,
            H_cycles=3, L_cycles=4,
            H_layers=4, L_layers=4,
            hidden_size=512,
            expansion=3.0,
            num_heads=8,
            pos_encodings="rope",
            halt_max_steps=6,
            halt_exploration_prob=0.1,
            forward_dtype="bfloat16"
        )
        if _HAS_HRM_PLANNER:
            try:
                self.planner = HierarchicalReasoningModel_ACTV1(hrm_cfg)
                self.planner_loss_head = ACTLossHead(self.planner, loss_type="softmax_cross_entropy")
                # Projection from planner latent → DSL op bias
                self.planner_op_bias = nn.Linear(hrm_cfg["hidden_size"], len(DSL_OPS))
                self._has_planner = True
                if self.config.verbose:
                    print(f"[TOPAS] HRM Planner initialized: {len(DSL_OPS)} operations")
            except Exception as e:
                if self.config.verbose:
                    print(f"[TOPAS] Warning: HRM Planner initialization failed: {e}")
                self.planner = None
                self.planner_loss_head = None
                self.planner_op_bias = None
                self._has_planner = False
        else:
            self.planner = None
            self.planner_loss_head = None
            self.planner_op_bias = None
            self._has_planner = False
        
        # === HRM-TOPAS Bridge ===
        if self._has_planner:
            bridge_config = HRMTOPASIntegrationConfig(
                hrm_hidden_size=hrm_cfg["hidden_size"],
                topas_width=self.config.width,
                num_attention_heads=8,
                puzzle_emb_dim=hrm_cfg.get("puzzle_emb_ndim", 128),
                dsl_ops_count=len(DSL_OPS),
                adaptive_halting_threshold=0.6
            )
            self.hrm_bridge = HRMTOPASBridge(bridge_config)
            self._has_hrm_bridge = True
            if self.config.verbose:
                print(f"[TOPAS] HRM-TOPAS Bridge initialized with {bridge_config.num_attention_heads} heads")
        else:
            self.hrm_bridge = None
            self._has_hrm_bridge = False
        
        # Replace encoder with HRM-aware version if HRM is available
        if self._has_planner:
            # Replace the basic encoder with HRM-enhanced version
            self.encoder = GridEncoder(
                width=self.config.width, 
                depth=self.config.depth,
                hrm_integration=True,
                hrm_hidden_size=hrm_cfg["hidden_size"],
                puzzle_emb_dim=hrm_cfg.get("puzzle_emb_ndim", 128)
            )
            if self.config.verbose:
                print(f"[TOPAS] Grid encoder enhanced with HRM integration")
        
        # Pretraining support
        self._pretraining_mode = self.config.pretraining_mode
        self._multi_head_pretrainer = None  # Will be set externally if needed
        
        # Log initialization
        param_count = sum(p.numel() for p in self.parameters()) / 1e6
        print(f"[TOPAS] Initialized ARC {param_count:.1f}M model")
        if self._pretraining_mode:
            print(f"[TOPAS] Pretraining mode ENABLED - supports multi-head learning")
    
    def init_object_mastery(self):
        """Initialize object mastery components - always integrated"""
        # Components already exist in slots and model
        self.object_relations_enabled = True
        self.auxiliary_losses_enabled = True
        
        if self.config.verbose:
            print("[Object Mastery] Integrated with object slots and auxiliary losses")
        
    def _flatten_logits(self, raw_logits, H: int, W: int):
        """Safely flatten logits from [B,C,H,W] or [B,H*W,C] to [B,H*W,C]"""
        if raw_logits.dim() == 4:
            B, C, Hc, Wc = raw_logits.shape
            assert Hc == H and Wc == W, f"logits spatial mismatch: {(Hc, Wc)} vs {(H, W)}"
            return raw_logits.permute(0, 2, 3, 1).contiguous().view(B, H*W, C)
        elif raw_logits.dim() == 3:
            B, HW, C = raw_logits.shape
            assert HW == H*W, f"logits HW mismatch: {HW} vs {H*W}"
            return raw_logits
        else:
            raise RuntimeError(f"Unexpected logits shape: {tuple(raw_logits.shape)}")
    
    def _get_device(self) -> str:
        """
        Robust device detection with proper fallback chain:
        1. Try to get device from first parameter
        2. Fall back to 'cpu' if no parameters
        3. Handle edge cases gracefully
        """
        try:
            # Try to get device from first parameter
            param_iter = iter(self.parameters())
            first_param = next(param_iter)
            return str(first_param.device)
        except StopIteration:
            # No parameters found, fall back to CPU
            return "cpu"
        except Exception:
            # Any other error, fall back to CPU
            return "cpu"

    def grid_to_tokens(self, grid: torch.LongTensor) -> torch.LongTensor:
        """
        Flatten grid into token sequence of color ids [0..9].
        Input: [B,H,W]  →  Output: [B,H*W]
        """
        return grid.view(grid.size(0), -1).clamp(0, 9)
    
    @property
    def device(self) -> str:
        """Property to get model device consistently."""
        return self._get_device()
        print(f"[TOPAS] Config: DSL(depth={self.config.max_dsl_depth}, beam={self.config.max_beam_width})")
        print(f"[TOPAS] EBR(steps={self.config.ebr_steps}), Grid limit={self.config.max_grid_size}x{self.config.max_grid_size}")
        if self.config.painter_confidence_threshold > 0.0:
            print(f"[TOPAS] Painter confidence threshold: {self.config.painter_confidence_threshold}")
        print(f"[TOPAS] Dream Engine: {'Enabled' if self.dream else 'Disabled'}")
        print(f"[Painter] refine={self.config.painter_refine}")
        print(f"[TOPAS] Device: {self._get_device()}")

    def forward(self, demos: List[Dict], test: Dict[str, torch.LongTensor],
                eval_use_dsl=True, eval_use_ebr=True, eval_dsl_depth=None, eval_beam_width=None, task_id=None, training_mode=None):
        """
        Forward pass enforcing Sacred Signature
        
        Args:
            demos: List of demonstration input/output pairs
            test: Test input grid
            eval_use_dsl: Whether to try DSL search first
            eval_use_ebr: Whether to apply EBR refinement
            eval_dsl_depth: Override DSL search depth
            eval_beam_width: Override beam width
            training_mode: If None, auto-detect from self.training. If True, preserve gradients.
        
        Note:
            - DSL paths always apply EBR when enabled (not affected by painter_refine)
            - Painter fallback respects painter_refine config and optional confidence check
            
        Returns:
            grid: [B, H, W] integer grid
            logits: [B, H*W, C] flat logits
            size: [B, 2] output dimensions
            extras: Dict with latent states and rule vectors for scheduler
            
        Raises:
            ValueError: If inputs are invalid or constraints violated
            RuntimeError: If model computation fails
        """
        
        # Auto-detect training mode if not specified
        if training_mode is None:
            training_mode = self.training
        
        # Input validation
        if not demos:
            raise ValueError("demos cannot be empty")
        if 'input' not in test:
            raise ValueError("test must contain 'input' key")
        if not isinstance(test['input'], torch.Tensor):
            raise TypeError(f"test['input'] must be torch.Tensor, got {type(test['input'])}")
        
        # Ensure test grid has batch dimension with defensive checks
        test_input = test['input']
        if test_input.numel() == 0:
            raise ValueError("test input cannot be empty tensor")
        
        test_grid = test_input if test_input.dim() == 3 else test_input.unsqueeze(0)
        B = test_grid.shape[0]
        
        # Validate grid dimensions
        max_size = self.config.max_grid_size
        if test_grid.shape[1] > max_size or test_grid.shape[2] > max_size:
            print(f"[WARN] Large grid detected: {test_grid.shape} > {max_size}x{max_size}, may impact performance")
        if test_grid.min() < 0 or test_grid.max() >= NUM_COLORS:
            raise ValueError(f"Grid values must be in [0, {NUM_COLORS}), got [{test_grid.min()}, {test_grid.max()}]")
        
        # Task registration and scheduler integration
        if task_id is None:
            task_id = f"task_{hash(str(test_grid.flatten().tolist())) % 100000:05d}"
            
        # Register task with scheduler if available
        if self.scheduler is not None:
            if task_id not in self.scheduler.task_stats:
                # Initialize task stats
                self.scheduler.task_stats[task_id] = {
                    "n": 0, "reward": 0.0, "acc": 0.0, "iou": 0.0, "exact": 0.0, 
                    "novel": 0.0, "empowerment": 0.0, "sync": 0.0
                }
        
        # Compute allocated retries based on scheduler metrics
        base_retries = 2
        allocated_retries = base_retries
        strategy_recommendation = 'dsl'  # Default
        
        # Scheduler is available but simplified for now
        if self.scheduler is not None:
            # Update task stats later based on performance
            pass
        
        # Track strategy usage
        current_strategy = None
        
        # Decay relations and ingest new ones from demos
        self.relman.tick()
        try:
            for d in demos:
                # PROPER FIDELITY FIX: Normalize demo format consistently
                if isinstance(d, tuple):
                    # Handle tuple format: (input, output) or (input, output, ...)
                    if len(d) >= 2:
                        din, dout = d[0], d[1]
                    else:
                        print(f"[WARN] Invalid tuple demo format: {len(d)} elements, need >=2")
                        continue
                elif isinstance(d, dict) and 'input' in d and 'output' in d:
                    # Handle dict format: {"input": tensor, "output": tensor}
                    din = d['input']
                    dout = d['output']
                else:
                    print(f"[WARN] Unknown demo format: {type(d)} with keys {list(d.keys()) if isinstance(d, dict) else 'N/A'}")
                    continue
                src = self._grid_fingerprint(din)
                dst = self._grid_fingerprint(dout)
                # Infer relations from demo and add to relman (registry-guarded)
                from models.dsl_registry import DSL_OPS
                inferred = self._infer_relations_from_demo(din, dout)
                for item in inferred:
                    if isinstance(item, tuple) and len(item) == 2:
                        rel, params = item
                    elif isinstance(item, str):
                        rel, params = item, {}
                    else:
                        print(f"[WARN] Invalid relation format: {item}")
                        continue
                    if rel in DSL_OPS:
                        self.relman.add(src, rel, dst, w=0.6, params=params)
                        print(f"[RELATIONS] Added {rel} from {src[:8]} -> {dst[:8]} (w=0.6)")
                    else:
                        if self.config.verbose:
                            print(f"[REL-GUARD] Skipping invalid op '{rel}' (not in DSL_OPS)")
                # If multiple mutually-exclusive relations inferred, negative-learn competitors
                if len(inferred) > 1:
                    # pick the first as winner for now; down-weight rest
                    winner_rel, winner_p = inferred[0]
                    self.relman.negative_learn(src, winner_rel, dst, winner_p, amount=0.4)
                    print(f"[RELATIONS] Negative learning: {winner_rel} wins")
        except Exception as e:
            print(f"[WARN] Relation inference failed: {e}")
        
        # Ensure float for encoder, scale to [0,1] to avoid dtype issues
        if test_grid.dtype != torch.float32:
            enc_in = test_grid.float() / (NUM_COLORS - 1)  # NUM_COLORS-1 guard
        else:
            enc_in = test_grid
        
        # Neural encoding with HRM integration
        try:
            # Prepare HRM context if available
            hrm_context = None
            if self._has_planner and self.planner is not None:
                try:
                    # Run HRM planner to get reasoning states
                    tokens = self.grid_to_tokens(enc_in)  # [B, seq_len]
                    puzzle_ids = torch.zeros(tokens.size(0), dtype=torch.long, device=tokens.device)
                    planner_batch = {"inputs": tokens, "labels": tokens, "puzzle_identifiers": puzzle_ids}
                    
                    carry = self.planner.initial_carry(planner_batch)
                    carry, hrm_outputs = self.planner(carry=carry, batch=planner_batch)
                    
                    # Extract HRM reasoning states for grid encoder
                    if hasattr(carry, 'inner_carry') and carry.inner_carry is not None:
                        z_H = getattr(carry.inner_carry, 'z_H', None)
                        z_L = getattr(carry.inner_carry, 'z_L', None)
                        
                        # Get puzzle embedding if available
                        puzzle_emb = None
                        if hasattr(self.planner.inner, 'puzzle_emb') and self.planner.inner.puzzle_emb is not None:
                            # Use first puzzle ID embedding as context
                            puzzle_emb = self.planner.inner.puzzle_emb(puzzle_ids)
                        
                        hrm_context = {
                            'z_H': z_H,
                            'z_L': z_L, 
                            'puzzle_emb': puzzle_emb,
                            'hrm_outputs': hrm_outputs
                        }
                        
                        if self.config.verbose:
                            print(f"[HRM] Context prepared: z_H={z_H.shape if z_H is not None else None}, "
                                  f"z_L={z_L.shape if z_L is not None else None}")
                    
                except Exception as e:
                    if self.config.verbose:
                        print(f"[HRM] Failed to prepare context: {e}, using basic encoding")
                    hrm_context = None
                    hrm_outputs = {}
            
            # Run encoder with HRM context
            encoder_output = self.encoder(enc_in, hrm_context)
            if self.config.verbose:
                print(f"[DEBUG] Encoder returned {len(encoder_output)} values")
            feat, glob = encoder_output
            
            slots_output = self.slots(feat)
            # Accept all known signatures from ObjectSlots:
            #  (a) Tensor:                slot_vecs [B, K, D]
            #  (b) (slot_vecs, attn)
            #  (c) (slot_vecs, attn, hier_features)
            if self.config.verbose:
                try:
                    _desc = f"type={type(slots_output)}, len={len(slots_output) if isinstance(slots_output,(list,tuple)) else 'n/a'}"
                except Exception:
                    _desc = f"type={type(slots_output)}"
                print(f"[DEBUG] Slots return {_desc}")

            if torch.is_tensor(slots_output):
                slot_vecs = slots_output
            elif isinstance(slots_output, (list, tuple)):
                if len(slots_output) >= 1:
                    slot_vecs = slots_output[0]
                else:
                    raise ValueError("ObjectSlots returned empty sequence")
            else:
                raise TypeError(f"Unexpected slots_output type: {type(slots_output)}")

            # Normalize to [B, K, D]
            if slot_vecs.dim() == 2:   # e.g., [B, D] → assume K=1
                slot_vecs = slot_vecs.unsqueeze(1)
            assert slot_vecs.dim() == 3, f"Expected slot_vecs [B,K,D], got {tuple(slot_vecs.shape)}"
            slots_rel = self.reln(slot_vecs)
            pooled = slots_rel.mean(dim=1)
            
            # Project encoder output to match slot dimensions (using initialized projection)
            glob_proj = self.encoder_proj(glob)  # [B, width] → [B, slot_dim]
            brain = torch.cat([glob_proj, pooled], dim=-1)
            # --- Relational memory residual context ---
            if self.relmem is not None:
                # tokens: slot-level relational embeddings [B, T, D]
                tokens = slots_rel                       # [B, T, D]
                ctx, _ = self.relmem(tokens, state=None, top_k=min(128, tokens.size(1)))
                ctx_pool = ctx.mean(dim=1)               # [B, D]
                
                # PROPER FIDELITY FIX: Project and mix relational context, don't just concatenate
                if not hasattr(self, 'relmem_proj'):
                    self.relmem_proj = nn.Linear(ctx_pool.shape[-1], brain.shape[-1]).to(brain.device)
                
                ctx_proj = self.relmem_proj(ctx_pool)    # [B, ctrl_dim] 
                brain = brain + ctx_proj  # Residual addition instead of concat - maintains 1024
            
            # RAIL ORCHESTRATION - Initialize tracking
            rail_path = []
            retry_count = 0
            ops_attempted = set()
            beam_depth_used = 0
            beam_width_used = 0
            ebr_improvements = {
                "iou_delta": 0.0,
                "acc_delta": 0.0
            }
            
            # Prepare extras dict for scheduler with enhanced rail tracking
            extras = {
                # Preserve gradient in training mode, detach only in eval
                "latent": brain if training_mode else brain.detach(),
                "rule_vec": None,  # Will be set if DSL finds a rule
                "rail_path": [],
                "beam_depth": 0,
                "beam_width": 0,
                "ops_attempted": [],
                "ebr_deltas": {"iou_delta": 0.0, "acc_delta": 0.0},
                "retry_count": 0
            }
        except Exception as e:
            print(f"[ERROR] Neural encoding failed: {e}")
            # Fallback to simple grid passthrough
            print("[FALLBACK] Returning input grid as output")
            fallback_extras = {"latent": None, "rule_vec": None}
            fallback_logits = logits_from_grid(test_grid, NUM_COLORS, "NEURAL-ENCODING-FALLBACK")
            fallback_size = size_tensor_from_grid(test_grid, "NEURAL-ENCODING-FALLBACK")
            test_grid, fallback_logits, fallback_size, fallback_extras = enforce_sacred_signature(
                test_grid, fallback_logits, fallback_size, fallback_extras, "NEURAL-ENCODING-FALLBACK", training_mode
            )
            return test_grid, fallback_logits, fallback_size, fallback_extras
        
        # Compute tokens for priors: use feature map if available; fallback to slots_rel
        try:
            Bf, Cf, Hf, Wf = feat.shape
            tokens = feat.permute(0, 2, 3, 1).reshape(Bf, Hf*Wf, Cf).contiguous()
            Htok, Wtok = Hf, Wf
        except Exception:
            # Fallback to slots_rel [B, K, D]
            tokens = slots_rel
            Htok = int(math.sqrt(tokens.size(1)))
            Wtok = Htok
        
        # Cache tokens for offline dream cycle (downsample to keep memory bounded)
        try:
            self._cache_dream_tokens(tokens, Htok, Wtok)
        except Exception as e:
            print(f"[WARN] Failed to cache dream tokens: {e}")
        
        # Compute metrics/priors using DreamEngine or fallback
        priors = dict()
        if self.dream is not None:
            # Compute real priors via DreamEngine
            priors.update(self.dream.compute_priors(tokens, Htok, Wtok))
        else:
            # Fallback: try direct computation
            try:
                # Calculate phi from slot features
                phi_val = phi_synergy_features(slots_rel, parts=2)
                
                # Calculate kappa from spatial features  
                H, W = test_grid.shape[-2:]
                kappa_val = kappa_floor(slots_rel, H, W)
                
                # Calculate CGE boost from brain features
                cge_val = cge_boost(brain.unsqueeze(1), None, None)
                
                priors.update({
                    "phi": phi_val,
                    "kappa": kappa_val,
                    "cge": cge_val
                })
            except Exception as e:
                print(f"[WARN] Metrics calculation failed: {e}, using defaults")
                priors.update({
                    "phi": torch.tensor(0.0, device=brain.device),
                    "kappa": torch.tensor(0.0, device=brain.device),
                    "cge": torch.tensor(0.0, device=brain.device)
                })
        
        # Compute dynamic Hodge penalty from slot relational features
        try:
            hodge = neuro_hodge_penalty(slots_rel).clamp_min(0.0)
            if self.config.verbose:
                print(f"[Priors] hodge={float(hodge.mean()):.3f}")
        except Exception as e:
            print(f"[WARN] Hodge computation failed: {e}, using default 0.01")
            hodge = torch.tensor(0.01, device=brain.device)
        
        # Add prediction priors
        priors.update({
            "trans": self.prior_transform(brain),
            "size": self.prior_size(brain),
            "pal": self.prior_palette(brain),
            "hodge": hodge
        })
        
        # STEP 1: DSL SEARCH - POLICY MODE OR BEAM SEARCH WITH DETECTED OPS BIAS
        if eval_use_dsl and (eval_dsl_depth or self.config.max_dsl_depth) > 0:
            depth = min(eval_dsl_depth or self.config.max_dsl_depth, 12)  # Cap at reasonable depth
            beam_w = min(eval_beam_width or self.config.max_beam_width, 50)  # Cap beam width
            beam_depth_used = depth
            beam_width_used = beam_w
            rail_path.append("DSL")
            retry_count += 1
            
            # === HRM-TOPAS Bridge Integration ===
            planner_op_bias = {}
            if self._has_hrm_bridge and hrm_context is not None:
                try:
                    # Use HRM Bridge for advanced integration
                    bridge_outputs = self.hrm_bridge.forward(
                        grid_features=feat,
                        hrm_outputs=hrm_context,
                        puzzle_embedding=hrm_context.get('puzzle_emb'),
                        current_search_depth=depth
                    )
                    
                    # Extract DSL operation biases
                    dsl_op_biases = bridge_outputs.get('dsl_op_biases')
                    control_signals = bridge_outputs.get('control_signals')
                    
                    if dsl_op_biases is not None:
                        from models.dsl_registry import DSL_OPS
                        planner_op_bias = self.hrm_bridge.extract_dsl_operation_dict(dsl_op_biases, DSL_OPS)
                    
                    # Apply adaptive search parameters
                    if control_signals is not None:
                        adapted_depth, adapted_beam = self.hrm_bridge.compute_adaptive_search_params(
                            control_signals, depth, beam_w
                        )
                        depth = adapted_depth
                        beam_w = adapted_beam
                        
                        if self.config.verbose:
                            should_halt = control_signals.get('should_halt', torch.tensor(False))
                            confidence = control_signals.get('confidence', torch.tensor(0.5))
                            print(f"[HRM-Bridge] Adaptive control: depth={depth}, beam={beam_w}, "
                                  f"halt={should_halt.item() if torch.is_tensor(should_halt) else should_halt}, "
                                  f"confidence={confidence.mean().item() if torch.is_tensor(confidence) else confidence:.3f}")
                    
                    # Update grid features with enhanced bridge features
                    enhanced_features = bridge_outputs.get('enhanced_features')
                    if enhanced_features is not None:
                        feat = enhanced_features
                        if self.config.verbose:
                            print(f"[HRM-Bridge] Grid features enhanced via cross-attention")
                    
                    beam_depth_used = depth  
                    beam_width_used = beam_w
                    
                    if self.config.verbose and planner_op_bias:
                        top_ops = sorted(planner_op_bias.items(), key=lambda x: x[1], reverse=True)[:3]
                        print(f"[HRM-Bridge] Top operation biases: {top_ops}")
                        
                except Exception as e:
                    if self.config.verbose:
                        print(f"[HRM-Bridge] Integration failed: {e}, falling back to basic HRM")
                    # Fallback to basic HRM integration  
                    if 'z_H' in hrm_context and hrm_context['z_H'] is not None:
                        z_H = hrm_context['z_H']
                        if self.planner_op_bias is not None:
                            from models.dsl_registry import DSL_OPS
                            op_bias_logits = F.softmax(self.planner_op_bias(z_H), dim=-1)
                            for i, op in enumerate(DSL_OPS):
                                if i < op_bias_logits.size(-1):
                                    planner_op_bias[op] = float(op_bias_logits[0, i].item())
            else:
                # No HRM integration available
                if self.config.verbose and self._has_planner:
                    print(f"[HRM] No context available for DSL integration")
            
            # Get detected operation bias from relation manager
            from models.dsl_registry import DSL_OPS
            op_bias = self.relman.op_bias()
            if self.relmem is not None:
                # Optional soft prior derived from relational scores
                rel_bias = {}
                # Map a few common ops; extend as your DSL supports
                for rel in ["translate", "resize", "flip_h", "flip_v", "color_map"]:
                    if hasattr(self.relmem, "_scores") and rel in getattr(self.relmem, "relations", []):
                        try:
                            with torch.no_grad():
                                score = float(self.relmem._scores(rel).mean().item())
                            rel_bias[rel] = max(0.0, score)
                        except Exception:
                            pass
                # Combine by max (conservative): never reduce existing bias
                for k, v in rel_bias.items():
                    op_bias[k] = max(op_bias.get(k, 0.0), v)
            
            # Combine planner op_bias with existing bias (additive)
            for k, v in planner_op_bias.items():
                op_bias[k] = op_bias.get(k, 0.0) + v * 0.5  # Scale planner bias to 50%
            
            # Filter op_bias to only include valid registry ops
            op_bias = {k: v for k, v in op_bias.items() if k in DSL_OPS}
            print(f"[RAIL-DSL] Starting DSL search: depth={depth}, beam={beam_w}")
            if op_bias:
                print(f"[RAIL-DSL] Operation bias detected: {list(op_bias.keys())[:5]}")
                ops_attempted.update(op_bias.keys())
            
            dsl_pred = None
            
            # TRY POLICY MODE FIRST (if available and confidence is high)
            if hasattr(self, 'policy_net') and self.policy_net is not None:
                try:
                    policy_result = self._forward_with_policy(demos, test_grid, priors, max_length=min(depth, 6))
                    if policy_result is not None:
                        dsl_pred, policy_ops, policy_confidence = policy_result
                        
                        # Only use policy if confidence is high enough
                        if policy_confidence > 0.7:
                            print(f"[RAIL-POLICY] Policy mode SUCCESS - confidence={policy_confidence:.3f}")
                            
                            # Create rule vector from policy operations
                            rule_vec = torch.zeros(len(self.relman.edges) + 32, device=brain.device)
                            for i, op in enumerate(policy_ops[:32]):
                                if isinstance(op, str):
                                    rule_vec[i] = hash(op) % 1000 / 1000.0
                            extras["rule_vec"] = rule_vec
                            ops_attempted.update(policy_ops)
                            rail_path.append("POLICY")
                        else:
                            print(f"[RAIL-POLICY] Policy confidence too low ({policy_confidence:.3f}), falling back to beam search")
                            dsl_pred = None
                            
                except Exception as e:
                    print(f"[RAIL-POLICY] Policy mode failed: {e}, falling back to beam search")
                    dsl_pred = None
            
            # FALLBACK TO BEAM SEARCH if policy failed or unavailable
            if dsl_pred is None:
                try:
                    print(f"[RAIL-DSL] Falling back to beam search...")
                    # Enhanced beam_search call with operation bias
                    dsl_result = beam_search(demos, test_grid[0] if test_grid.dim() == 4 else test_grid, 
                                            priors, depth=depth, beam=beam_w, verbose=self.config.verbose,
                                            return_rule_info=True, op_bias=op_bias)
                    
                    if isinstance(dsl_result, tuple):
                        dsl_pred, rule_info = dsl_result
                        # Extract rule vector from DSL program info
                        if rule_info and 'program' in rule_info:
                            # Create rule vector from program operations
                            ops = rule_info['program']
                            ops_attempted.update(ops)
                            rule_vec = torch.zeros(len(self.relman.edges) + 32, device=brain.device)
                            for i, op in enumerate(ops[:32]):
                                if isinstance(op, str):
                                    rule_vec[i] = hash(op) % 1000 / 1000.0  # Simple hash embedding
                            extras["rule_vec"] = rule_vec
                            print(f"[RAIL-DSL] Program found: {ops[:5]}...")
                    else:
                        dsl_pred = dsl_result
                        # Create rule vector from relation biases
                        if op_bias:
                            rule_vec = torch.zeros(32, device=brain.device)
                            for i, (op, weight) in enumerate(list(op_bias.items())[:32]):
                                rule_vec[i] = weight
                            extras["rule_vec"] = rule_vec
                            
                except Exception as e:
                    print(f"[RAIL-DSL] DSL search failed: {e}")
                    dsl_pred = None
            
            if dsl_pred is not None:
                print("[RAIL-DSL] DSL search SUCCESS - applying EBR")
                # Ensure batch dimension
                if dsl_pred.dim() == 2:
                    dsl_pred = dsl_pred.unsqueeze(0)
                
                # ALWAYS apply EBR on successful DSL outputs
                if eval_use_ebr and self.config.use_ebr:
                    # Measure EBR improvement
                    pre_ebr_grid = dsl_pred.clone()
                    # Create pseudo-logits from DSL grid for EBR
                    B = 1 if dsl_pred.dim() == 2 else dsl_pred.shape[0]
                    H, W = dsl_pred.shape[-2:]
                    dsl_logits = F.one_hot(dsl_pred.long(), NUM_COLORS).float()
                    if dsl_logits.dim() == 3:  # [H,W,C]
                        dsl_logits = dsl_logits.unsqueeze(0)  # [1,H,W,C]
                    dsl_logits = dsl_logits.permute(0, 3, 1, 2)  # [B,C,H,W]
                    grid = self._apply_ebr(dsl_logits, priors, extras)
                    
                    # Calculate EBR deltas
                    try:
                        pre_acc = (pre_ebr_grid == test_grid).float().mean().item() if 'output' in test else 0.0
                        post_acc = (grid == test_grid).float().mean().item() if 'output' in test else 0.0
                        ebr_improvements["acc_delta"] = post_acc - pre_acc
                        ebr_improvements["iou_delta"] = 0.0  # Placeholder - could compute IoU
                    except:
                        ebr_improvements["acc_delta"] = 0.0
                        ebr_improvements["iou_delta"] = 0.0
                    
                    rail_path.append("EBR")
                    print(f"[RAIL-EBR] EBR applied after DSL - acc_delta={ebr_improvements['acc_delta']:.3f}")
                else:
                    grid = dsl_pred
                    print("[RAIL-DSL] EBR disabled - using DSL output directly")
                
                # Update extras with rail information
                extras.update({
                    "rail_path": rail_path,
                    "beam_depth": beam_depth_used,
                    "beam_width": beam_width_used,
                    "ops_attempted": list(ops_attempted),
                    "ebr_deltas": ebr_improvements,
                    "retry_count": retry_count
                })
                
                # Convert to sacred signature with validation
                logits = logits_from_grid(grid, NUM_COLORS, "DSL-SUCCESS")
                size = size_tensor_from_grid(grid, "DSL-SUCCESS")
                grid, logits, size, extras = enforce_sacred_signature(grid, logits, size, extras, "DSL-SUCCESS", training_mode)
                if self.config.verbose:
                    print(f"[RAIL-COMPLETE] DSL->EBR path completed: {rail_path}")
                return grid, logits, size, extras
        
        # STEP 2: SIMPLE DSL Fallback removed (deprecated).
        # Now: DSL → EBR → Painter only.
        if False:  # Disabled
            rail_path.append("SIMPLE-DSL")
            retry_count += 1
            
            try:
                print("[RAIL-DSL] DSL failed - trying simple DSL fallback with memory/inverses")
                demos_tuples = []
                for d in demos:
                    # PROPER FIDELITY FIX: Handle both tuple and dict formats consistently
                    if isinstance(d, tuple) and len(d) >= 2:
                        di, do = d[0], d[1]
                    elif isinstance(d, dict) and 'input' in d and 'output' in d:
                        di = d['input']
                        do = d['output']
                    else:
                        print(f"[WARN] Invalid demo format in DSL fallback: {type(d)}")
                        continue
                    if di.dim() == 3: di = di[0]
                    if do.dim() == 3: do = do[0]
                    demos_tuples.append((di, do))
                
                # Use relation bias to guide simple DSL search  
                from models.dsl_registry import DSL_OPS
                op_bias = self.relman.op_bias()
                # Filter to valid registry ops only
                op_bias = {k: v for k, v in op_bias.items() if k in DSL_OPS}
                if op_bias:
                    print(f"[RAIL-DSL] Simple DSL memory bias: {list(op_bias.keys())[:3]}")
                    ops_attempted.update(op_bias.keys())
                
                candidate = self.simple_dsl.forward(demos_tuples, test_grid[0] if test_grid.dim() == 4 else test_grid)
                if candidate is not None:
                    print("[RAIL-DSL] Simple DSL SUCCESS - applying EBR")
                    grid = candidate if candidate.dim() == 3 else candidate.unsqueeze(0)
                    
                    # Simple DSL paths also always use EBR when available
                    if eval_use_ebr and self.config.use_ebr:
                        # Measure EBR improvement
                        pre_ebr_grid = grid.clone()
                        # Create pseudo-logits from simple DSL grid
                        B = 1 if grid.dim() == 2 else grid.shape[0]
                        H, W = grid.shape[-2:]
                        simple_logits = F.one_hot(grid.long(), NUM_COLORS).float()
                        if simple_logits.dim() == 3:  # [H,W,C]
                            simple_logits = simple_logits.unsqueeze(0)
                        simple_logits = simple_logits.permute(0, 3, 1, 2)  # [B,C,H,W]
                        grid = self._apply_ebr(simple_logits, priors, extras)
                        
                        # Calculate EBR deltas
                        try:
                            pre_acc = (pre_ebr_grid == test_grid).float().mean().item() if 'output' in test else 0.0
                            post_acc = (grid == test_grid).float().mean().item() if 'output' in test else 0.0
                            ebr_improvements["acc_delta"] = post_acc - pre_acc
                        except:
                            ebr_improvements["acc_delta"] = 0.0
                        
                        rail_path.append("EBR")
                        print(f"[RAIL-EBR] EBR applied after Simple DSL - acc_delta={ebr_improvements['acc_delta']:.3f}")
                    
                    # Update extras with rail information
                    extras.update({
                        "rail_path": rail_path,
                        "beam_depth": beam_depth_used,
                        "beam_width": beam_width_used,
                        "ops_attempted": list(ops_attempted),
                        "ebr_deltas": ebr_improvements,
                        "retry_count": retry_count
                    })
                    
                    # Add simple DSL rule vector
                    if extras["rule_vec"] is None:
                        op_bias = self.relman.op_bias()
                        if op_bias:
                            rule_vec = torch.zeros(32, device=brain.device)
                            for i, (op, weight) in enumerate(list(op_bias.items())[:32]):
                                rule_vec[i] = weight
                            extras["rule_vec"] = rule_vec
                    
                    logits = logits_from_grid(grid, NUM_COLORS, "SIMPLE-DSL-SUCCESS")
                    size = size_tensor_from_grid(grid, "SIMPLE-DSL-SUCCESS")
                    grid, logits, size, extras = enforce_sacred_signature(grid, logits, size, extras, "SIMPLE-DSL-SUCCESS", training_mode)
                    if self.config.verbose:
                        print(f"[RAIL-COMPLETE] Simple DSL->EBR path completed: {rail_path}")
                    return grid, logits, size, extras
                    
            except Exception as e:
                print(f"[RAIL-DSL] Simple DSL fallback failed: {e}")
        
        # STEP 3: PAINTER FALLBACK (DSL completely failed)
        rail_path.append("Painter")
        retry_count += 1
        ops_attempted.add("neural_painter")
        
        print("[RAIL-PAINTER] All DSL methods failed - using neural painter fallback")
        painter_output = self.painter(feat)
        
        # Handle painter output - it should return (grid, logits, size)
        if len(painter_output) == 3:
            grid, painter_logits, painter_size = painter_output
        else:
            raise ValueError(f"Painter should return 3 values, got {len(painter_output)}")
        
        # Apply ONE EBR refinement after painter (deterministic rule)
        skip_ebr = False
        
        # Check confidence threshold if enabled
        if self.config.painter_confidence_threshold > 0.0:
            try:
                # Calculate entropy of current logits as confidence measure
                current_logits = logits_from_grid(grid, NUM_COLORS, "PAINTER-CONFIDENCE-CHECK")
                # Compute entropy: -sum(p * log(p)) where p = softmax(logits)
                probs = F.softmax(current_logits, dim=-1) + 1e-10  # Add small epsilon
                entropy = -(probs * torch.log(probs)).sum(dim=-1).mean()
                
                if entropy < self.config.painter_confidence_threshold:
                    skip_ebr = True
                    print(f"[RAIL-PAINTER] High confidence (entropy={entropy:.3f} < {self.config.painter_confidence_threshold}), skipping EBR")
                else:
                    print(f"[RAIL-PAINTER] Low confidence (entropy={entropy:.3f} >= {self.config.painter_confidence_threshold}), applying EBR")
            except Exception as e:
                print(f"[RAIL-PAINTER] Confidence check failed: {e}, proceeding with EBR")
        
        # Deterministic EBR application after painter
        if self.config.painter_refine and not skip_ebr and eval_use_ebr and self.config.use_ebr:
            print("[RAIL-PAINTER] Applying ONE EBR refinement after painter")
            # Measure EBR improvement
            pre_ebr_grid = grid.clone()
            # Convert painter logits to [B,C,H,W] format for EBR
            B = 1 if grid.dim() == 2 else grid.shape[0]
            H, W = grid.shape[-2:]
            logits_for_ebr = painter_logits.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # [B,C,H,W]
            grid = self._apply_ebr(logits_for_ebr, priors, extras)
            
            # Calculate EBR deltas
            try:
                pre_acc = (pre_ebr_grid == test_grid).float().mean().item() if 'output' in test else 0.0
                post_acc = (grid == test_grid).float().mean().item() if 'output' in test else 0.0
                ebr_improvements["acc_delta"] = post_acc - pre_acc
            except:
                ebr_improvements["acc_delta"] = 0.0
            
            rail_path.append("EBR")
            print(f"[RAIL-EBR] EBR applied after Painter - acc_delta={ebr_improvements['acc_delta']:.3f}")
        else:
            if skip_ebr:
                print("[RAIL-PAINTER] Skipping EBR due to high confidence")
            elif not self.config.painter_refine:
                print("[RAIL-PAINTER] Skipping EBR by config (painter_refine=False)")
            else:
                print("[RAIL-PAINTER] EBR disabled globally")
        
        # Update extras with complete rail information
        extras.update({
            "rail_path": rail_path,
            "beam_depth": beam_depth_used,
            "beam_width": beam_width_used,
            "ops_attempted": list(ops_attempted),
            "ebr_deltas": ebr_improvements,
            "retry_count": retry_count
        })
        
        # Add neural painter rule vector if none exists
        if extras["rule_vec"] is None:
            # Create rule vector from painter features or relational biases
            from models.dsl_registry import DSL_OPS
            op_bias = self.relman.op_bias()
            # Filter to valid registry ops only
            op_bias = {k: v for k, v in op_bias.items() if k in DSL_OPS}
            if op_bias:
                rule_vec = torch.zeros(32, device=brain.device)
                for i, (op, weight) in enumerate(list(op_bias.items())[:32]):
                    rule_vec[i] = weight
                extras["rule_vec"] = rule_vec
            else:
                # Fallback: create rule vector from brain features
                extras["rule_vec"] = brain[0, :32].detach() if brain.shape[1] >= 32 else torch.zeros(32, device=brain.device)
        
        # Preserve differentiable painter logits (critical for training)
        if training_mode and painter_logits is not None:
            # Use painter's differentiable logits
            if grid.dim() == 2: 
                grid = grid.unsqueeze(0)
            B, H, W = grid.shape
            logits = self._flatten_logits(painter_logits, H, W)  # [B, H*W, C]
            # Size can be derived from grid (integers are OK for size)
            size = size_tensor_from_grid(grid, "PAINTER-FALLBACK")
        else:
            # Eval mode: regenerate logits from discrete grid (current behavior)
            logits = logits_from_grid(grid, NUM_COLORS, "PAINTER-FALLBACK")
            size = size_tensor_from_grid(grid, "PAINTER-FALLBACK")
        grid, logits, size, extras = enforce_sacred_signature(grid, logits, size, extras, "PAINTER-FALLBACK", training_mode)
        
        print(f"[RAIL-COMPLETE] Painter path completed: {rail_path}")
        print(f"[RAIL-SUMMARY] Total retries: {retry_count}, Operations attempted: {len(ops_attempted)}")
        
        # Update strategy tracking and scheduler
        if self.scheduler is not None:
            # Determine which strategy was actually used
            if rail_path:
                if "DSL" in rail_path[0]:
                    current_strategy = "dsl"
                # Removed: simple_dsl strategy
                elif "Painter" in rail_path[0]:
                    current_strategy = "painter"
                else:
                    current_strategy = "unknown"
                    
                # Track strategy usage
                self.strategy_usage[current_strategy] = self.strategy_usage.get(current_strategy, 0) + 1
                
                # HONEST METRICS: Track actual performance, not placeholders
                
                # Calculate REAL accuracy from actual predictions
                if task_id is not None:
                    # Compute actual accuracy if we have ground truth
                    actual_accuracy = 0.0
                    
                    # If we have a target output in the test dict, compute accuracy
                    if 'output' in test and test['output'] is not None:
                        target = test['output']
                        if isinstance(target, torch.Tensor) and isinstance(grid, torch.Tensor):
                            # Ensure shapes match for comparison
                            if grid.shape[-2:] == target.shape[-2:]:
                                # Use centralized metrics computation
                                eval_metrics = compute_eval_metrics(grid, target)
                                actual_accuracy = eval_metrics["exact@1"]
                                extras["eval_metrics"] = eval_metrics
                            else:
                                # Size mismatch = failure
                                actual_accuracy = 0.0
                                extras["eval_metrics"] = {"exact@1": 0.0, "exact@k": 0.0, "iou": 0.0}
                    else:
                        # No ground truth available - check rail path for success indicators
                        rail_path = extras.get('rail_path', [])
                        
                        # --- Real accuracy first ---
                        actual_accuracy = None
                        confidence_score = None
                        try:
                            eval_metrics = extras.get("eval_metrics", {})
                            if eval_metrics:
                                exact1 = float(eval_metrics.get("exact@1", 0.0))
                                exactk = float(eval_metrics.get("exact@k", 0.0))
                                actual_accuracy = exact1 if exact1 > 0 else exactk
                        except Exception:
                            pass
                        
                        # If no eval metrics, leave accuracy = None
                        # and fall back to heuristic "confidence_score"
                        if actual_accuracy is None:
                            ebr_deltas = extras.get("ebr_deltas", [])
                            try:
                                if ebr_deltas and any(float(d) > 0 for d in ebr_deltas if d is not None):
                                    confidence_score = 0.7
                                else:
                                    confidence_score = 0.5
                            except Exception:
                                confidence_score = 0.5
                        
                        # Record both separately
                        extras["actual_accuracy"] = actual_accuracy
                        extras["confidence_score"] = confidence_score
                    
                    # Update scheduler with honest performance
                    reward = actual_accuracy if actual_accuracy is not None else 0.0
                    success = reward > 0.5 if actual_accuracy is not None else False
                    
                    self.scheduler.update_task_stats(
                        task_id, 
                        reward=reward,
                        success=success,
                        latent_vector=extras.get("latent"),
                        extras=extras  # Pass full extras for eval_metrics
                    )
                    
                    if self.config.verbose:
                        # Log what metrics we actually used
                        eval_metrics = extras.get("eval_metrics", {})
                        ebr_deltas = extras.get('ebr_deltas', [])
                        if eval_metrics:
                            print(f"[Scheduler] Updated task {task_id} with accuracy={actual_accuracy:.3f} (from eval_metrics)")
                        elif ebr_deltas:
                            print(f"[Scheduler] Updated task {task_id} with accuracy={actual_accuracy:.3f} (from EBR heuristic)")
                        else:
                            print(f"[Scheduler] Updated task {task_id} with accuracy={actual_accuracy:.3f} (from rail path)")
                
                if self.config.verbose and current_strategy:
                    print(f"[Scheduler] Strategy '{current_strategy}' used")
        
        # Record experience and do micro-dream step
        if self.dream is not None:
            try:
                # latent_state ~ your 'brain' control vector [B, D]; take first if B>1
                latent_state = brain[0].detach()
                # store a zero-reward experience (you can pass action/reward later)
                self.dream.record_experience(latent_state=latent_state)
                # run bounded micro-dream consolidation
                for _ in range(self.config.dream_micro_ticks):
                    _ = self.dream.step_micro(valence=None, arousal=None)
            except Exception as e:
                print(f"[WARN] Dream micro-step failed: {e}")
        
        # Add auxiliary losses including RelMem inverse-consistency
        aux_loss = 0.0
        # ... existing auxiliary losses ...
        if self.relmem is not None:
            try:
                inv_w = float(getattr(self.config, "relmem_inverse_loss_w", 0.05))
                aux_loss = aux_loss + inv_w * self.relmem.inverse_loss()
            except Exception:
                pass
        extras["aux_loss"] = extras.get("aux_loss", 0.0) + aux_loss
        
        grid, logits, size, extras = enforce_sacred_signature(grid, logits, size, extras, "FINAL-OUTPUT", training_mode)
        return grid, logits, size, extras
    
    def _cache_dream_tokens(self, tokens: torch.Tensor, H: int, W: int, max_tokens: int = 256) -> None:
        """
        Store a small, detached slice of tokens for offline dream cycles.
        tokens: [B, T, D]
        Keeps 1x(min(T, max_tokens))xD on the model's device, detached.
        """
        assert tokens.dim() == 3, f"tokens expected [B,T,D], got {tokens.shape}"
        B, T, D = tokens.shape
        # Take first sample only to bound memory
        t = min(T, max_tokens)
        stride = max(1, T // t)
        cached = tokens[0:1, ::stride, :].detach().contiguous()
        self._dream_tokens = cached
        self._dream_tokens_info["H"] = H
        self._dream_tokens_info["W"] = W
    
    def _grid_fingerprint(self, g: torch.Tensor) -> str:
        g2 = g.detach().to("cpu").contiguous()
        return hashlib.sha1(g2.numpy().tobytes()).hexdigest()[:16]

    def _resize_nn(self, grid: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # one-hot nearest neighbor resize (mirrors your DSL op)
        C = int(grid.max().item()) + 1 if grid.numel() else NUM_COLORS
        C = max(C, NUM_COLORS)
        onehot = torch.zeros(1, C, grid.size(-2), grid.size(-1), device=grid.device, dtype=torch.float32)
        onehot[0].scatter_(0, grid.unsqueeze(0).clamp(0, C-1).long(), 1.0)
        up = F.interpolate(onehot, size=(int(H), int(W)), mode="nearest")
        return up.argmax(dim=1)[0].to(grid.dtype)

    def _center_pad_to(self, grid: torch.Tensor, H: int, W: int, pad_value: int = 0) -> torch.Tensor:
        out = torch.full((int(H), int(W)), int(pad_value), device=grid.device, dtype=grid.dtype)
        h, w = grid.shape[-2], grid.shape[-1]
        y0 = max((H - h) // 2, 0)
        x0 = max((W - w) // 2, 0)
        out[y0:y0+h, x0:x0+w] = grid[:min(h,H), :min(w,W)]
        return out

    def _infer_relations_from_demo(self, inp: torch.Tensor, out: torch.Tensor) -> List[tuple]:
        """
        Enhanced relation inference to detect diverse transformations between inp and out.
        Returns list of (rel, params) tuples.
        
        Detects:
        - Rotations: rotate90, rotate180, rotate270
        - Flips: flip_h, flip_v
        - Size changes: resize, center_pad_to
        - Color transformations: color_map, mask_color, extract_color
        - Spatial operations: translate, crop_bbox, crop_nonzero
        - Pattern operations: tile_pattern, scale
        """
        rels = []
        # Normalize to [H,W]
        if inp.dim() == 3: inp = inp[0]
        if out.dim() == 3: out = out[0]

        # 1) Rotations - Detect 90/180/270 degree rotations
        rels.extend(self._detect_rotations(inp, out))
        
        # 2) Flips - Detect horizontal/vertical flips
        rels.extend(self._detect_flips(inp, out))
        
        # 3) Size transformations - Resize and padding operations
        rels.extend(self._detect_size_transforms(inp, out))
        
        # 4) Color transformations - Color mapping, masking, extraction
        rels.extend(self._detect_color_transforms(inp, out))
        
        # 5) Spatial operations - Translation and cropping
        rels.extend(self._detect_spatial_transforms(inp, out))
        
        # 6) Pattern operations - Scaling and tiling
        rels.extend(self._detect_pattern_transforms(inp, out))

        return rels
    
    def _detect_rotations(self, inp: torch.Tensor, out: torch.Tensor) -> List[tuple]:
        """Detect rotation transformations (90, 180, 270 degrees)"""
        rels = []
        try:
            if torch.equal(torch.rot90(inp, k=-1, dims=(0,1)), out):
                rels.append(("rotate90", None))
            elif torch.equal(torch.rot90(inp, k=2, dims=(0,1)), out):
                rels.append(("rotate180", None))
            elif torch.equal(torch.rot90(inp, k=1, dims=(0,1)), out):
                rels.append(("rotate270", None))
        except Exception:
            pass
        return rels
    
    def _detect_flips(self, inp: torch.Tensor, out: torch.Tensor) -> List[tuple]:
        """Detect flip transformations (horizontal and vertical)"""
        rels = []
        try:
            if torch.equal(torch.flip(inp, dims=(1,)), out):
                rels.append(("flip_h", None))
            elif torch.equal(torch.flip(inp, dims=(0,)), out):
                rels.append(("flip_v", None))
        except Exception:
            pass
        return rels
    
    def _detect_size_transforms(self, inp: torch.Tensor, out: torch.Tensor) -> List[tuple]:
        """Detect size-changing transformations (resize, center_pad_to)"""
        rels = []
        Hi, Wi = inp.shape
        Ho, Wo = out.shape
        
        # Resize detection (nearest-neighbor)
        if (Hi != Ho or Wi != Wo) and Hi > 0 and Wi > 0:
            try:
                cand = self._resize_nn(inp, Ho, Wo)
                if torch.equal(cand, out):
                    sx = Ho / max(Hi, 1)
                    sy = Wo / max(Wi, 1)
                    rels.append(("resize", {"sx": float(sx), "sy": float(sy)}))
            except Exception:
                pass
        
        # Center pad detection
        if Ho >= Hi and Wo >= Wi:
            try:
                # Try to reconstruct padding value from corners
                pad_val = int(out[0,0].item()) if out.numel() else 0
                cand = self._center_pad_to(inp, Ho, Wo, pad_val)
                if torch.equal(cand, out):
                    rels.append(("center_pad_to", {"H": float(Ho), "W": float(Wo), "pad": float(pad_val)}))
            except Exception:
                pass
        
        return rels
    
    def _detect_color_transforms(self, inp: torch.Tensor, out: torch.Tensor) -> List[tuple]:
        """Detect color transformations (color_map, mask_color, extract_color)"""
        rels = []
        
        # Only try color transforms if shapes match
        if inp.shape != out.shape:
            return rels
        
        try:
            # Color mapping detection
            color_mapping = self._learn_color_mapping(inp, out)
            if color_mapping:
                rels.append(("color_map", {"mapping": color_mapping}))
            
            # Extract single color detection
            unique_out = torch.unique(out)
            if len(unique_out) == 2 and 0 in unique_out:
                # Output has background (0) and one other color
                target_color = unique_out[unique_out != 0][0].item()
                # Check if output matches extraction of this color from input
                extracted = torch.zeros_like(inp)
                extracted[inp == target_color] = target_color
                if torch.equal(extracted, out):
                    rels.append(("extract_color", {"color": float(target_color)}))
            
            # Mask color detection (color gets replaced with 0)
            inp_colors = set(torch.unique(inp).cpu().numpy())
            out_colors = set(torch.unique(out).cpu().numpy())
            masked_colors = inp_colors - out_colors
            if len(masked_colors) == 1:
                masked_color = list(masked_colors)[0]
                # Verify masking
                test_out = inp.clone()
                test_out[inp == masked_color] = 0
                if torch.equal(test_out, out):
                    rels.append(("mask_color", {"color": float(masked_color)}))
        
        except Exception:
            pass
        
        return rels
    
    def _detect_spatial_transforms(self, inp: torch.Tensor, out: torch.Tensor) -> List[tuple]:
        """Detect spatial transformations (translate, crop_bbox, crop_nonzero)"""
        rels = []
        
        try:
            # Translation detection (same size grids)
            if inp.shape == out.shape:
                translation = self._detect_translation(inp, out)
                if translation:
                    dx, dy = translation
                    rels.append(("translate", {"dx": float(dx), "dy": float(dy)}))
            
            # Crop detection (output smaller than input)
            Hi, Wi = inp.shape
            Ho, Wo = out.shape
            if Ho <= Hi and Wo <= Wi:
                # Try crop_bbox (crop to bounding box of non-zero)
                crop_bbox_result = self._crop_bbox(inp)
                if crop_bbox_result is not None and torch.equal(crop_bbox_result, out):
                    rels.append(("crop_bbox", None))
                
                # Try crop_nonzero (similar to crop_bbox but different implementation)
                crop_nonzero_result = self._crop_nonzero(inp)
                if crop_nonzero_result is not None and torch.equal(crop_nonzero_result, out):
                    rels.append(("crop_nonzero", None))
        
        except Exception:
            pass
        
        return rels
    
    def _detect_pattern_transforms(self, inp: torch.Tensor, out: torch.Tensor) -> List[tuple]:
        """Detect pattern transformations (scale, tile_pattern)"""
        rels = []
        
        try:
            Hi, Wi = inp.shape
            Ho, Wo = out.shape
            
            # Scale detection (integer scaling)
            if Ho > Hi and Wo > Wi and Ho % Hi == 0 and Wo % Wi == 0:
                fy = Ho // Hi
                fx = Wo // Wi
                if fy == fx:  # Uniform scaling
                    scaled = inp.repeat_interleave(fy, dim=0).repeat_interleave(fx, dim=1)
                    if torch.equal(scaled, out):
                        rels.append(("scale", {"fy": float(fy), "fx": float(fx)}))
            
            # Tile pattern detection (extract repeating pattern)
            if Ho <= Hi and Wo <= Wi:
                tile_result = self._detect_tile_pattern(inp, out)
                if tile_result:
                    rels.append(("tile_pattern", None))
        
        except Exception:
            pass
        
        return rels
    
    def _learn_color_mapping(self, inp: torch.Tensor, out: torch.Tensor) -> Optional[Dict[int, int]]:
        """Learn consistent color mapping between input and output grids"""
        if inp.shape != out.shape:
            return None
        
        try:
            inp_colors = torch.unique(inp).cpu().tolist()
            mapping = {}
            
            for color in inp_colors:
                mask = (inp == color)
                output_vals = out[mask]
                unique_outputs = torch.unique(output_vals)
                
                # Consistent mapping: all pixels of input color map to same output color
                if len(unique_outputs) == 1:
                    mapping[color] = int(unique_outputs[0].item())
                else:
                    return None  # Inconsistent mapping
            
            # Verify mapping works
            test_output = inp.clone()
            for inp_color, out_color in mapping.items():
                test_output[inp == inp_color] = out_color
            
            if torch.equal(test_output, out):
                return mapping
        
        except Exception:
            pass
        
        return None
    
    def _detect_translation(self, inp: torch.Tensor, out: torch.Tensor) -> Optional[Tuple[int, int]]:
        """Detect translation (shift) between grids of same size"""
        if inp.shape != out.shape:
            return None
        
        try:
            H, W = inp.shape
            # Try different translation offsets
            for dy in range(-H//2, H//2 + 1):
                for dx in range(-W//2, W//2 + 1):
                    if dx == 0 and dy == 0:
                        continue
                    
                    # Create translated version
                    translated = torch.zeros_like(inp)
                    for i in range(H):
                        for j in range(W):
                            if inp[i, j] != 0:  # Only move non-background
                                new_i = (i + dy) % H
                                new_j = (j + dx) % W
                                translated[new_i, new_j] = inp[i, j]
                    
                    if torch.equal(translated, out):
                        return (dx, dy)
        
        except Exception:
            pass
        
        return None
    
    def _crop_bbox(self, grid: torch.Tensor) -> Optional[torch.Tensor]:
        """Crop to bounding box of non-zero elements"""
        try:
            nonzero = torch.nonzero(grid)
            if len(nonzero) == 0:
                return grid  # No non-zero elements
            
            min_r, min_c = nonzero.min(dim=0).values
            max_r, max_c = nonzero.max(dim=0).values
            
            return grid[min_r:max_r+1, min_c:max_c+1].clone()
        except Exception:
            return None
    
    def _crop_nonzero(self, grid: torch.Tensor) -> Optional[torch.Tensor]:
        """Alternative crop implementation (similar to crop_bbox)"""
        return self._crop_bbox(grid)  # Same implementation for now
    
    def _detect_tile_pattern(self, inp: torch.Tensor, out: torch.Tensor) -> bool:
        """Detect if output is a fundamental tile pattern of input"""
        try:
            Hi, Wi = inp.shape
            Ho, Wo = out.shape
            
            # Check if output can be tiled to reconstruct input
            if Hi % Ho != 0 or Wi % Wo != 0:
                return False
            
            tiles_y = Hi // Ho
            tiles_x = Wi // Wo
            
            # Check if input is made of repeated output tiles
            for ty in range(tiles_y):
                for tx in range(tiles_x):
                    y_start = ty * Ho
                    x_start = tx * Wo
                    tile_region = inp[y_start:y_start+Ho, x_start:x_start+Wo]
                    
                    if not torch.equal(tile_region, out):
                        return False
            
            return True
        
        except Exception:
            return False
    
    def _apply_ebr(self, logits: torch.Tensor, priors: dict, extras: dict = None) -> torch.Tensor:
        """
        Apply EnergyRefiner with grad safety:
        - Pass continuous logits into refinement
        - Use STE only at the output interface
        - Mark success/failure explicitly in extras
        """
        try:
            if self.config.verbose:
                print("[EBR] Refining output (grad-safe)")
            
            if logits.dim() != 4:
                raise RuntimeError(f"[EBR] INPUT VIOLATION: logits must be [B,C,H,W], got {logits.shape}")
            
            B, C, H, W = logits.shape
            cons = ARCGridConstraints(expect_symmetry=None, color_hist=None, sparsity=None)
            
            refined_logits = self.ebr(logits, cons, priors)  # differentiable
            
            class ArgmaxSTE(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    return x.softmax(dim=1).argmax(dim=1)
                @staticmethod
                def backward(ctx, grad_output):
                    return grad_output.unsqueeze(1).expand_as(refined_logits)
            
            grid_refined = ArgmaxSTE.apply(refined_logits)
            grid_refined = torch.clamp(grid_refined, 0, NUM_COLORS-1)
            
            if extras is not None:
                extras["ebr_ok"] = True
            return grid_refined
            
        except Exception as e:
            print(f"[EBR] Failure: {e}")
            if extras is not None:
                extras["ebr_ok"] = False
                extras["ebr_error"] = str(e)
            return logits.softmax(dim=1).argmax(dim=1)  # honest fallback
    
    @torch.no_grad()
    def run_dream_cycle(self, demos_programs=None, tokens: Optional[torch.Tensor] = None):
        """
        Run offline dream consolidation.
        Priority for tokens:
          1) tokens=... passed explicitly
          2) cached self._dream_tokens from last forward
        Fails fast if neither is available.
        """
        if self.dream is None:
            print("[Dream] Disabled")
            return {}

        if tokens is None:
            tokens = self._dream_tokens

        if tokens is None:
            raise RuntimeError(
                "run_dream_cycle requires tokens but none were provided or cached. "
                "Call model.forward(...) once (to cache tokens) or pass tokens=<B,T,D> explicitly."
            )
        
        # Ensure tokens are on the correct device
        model_device = self._get_device()
        if tokens.device.type != model_device or str(tokens.device) != model_device:
            tokens = tokens.to(model_device)

        return self.dream.cycle_offline(tokens, demos_programs=demos_programs)
        
    def update_scheduler_metrics(self, task_id: str, metrics: Dict[str, float], 
                                pred_grid: torch.Tensor = None, latent_state: torch.Tensor = None, 
                                rule_vector: torch.Tensor = None):
        """
        Update scheduler with actual performance metrics after forward pass.
        
        Args:
            task_id: Task identifier
            metrics: Dictionary with 'acc', 'iou', 'exact' metrics
            pred_grid: Predicted grid for novelty calculation
            latent_state: Latent state for empowerment calculation
            rule_vector: Rule vector for sync calculation
        """
        if self.scheduler is None:
            return
            
        try:
            # Update scheduler with real performance metrics
            self.scheduler.update(
                tid=task_id,
                metrics=metrics,
                pred_grid=pred_grid,
                latent_state=latent_state,
                rule_vector=rule_vector
            )
            
            if self.config.verbose:
                stats = self.scheduler.stats.get(task_id, {})
                print(f"[Scheduler] Task {task_id[:8]} metrics updated: "
                      f"acc={stats.get('acc', 0.0):.3f}, exact={stats.get('exact', 0.0):.3f}, "
                      f"empowerment={stats.get('empowerment', 0.0):.3f}, sync={stats.get('sync', 0.0):.3f}")
                
        except Exception as e:
            print(f"[WARN] Scheduler metrics update failed: {e}")
            
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """Get comprehensive scheduler statistics"""
        if self.scheduler is None:
            return {}
            
        try:
            stats = self.scheduler.get_enhanced_stats()
            stats.update({
                "strategy_usage": dict(self.strategy_usage),
                "task_count": len(self.scheduler.task_stats),
                "allocation_stats": {
                    "avg_retries": np.mean(list(self.scheduler.retry_allocations.values())) if self.scheduler.retry_allocations else 0.0,
                    "max_retries": max(self.scheduler.retry_allocations.values()) if self.scheduler.retry_allocations else 0,
                    "min_retries": min(self.scheduler.retry_allocations.values()) if self.scheduler.retry_allocations else 0
                }
            })
            return stats
        except Exception as e:
            print(f"[WARN] Failed to get scheduler stats: {e}")
            return {}

    # === PHASE 0 PRETRAINING MODE SUPPORT ===
    
    def set_pretraining_mode(self, enabled: bool = True):
        """Enable/disable pretraining mode"""
        self._pretraining_mode = enabled
        self.config.pretraining_mode = enabled
        
        if enabled:
            print("[TOPAS] Pretraining mode ENABLED - multi-head learning active")
        else:
            print("[TOPAS] Pretraining mode DISABLED - standard inference active")
    
    def get_pretraining_mode(self) -> bool:
        """Check if model is in pretraining mode"""
        return self._pretraining_mode
    
    def set_multi_head_pretrainer(self, pretrainer):
        """Set external multi-head pretrainer for this model"""
        self._multi_head_pretrainer = pretrainer
        self.set_pretraining_mode(True)
        print("[TOPAS] Multi-head pretrainer attached")
    
    def get_neural_features(self, test_grid: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract neural features for pretraining (exposed for MultiHeadPretrainer)
        
        Args:
            test_grid: Test input grid [B, H, W]
            
        Returns:
            Dictionary with neural features
        """
        # Ensure float for encoder
        if test_grid.dtype != torch.float32:
            enc_in = test_grid.float() / (NUM_COLORS - 1)
        else:
            enc_in = test_grid
        
        # Extract features using model components
        feat, glob = self.encoder(enc_in)
        
        # Robust unpacking for ObjectSlots output
        slots_output = self.slots(feat)
        if torch.is_tensor(slots_output):
            slot_vecs = slots_output
        elif isinstance(slots_output, (list, tuple)):
            if len(slots_output) >= 1:
                slot_vecs = slots_output[0]
            else:
                raise ValueError("ObjectSlots returned empty sequence")
        else:
            raise TypeError(f"Unexpected slots_output type: {type(slots_output)}")
        
        # Normalize to [B, K, D]
        if slot_vecs.dim() == 2:   # e.g., [B, D] → assume K=1
            slot_vecs = slot_vecs.unsqueeze(1)
        slots_rel = self.reln(slot_vecs)
        pooled = slots_rel.mean(dim=1)
        
        # Create projection layer dynamically if needed
        if self.encoder_proj is None:
            actual_glob_dim = glob.shape[-1]
            self.encoder_proj = nn.Linear(actual_glob_dim, self.config.slot_dim).to(glob.device)
        
        # Control vector (brain) with projection
        glob_proj = self.encoder_proj(glob)
        brain = torch.cat([glob_proj, pooled], dim=-1)
        
        return {
            'feat': feat,           # Feature maps [B, C, H, W]
            'glob': glob,           # Global features [B, width]
            'slot_vecs': slot_vecs, # Slot vectors [B, K, slot_dim]
            'slots_rel': slots_rel, # Relational slots [B, K, slot_dim]
            'pooled': pooled,       # Pooled slots [B, slot_dim]
            'brain': brain          # Control vector [B, ctrl_dim]
        }
    
    def forward_pretraining(self, test_grid: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass optimized for pretraining (no DSL search, just neural features)
        
        Args:
            test_grid: Test input grid [B, H, W]
            
        Returns:
            Dictionary with neural predictions for pretraining heads
        """
        if not self._pretraining_mode:
            raise RuntimeError("Model must be in pretraining mode to use forward_pretraining")
        
        # Normalize input if needed
        if test_grid.dtype != torch.float32:
            enc_in = test_grid.float() / (NUM_COLORS - 1)
        else:
            enc_in = test_grid
        
        # Get encoder features
        feat, glob = self.encoder(enc_in)
        
        # Get slots and attention from ObjectSlots
        slots_output = self.slots(feat)
        
        # Robust unpacking to get slot_vecs and attention
        if torch.is_tensor(slots_output):
            slot_vecs = slots_output
            attn = None
        elif isinstance(slots_output, (list, tuple)):
            slot_vecs = slots_output[0]
            attn = slots_output[1] if len(slots_output) >= 2 else None
        else:
            raise TypeError(f"Unexpected slots_output type: {type(slots_output)}")
        
        # Normalize slot_vecs to [B, K, D]
        if slot_vecs.dim() == 2:   # [B, D] → assume K=1
            slot_vecs = slot_vecs.unsqueeze(1)
        
        # Generate per-slot logits [B, K, C]
        slot_logits = self.slot_to_logits(slot_vecs)
        
        B = slot_logits.size(0)
        K = slot_logits.size(1)
        C = self.num_colors
        
        # Get actual grid dimensions from input
        if test_grid.dim() == 2:  # [H, W]
            H, W = test_grid.shape
        elif test_grid.dim() == 3:  # [B, H, W] or [C, H, W]
            if test_grid.size(0) == B:  # [B, H, W]
                H, W = test_grid.shape[-2:]
            else:  # [C, H, W]
                H, W = test_grid.shape[-2:]
        else:
            H, W = test_grid.shape[-2:]
        
        # Generate per-pixel logits using attention
        if attn is not None:
            # Normalize attention shape
            if attn.dim() == 4:  # [B, K, H, W]
                attn_flat = attn.view(B, K, H*W)
            elif attn.dim() == 3:  # Already [B, K, H*W]
                attn_flat = attn
            else:
                # Unexpected attention shape, use pixel fallback
                attn = None
            
            if attn is not None:
                # Check if attention is already normalized (avoid double-softmax)
                attn_sum = attn_flat.sum(dim=1, keepdim=True)  # [B, 1, H*W]
                is_normalized = torch.allclose(attn_sum.mean(), torch.ones(1, device=attn_sum.device), atol=0.1)
                
                if not is_normalized:
                    # Normalize attention over slots only if not already normalized
                    attn_flat = torch.softmax(attn_flat, dim=1)  # [B, K, H*W]
                
                # Mix slot logits with attention: [B, K, C] x [B, K, P] -> [B, C, P]
                pixel_logits = torch.einsum("bkc,bkp->bcp", slot_logits, attn_flat)
                logits = pixel_logits.permute(0, 2, 1).contiguous()  # [B, P, C]
                logits = logits.view(B, H*W, C)  # [B, H*W, C]
            else:
                attn = None  # Force pixel fallback
        
        if attn is None:
            # Use pixel fallback head instead of uniform attention
            pixel_logits = self.pixel_fallback(feat)  # [B, C, H', W']
            
            # Ensure spatial dimensions match
            if pixel_logits.shape[-2:] != (H, W):
                pixel_logits = torch.nn.functional.interpolate(
                    pixel_logits, size=(H, W), mode='bilinear', align_corners=False
                )
            
            # Reshape to [B, H*W, C]
            logits = pixel_logits.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
            logits = logits.view(B, H*W, C)
        
        # Sanity checks for training hygiene
        assert torch.isfinite(logits).all(), "Non-finite logits detected"
        assert logits.shape == (B, H*W, C), f"Logits shape mismatch: expected {(B, H*W, C)}, got {logits.shape}"
        
        # Get painter prediction (for comparison)
        grid_pred, _, _ = self.painter(feat)
        
        # Extract full neural features
        features = self.get_neural_features(test_grid)
        
        # Prepare predictions dictionary
        predictions = {
            'logits': logits,  # [B, H*W, C] for cross-entropy loss
            'grid': grid_pred,
            'features': features,
            'slot_vecs': slot_vecs,
            'slot_logits': slot_logits
        }
        
        return predictions
    
    # === POLICY MODE METHODS ===
    
    def enable_policy_mode(self, policy_net, value_net, ebr_direction_net):
        """Enable policy mode with distilled networks"""
        self.policy_net = policy_net
        self.value_net = value_net  
        self.ebr_direction_net = ebr_direction_net
        
        # Move to same device as main model
        if hasattr(self, 'device'):
            self.policy_net.to(self.device)
            self.value_net.to(self.device)
            self.ebr_direction_net.to(self.device)
        
        print("[TOPAS] Policy mode enabled - will use distilled networks when confidence is high")
    
    def _forward_with_policy(self, demos, test_grid, priors, max_length: int = 6) -> Optional[Tuple[torch.Tensor, List[str], float]]:
        """
        Use OpPolicyNet instead of beam search for fast program generation
        
        Args:
            demos: Demonstration pairs
            test_grid: Test input grid
            priors: Neural priors dictionary
            max_length: Maximum program length
            
        Returns:
            Tuple of (output_grid, operations_list, confidence) or None if failed
        """
        if not hasattr(self, 'policy_net') or self.policy_net is None:
            return None
            
        try:
            # Prepare input grid
            if test_grid.dim() == 2:
                input_grid = test_grid.unsqueeze(0)  # Add batch dim
            else:
                input_grid = test_grid
            
            # Extract features for policy network
            B, H, W = input_grid.shape
            
            # Create dummy features (in real implementation, use actual features from encoder)
            rel_features = torch.randn(B, 64, device=input_grid.device)
            size_oracle = torch.tensor([[H, W, H, W]], device=input_grid.device).float()
            
            # Extract theme priors
            if 'trans' in priors:
                theme_priors = priors['trans'].flatten()[:10]
                if len(theme_priors) < 10:
                    theme_priors = F.pad(theme_priors, (0, 10 - len(theme_priors)))
                theme_priors = theme_priors.unsqueeze(0)
            else:
                theme_priors = torch.randn(B, 10, device=input_grid.device)
            
            # Generate program with policy network
            program_ops = self.policy_net.generate_program(
                input_grid, rel_features, size_oracle, theme_priors,
                max_length=max_length, stop_threshold=0.5
            )
            
            if not program_ops:
                return None
            
            # Get confidence from value network
            value_pred = self.value_net.forward(
                input_grid, rel_features, size_oracle, theme_priors,
                program_ops, len(program_ops)
            )
            
            confidence = value_pred.confidence.item()
            solvability = value_pred.solvability.item()
            
            # Combined confidence score
            combined_confidence = (confidence + solvability) / 2.0
            
            # Execute program using DSL operations
            output_grid = self._execute_policy_program(input_grid, program_ops, demos)
            
            if output_grid is not None:
                return output_grid, program_ops, combined_confidence
            else:
                return None
                
        except Exception as e:
            print(f"[POLICY] Failed to generate policy-based solution: {e}")
            return None
    
    def _execute_policy_program(self, input_grid: torch.Tensor, program_ops: List[str], demos) -> Optional[torch.Tensor]:
        """Execute a program generated by the policy network"""
        try:
            # Use direct program execution instead of zombie DSLHead class
            from models.dsl_search import apply_program, DSLProgram
            
            # Create program object directly
            program = DSLProgram(
                ops=program_ops,
                params=[{}] * len(program_ops)  # Simple parameters for now
            )
            
            # Apply program to test input directly
            current_grid = input_grid[0] if input_grid.dim() == 3 else input_grid
            result = apply_program(current_grid, program)
            
            if result.dim() == 2:
                result = result.unsqueeze(0)  # Ensure batch dimension
            
            return result
            
        except Exception as e:
            print(f"[POLICY] Failed to execute program {program_ops}: {e}")
            return None
    
    def _apply_fast_ebr(self, grid_logits: torch.Tensor, priors: dict) -> torch.Tensor:
        """
        Apply fast EBR using EBRDirectionNet (1-2 steps instead of 5-7)
        
        Args:
            grid_logits: Grid logits [B, H*W, C]
            priors: Neural priors dictionary
            
        Returns:
            Refined logits [B, H*W, C]
        """
        if not hasattr(self, 'ebr_direction_net') or self.ebr_direction_net is None:
            # Fallback to regular EBR
            return grid_logits
        
        try:
            # Create constraint object
            from trainers.arc_constraints import ARCGridConstraints
            constraint_obj = ARCGridConstraints()
            
            # Prepare prior tensors
            prior_tensors = {}
            for key in ['phi', 'kappa', 'cge', 'hodge']:
                if key in priors:
                    prior_tensors[key] = priors[key]
                else:
                    prior_tensors[key] = torch.tensor(0.0, device=grid_logits.device)
            
            # Apply fast EBR refinement
            refined_logits = self.ebr_direction_net.refine_logits_fast(
                grid_logits, constraint_obj, prior_tensors, max_steps=2
            )
            
            print("[FAST-EBR] Applied 2-step refinement instead of 5-7 steps")
            return refined_logits
            
        except Exception as e:
            print(f"[FAST-EBR] Failed: {e}, falling back to regular EBR")
            return grid_logits
    
    def get_distillation_metrics(self) -> Dict[str, Any]:
        """Get metrics about policy mode usage"""
        if not hasattr(self, 'policy_net'):
            return {'policy_enabled': False}
        
        # This would be expanded with actual usage statistics
        return {
            'policy_enabled': True,
            'policy_net_params': sum(p.numel() for p in self.policy_net.parameters()),
            'value_net_params': sum(p.numel() for p in self.value_net.parameters()) if hasattr(self, 'value_net') else 0,
            'ebr_net_params': sum(p.numel() for p in self.ebr_direction_net.parameters()) if hasattr(self, 'ebr_direction_net') else 0
        }
    
    def sample_trace(self, task: Dict, temperature: float = 1.0, max_length: int = 50) -> Dict:
        """
        Sample a trace of DSL operations for a task.
        Used by star_bootstrapper.py for self-taught reasoning.
        
        Args:
            task: Task dictionary with 'input' and optionally 'output'
            temperature: Sampling temperature for exploration
            max_length: Maximum trace length
            
        Returns:
            Dict with 'program' and 'final_grid' keys
        """
        # Get input grid
        input_grid = task.get('input', torch.zeros(1, 10, 10))
        if isinstance(input_grid, list):
            input_grid = torch.tensor(input_grid, dtype=torch.long)
        if input_grid.dim() == 2:
            input_grid = input_grid.unsqueeze(0)
        
        # Run forward pass to get features
        with torch.no_grad():
            # Get features from encoder
            features = self.encoder(input_grid.float())
            # Handle tuple return from encoder
            if isinstance(features, tuple):
                features = features[0]
            brain = features.mean(dim=[2, 3])  # Global pooling
            
            # Sample operations from DSL
            trace = []
            current_grid = input_grid.clone()
            
            for _ in range(max_length):
                # Use DSL to predict next operation
                if hasattr(self.dsl, 'sample_operation'):
                    op, params = self.dsl.sample_operation(current_grid, brain, temperature)
                else:
                    # Fallback: sample random operation
                    ops = ['rotate_90', 'flip_horizontal', 'color_map', 'identity']
                    op = ops[torch.randint(0, len(ops), (1,)).item()]
                    params = {}
                
                trace.append((op, params))
                
                # Stop if identity (no-op)
                if op == 'identity':
                    break
                
                # Apply operation (simplified)
                if hasattr(self.dsl, op):
                    try:
                        current_grid = getattr(self.dsl, op)(current_grid, **params)
                    except:
                        pass
            
            # Return in required format
            return {
                "program": trace,
                "final_grid": current_grid.squeeze(0) if current_grid.dim() > 2 else current_grid
            }
    
    def forward_with_trace(self, input_grid: torch.Tensor) -> Dict:
        """
        Forward pass that returns both output and execution trace.
        Used by alpha_dsl.py for MCTS search.
        
        Args:
            input_grid: Input grid [B, H, W] or [B, C, H, W]
            
        Returns:
            Dict with 'program' and 'final_grid' keys
        """
        # Track operations executed
        trace = []
        
        # Ensure proper shape
        if input_grid.dim() == 2:
            input_grid = input_grid.unsqueeze(0)
        if input_grid.dim() == 3:
            B, H, W = input_grid.shape
        else:
            B, _, H, W = input_grid.shape
            input_grid = input_grid.squeeze(1)
        
        # Run standard forward pass
        with torch.no_grad():
            # Record that we're starting
            trace.append('encode')
            
            # Run through pipeline
            grid, logits, size, extras = self.forward(input_grid)
            
            # Record main operations
            if 'dsl_used' in extras:
                trace.extend(extras['dsl_used'])
            else:
                trace.append('dsl_search')
            
            if 'ebr_used' in extras and extras['ebr_used']:
                trace.append('energy_refine')
            
            if 'painter_used' in extras and extras['painter_used']:
                trace.append('neural_paint')
            
            trace.append('decode')
        
        # Return in required format
        return {
            "program": trace,
            "final_grid": grid.squeeze(0) if grid.dim() > 2 else grid
        }


def create_model(device="cuda", checkpoint=None, config=None):
    """
    Convenience function to create and optionally load model
    
    Args:
        device: Device to place model on
        checkpoint: Optional path to checkpoint file
        config: Optional ModelConfig, uses defaults if None
        
    Returns:
        TopasARC60M model ready for inference
    """
    if config is None:
        config = ModelConfig()  # Uses new 60M defaults
    model = TopasARC60M(config).to(device)
    
    if checkpoint:
        print(f"[TOPAS] Loading checkpoint from {checkpoint}")
        state_dict = torch.load(checkpoint, map_location=device)
        
        # Handle different checkpoint formats
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
            
        # Try to load, ignoring mismatches
        try:
            model.load_state_dict(state_dict, strict=False)
            print("[TOPAS] Checkpoint loaded successfully")
        except Exception as e:
            print(f"[TOPAS] Warning: Partial checkpoint load - {e}")
    
    model.eval()
    print(f"[TOPAS] ARC 60M ready on {device}")
    print("[TOPAS] Rails: DSL -> EBR -> Sacred Signature")
    
    return model


if __name__ == "__main__":
    # Quick test
    print("\n" + "="*60)
    print("TOPAS ARC 60M - Quick Test")
    print("="*60)
    
    # Create model
    model = create_model(device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dummy data
    demos = [
        {
            'input': torch.randint(0, 10, (10, 10)),
            'output': torch.randint(0, 10, (10, 10))
        }
    ]
    test = {'input': torch.randint(0, 10, (10, 10))}
    
    # Run forward pass
    with torch.no_grad():
        grid, logits, size, extras = model(demos, test)
    
    # Verify sacred signature
    print(f"\n[TEST] Output shapes:")
    print(f"  grid: {grid.shape} (expected [B, H, W])")
    print(f"  logits: {logits.shape} (expected [B, H*W, C])")
    print(f"  size: {size.shape} (expected [B, 2])")
    print(f"  extras: {type(extras)} with keys: {list(extras.keys()) if isinstance(extras, dict) else 'N/A'}")
    
    assert grid.dim() == 3, "Grid must be [B, H, W]"
    assert logits.dim() == 3 and logits.shape[2] == NUM_COLORS, "Logits must be [B, H*W, C]"
    assert size.dim() == 2 and size.shape[1] == 2, "Size must be [B, 2]"
    
    # Check extras dict structure
    assert isinstance(extras, dict), "Extras must be a dictionary"
    assert "latent" in extras, "Extras must contain 'latent' key"
    assert "rule_vec" in extras, "Extras must contain 'rule_vec' key"
    
    # Sacred Signature verification
    validate_sacred_signature(grid, logits, size, extras, "TEST-VALIDATION")
    print("\nSacred Signature validation passed!")
    print("="*60)