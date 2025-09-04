
import numpy as np
import torch
import json
import os
import time
import hashlib
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict
import logging

@dataclass
class Template:
    """Represents a reusable program template with MDL-style compression"""
    ops: List[Tuple[str, dict]]  # List of (operation_name, parameters)
    support: int  # Number of programs that use this template
    score: float  # MDL compression gain score
    signature: str  # Unique identifier for deduplication
    created_at: float = 0.0  # Timestamp for TTL/decay
    usage_count: int = 0  # Track how often template is used
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
    
    def to_dict(self) -> dict:
        return {
            'ops': self.ops,
            'support': self.support,
            'score': self.score,
            'signature': self.signature,
            'created_at': self.created_at,
            'usage_count': self.usage_count
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Template':
        # Ensure ops are tuples, not lists (JSON serialization converts tuples to lists)
        ops = data.get('ops', [])
        if ops and isinstance(ops[0], list):
            ops = [tuple(op) if isinstance(op, list) else op for op in ops]
        return cls(
            ops=ops,
            support=data.get('support', 0),
            score=data.get('score', 0.0),
            signature=data.get('signature', ''),
            created_at=data.get('created_at', 0.0),
            usage_count=data.get('usage_count', 0)
        )
    
    def apply_to_grid(self, grid: torch.Tensor, dsl_ops: dict) -> torch.Tensor:
        """Apply template operations to a grid"""
        result = grid.clone()
        for op_name, params in self.ops:
            if op_name in dsl_ops:
                result = dsl_ops[op_name](result, **params)
        return result
    
    def get_primitive_ops(self) -> List[str]:
        """Get list of primitive operation names"""
        return [op_name for op_name, _ in self.ops]

class TemplateLibrary:
    """Manages templates with persistence and TTL/decay"""
    
    def __init__(self, storage_path: str = "templates.json", ttl_hours: float = 24.0, decay_rate: float = 0.95):
        self.storage_path = storage_path
        self.ttl_hours = ttl_hours
        self.decay_rate = decay_rate
        self.templates: Dict[str, Template] = {}
        self.last_cleanup = time.time()
        self.cleanup_interval = 3600  # Cleanup every hour
        
        # Setup module-local logger
        self.logger = logging.getLogger(__name__)
        
        # Load existing templates
        self.load()
    
    def add_template(self, template: Template) -> bool:
        """Add template to library, return True if new"""
        if template.signature not in self.templates:
            self.templates[template.signature] = template
            self.logger.info(f"Added new template: {template.signature} (support={template.support}, score={template.score:.3f})")
            return True
        else:
            # Update existing template with better score
            existing = self.templates[template.signature]
            if template.score > existing.score:
                existing.score = template.score
                existing.support = max(existing.support, template.support)
                existing.usage_count += 1
                self.logger.info(f"Updated template: {template.signature} (score={template.score:.3f}, support={template.support})")
            return False
    
    def get_templates_by_score(self, min_score: float = 0.0, max_count: int = 100) -> List[Template]:
        """Get templates sorted by score"""
        self._cleanup_if_needed()
        templates = [t for t in self.templates.values() if t.score >= min_score]
        return sorted(templates, key=lambda t: t.score, reverse=True)[:max_count]
    
    def get_templates_by_ops(self, op_names: Set[str]) -> List[Template]:
        """Get templates containing specific operations"""
        self._cleanup_if_needed()
        matching = []
        for template in self.templates.values():
            template_ops = set(template.get_primitive_ops())
            if op_names.issubset(template_ops):
                matching.append(template)
        return sorted(matching, key=lambda t: t.score, reverse=True)
    
    def _cleanup_if_needed(self):
        """Clean up expired templates if needed"""
        now = time.time()
        if now - self.last_cleanup < self.cleanup_interval:
            return
        
        self.last_cleanup = now
        expired_signatures = []
        
        for sig, template in self.templates.items():
            age_hours = (now - template.created_at) / 3600
            
            # Apply decay to score
            decay_factor = self.decay_rate ** age_hours
            template.score *= decay_factor
            
            # Remove if expired or score too low
            if age_hours > self.ttl_hours or template.score < 0.001:
                expired_signatures.append(sig)
        
        # Remove expired templates
        for sig in expired_signatures:
            del self.templates[sig]
            self.logger.info(f"Removed expired template: {sig}")
    
    def save(self):
        """Save templates to disk"""
        try:
            data = {sig: template.to_dict() for sig, template in self.templates.items()}
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
            self.logger.info(f"Saved {len(self.templates)} templates to {self.storage_path}")
        except Exception as e:
            self.logger.error(f"Failed to save templates: {e}")
    
    def load(self):
        """Load templates from disk"""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                
                for sig, template_data in data.items():
                    template = Template.from_dict(template_data)
                    self.templates[sig] = template
                
                self.logger.info(f"Loaded {len(self.templates)} templates from {self.storage_path}")
        except Exception as e:
            self.logger.error(f"Failed to load templates: {e}")
    
    def __len__(self):
        return len(self.templates)

class WormholeConsolidator:
    """Consolidates mined program fragments into reusable templates with MDL-style compression gains"""
    
    def __init__(self, 
                 min_support: int = 2,
                 compression_threshold: float = 0.5,
                 lambda_cost: float = 1.0,
                 max_templates: int = 1000,
                 template_storage: str = "templates.json"):
        self.min_support = min_support
        self.compression_threshold = compression_threshold
        self.lambda_cost = lambda_cost  # Cost coefficient for MDL scoring
        self.max_templates = max_templates
        
        # Template library
        self.library = TemplateLibrary(template_storage)
        
        # Setup module-local logger only
        self.logger = logging.getLogger(__name__)
    
    def _compute_signature(self, ops: List[Tuple[str, dict]]) -> str:
        """Compute unique signature for a sequence of operations"""
        # Create stable string representation using sorted parameter tuples
        sig_parts = []
        for op_name, params in ops:
            if params:
                # Use sorted tuple of items for stability
                param_str = str(tuple(sorted(params.items())))
            else:
                param_str = ""
            sig_parts.append(f"{op_name}:{param_str}")
        
        sig_str = "|".join(sig_parts)
        return hashlib.md5(sig_str.encode()).hexdigest()[:16]
    
    def _compute_template_cost(self, ops: List[Tuple[str, dict]]) -> float:
        """Compute cost of storing a template (MDL principle)"""
        # Base cost: number of operations
        cost = len(ops)
        
        # Parameter complexity cost
        for _, params in ops:
            if params:
                # Add cost for each parameter
                cost += len(params)
                # Add cost for complex parameter values
                for value in params.values():
                    if isinstance(value, (list, dict)):
                        cost += len(str(value)) * 0.1  # String length as proxy
                    elif isinstance(value, (int, float)):
                        cost += 0.5  # Simple numeric parameters are cheaper
        
        return cost
    
    def _compute_mdl_gain(self, 
                         original_programs: List[List[Tuple[str, dict]]], 
                         template: List[Tuple[str, dict]], 
                         usage_count: int) -> float:
        """Compute MDL compression gain for a template"""
        # Original total length (sum of all program lengths that use this template)
        original_lengths = sum(len(prog) for prog in original_programs)
        
        # Template cost (cost to store the template definition)
        template_cost = self._compute_template_cost(template)
        
        # Templated lengths: for each program using the template:
        # templated_len = 1 (reference) + len(noncovered_ops)
        templated_lengths = 0
        template_ops = [op for op, _ in template]
        
        for program in original_programs:
            # Count ops not covered by template (simplified: assume template covers start of program)
            noncovered_ops = len(program) - len(template_ops) if len(program) >= len(template_ops) else len(program)
            noncovered_ops = max(0, noncovered_ops)  # Ensure non-negative
            templated_len = 1 + noncovered_ops  # 1 reference + uncovered ops
            templated_lengths += templated_len
        
        # MDL gain: (original_lengths - templated_lengths) - λ*template_cost
        gain = (original_lengths - templated_lengths) - self.lambda_cost * template_cost
        
        return gain
    
    def consolidate(self, programs: List[List[Tuple[str, dict]]], top_k: int = 50) -> List[Template]:
        """Create templates from program patterns with MDL-style scoring"""
        if not programs:
            return []
        
        self.logger.info(f"Consolidating {len(programs)} programs into templates...")
        
        # Extract all subsequences with their frequencies
        pattern_frequencies = defaultdict(list)  # pattern -> list of source programs
        
        for prog_idx, program in enumerate(programs):
            # Extract subsequences of length 1 to min(len(program), 4)
            for start in range(len(program)):
                for end in range(start + 1, min(start + 5, len(program) + 1)):
                    # Convert to hashable format: tuple of (op_name, sorted tuple of params)
                    pattern_items = []
                    for op_name, params in program[start:end]:
                        if isinstance(params, dict):
                            param_key = tuple(sorted(params.items())) if params else tuple()
                        else:
                            param_key = tuple()
                        pattern_items.append((op_name, param_key))
                    pattern = tuple(pattern_items)
                    pattern_frequencies[pattern].append(prog_idx)
        
        # Convert to templates and score them
        templates = []
        for pattern, prog_indices in pattern_frequencies.items():
            support = len(prog_indices)
            
            # Filter by minimum support
            if support < self.min_support:
                continue
            
            # Convert pattern back to list of (op_name, params) tuples
            ops = []
            for op_name, param_key in pattern:
                if param_key:
                    params = dict(param_key)  # param_key is now tuple of sorted items
                else:
                    params = {}
                ops.append((op_name, params))
            
            # Compute signature
            signature = self._compute_signature(ops)
            
            # Get original programs that use this pattern
            original_programs = [programs[i] for i in prog_indices]
            
            # Compute MDL gain
            mdl_gain = self._compute_mdl_gain(original_programs, ops, support)
            
            # Only keep templates with positive compression gain
            if mdl_gain > self.compression_threshold:
                template = Template(
                    ops=ops,
                    support=support,
                    score=mdl_gain,
                    signature=signature
                )
                templates.append(template)
                self.logger.info(f"Created template {signature}: {len(ops)} ops, support={support}, gain={mdl_gain:.3f}")
        
        # Deduplicate by signature and keep top-K by score
        unique_templates = {}
        for template in templates:
            if template.signature not in unique_templates:
                unique_templates[template.signature] = template
            else:
                # Keep the one with higher score
                existing = unique_templates[template.signature]
                if template.score > existing.score:
                    unique_templates[template.signature] = template
        
        # Sort by score and take top-K
        final_templates = sorted(unique_templates.values(), key=lambda t: t.score, reverse=True)[:top_k]
        
        # Add to library
        new_count = 0
        for template in final_templates:
            if self.library.add_template(template):
                new_count += 1
        
        # Save library
        self.library.save()
        
        self.logger.info(f"Consolidation complete: {len(final_templates)} templates generated, {new_count} new")
        return final_templates
    
    def get_library(self) -> TemplateLibrary:
        """Get the template library"""
        return self.library
    
    def get_best_templates(self, count: int = 20) -> List[Template]:
        """Get the best templates by score"""
        return self.library.get_templates_by_score(max_count=count)

class WormholeTemplateMiner:
    """Enhanced template miner with MDL compression and TTL/decay"""
    
    def __init__(self, library_path: str = "wormhole_library.json", kappa: float = 1/1.6487212707, ripple_freq: int = 200):
        self.kappa = kappa
        self.ripple_freq = ripple_freq
        self.templates = {}  # signature → Template
        self.mdl_scores = {}  # Minimum Description Length scores
        self.ttl_decay = {}  # Time-To-Live counters
        self.library_path = library_path
        self.default_ttl = 100  # Default TTL for new templates
        
        # Initialize template library and consolidator
        self.library = TemplateLibrary(library_path)
        self.consolidator = WormholeConsolidator(
            min_support=2, 
            compression_threshold=0.1,
            template_storage=library_path
        )
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Load existing templates
        self.load_library()
    
    def mine_template(self, program: List[Tuple[str, dict]], demos: List[Tuple]) -> Optional[Template]:
        """Extract micro-program patterns and score by MDL"""
        try:
            # Convert single program to list for consolidator
            programs = [program] if program else []
            
            if not programs:
                return None
            
            # Extract patterns from this program
            patterns = self._extract_patterns(program)
            
            # Score each pattern by MDL
            best_template = None
            best_mdl = float('inf')
            
            for pattern in patterns:
                if len(pattern) < 2:  # Skip single operations
                    continue
                
                # Compute MDL score
                mdl_score = self._compute_mdl(pattern, [program], demos)
                
                if mdl_score < best_mdl:
                    best_mdl = mdl_score
                    
                    # Create template
                    signature = self._compute_signature(pattern)
                    template = Template(
                        ops=pattern,
                        support=1,
                        score=1.0 / (mdl_score + 1e-6),  # Convert MDL to score (lower MDL = higher score)
                        signature=signature,
                        created_at=time.time()
                    )
                    best_template = template
            
            if best_template:
                # Add TTL tracking
                self.ttl_decay[best_template.signature] = self.default_ttl
                self.mdl_scores[best_template.signature] = best_mdl
                self.templates[best_template.signature] = best_template
                
                self.logger.info(f"Mined template {best_template.signature}: {len(best_template.ops)} ops, MDL={best_mdl:.3f}")
            
            return best_template
            
        except Exception as e:
            self.logger.error(f"Template mining failed: {e}")
            return None
    
    def _extract_patterns(self, program: List[Tuple[str, dict]]) -> List[List[Tuple[str, dict]]]:
        """Extract all meaningful subsequences from a program"""
        patterns = []
        
        # Extract subsequences of length 2-4
        for start in range(len(program)):
            for end in range(start + 2, min(start + 5, len(program) + 1)):
                pattern = program[start:end]
                patterns.append(pattern)
        
        return patterns
    
    def _compute_mdl(self, pattern: List[Tuple[str, dict]], programs: List[List], demos: List[Tuple]) -> float:
        """Compute Minimum Description Length for a pattern"""
        try:
            # Program length cost (number of operations + parameter complexity)
            prog_cost = len(pattern)
            for _, params in pattern:
                if params:
                    prog_cost += len(params) * 0.5  # Parameter cost
            
            # Data encoding cost given pattern (how well it explains the data)
            data_cost = 0.0
            for demo_input, demo_output in demos:
                # Estimate how much this pattern contributes to solving the demo
                # Simplified: use pattern length as proxy for encoding cost
                data_cost += len(pattern) * 0.1
            
            # Total MDL = program cost + data cost
            mdl = prog_cost + data_cost
            
            return mdl
            
        except Exception as e:
            self.logger.error(f"MDL computation failed: {e}")
            return float('inf')
    
    def _compute_signature(self, pattern: List[Tuple[str, dict]]) -> str:
        """Compute unique signature for a pattern"""
        sig_parts = []
        for op_name, params in pattern:
            if params:
                param_str = str(tuple(sorted(params.items())))
            else:
                param_str = ""
            sig_parts.append(f"{op_name}:{param_str}")
        
        sig_str = "|".join(sig_parts)
        return hashlib.md5(sig_str.encode()).hexdigest()[:16]
    
    def persist_library(self):
        """Save templates to disk with TTL information"""
        try:
            # Combine template data with TTL info
            library_data = {
                'templates': {sig: template.to_dict() for sig, template in self.templates.items()},
                'ttl_decay': self.ttl_decay,
                'mdl_scores': self.mdl_scores,
                'saved_at': time.time()
            }
            
            with open(self.library_path, 'w') as f:
                json.dump(library_data, f, indent=2)
            
            self.logger.info(f"Persisted {len(self.templates)} templates to {self.library_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to persist library: {e}")
    
    def load_library(self):
        """Load templates from disk with TTL information"""
        try:
            if os.path.exists(self.library_path):
                with open(self.library_path, 'r') as f:
                    library_data = json.load(f)
                
                # Load templates
                if 'templates' in library_data:
                    for sig, template_data in library_data['templates'].items():
                        template = Template.from_dict(template_data)
                        self.templates[sig] = template
                
                # Load TTL and MDL data
                self.ttl_decay = library_data.get('ttl_decay', {})
                self.mdl_scores = library_data.get('mdl_scores', {})
                
                self.logger.info(f"Loaded {len(self.templates)} templates from {self.library_path}")
        
        except Exception as e:
            self.logger.error(f"Failed to load library: {e}")
    
    def cleanup_stale(self):
        """Remove templates with expired TTL"""
        expired_signatures = []
        
        for sig in list(self.ttl_decay.keys()):
            self.ttl_decay[sig] -= 1  # Decrement TTL
            
            if self.ttl_decay[sig] <= 0:
                expired_signatures.append(sig)
        
        # Remove expired templates
        for sig in expired_signatures:
            if sig in self.templates:
                del self.templates[sig]
            if sig in self.ttl_decay:
                del self.ttl_decay[sig]
            if sig in self.mdl_scores:
                del self.mdl_scores[sig]
            
            self.logger.info(f"Removed expired template: {sig}")
        
        return len(expired_signatures)
    
    def tick(self):
        """Decrement TTLs (called after each task)"""
        for sig in self.ttl_decay:
            self.ttl_decay[sig] = max(0, self.ttl_decay[sig] - 1)
    
    def refresh_ttl(self, template_id: str, bonus_ttl: int = 50):
        """Refresh TTL for successful templates"""
        if template_id in self.ttl_decay:
            self.ttl_decay[template_id] += bonus_ttl
            self.logger.info(f"Refreshed TTL for template {template_id}: +{bonus_ttl}")
    
    def get_relevant_templates(self, demos: List[Tuple], max_count: int = 10) -> List[Template]:
        """Get templates most relevant for given demonstrations"""
        # Score templates by relevance and MDL
        scored_templates = []
        
        for sig, template in self.templates.items():
            if self.ttl_decay.get(sig, 0) <= 0:
                continue  # Skip expired templates
            
            # Combine template score with inverse MDL (lower MDL = better)
            mdl_score = self.mdl_scores.get(sig, 1.0)
            combined_score = template.score * (1.0 / (mdl_score + 1e-6))
            
            scored_templates.append((combined_score, template))
        
        # Sort by score and return top templates
        scored_templates.sort(key=lambda x: x[0], reverse=True)
        return [template for _, template in scored_templates[:max_count]]
    
    def mine_from_programs(self, programs, top_k: int = 5):
        """Legacy interface for backward compatibility"""
        subseq_scores = {}
        for prog in programs:
            # Handle both list of strings and list of tuples
            if prog and isinstance(prog[0], tuple):
                names = [op for op, _ in prog]
            else:
                names = prog  # Already list of operation names
            
            for i in range(len(names)):
                for j in range(i+1, min(i+4, len(names))+1):
                    tpl = tuple(names[i:j])
                    subseq_scores[tpl] = subseq_scores.get(tpl, 0)+1
        ranked = sorted(subseq_scores.items(), key=lambda x: (-x[1], len(x[0])))
        return [tpl for tpl,_ in ranked[:top_k]]

    def mine_with_consolidator(self, patterns, top_k: int = 5):
        """Mine patterns using the WormholeConsolidator for better template extraction"""
        # Convert patterns to program format if needed
        programs = []
        for pattern in patterns:
            if isinstance(pattern, (list, tuple)):
                # Convert to ops format: [(op_name, params), ...]
                program = []
                for item in pattern:
                    if isinstance(item, str):
                        # Simple operation name
                        program.append((item, {}))
                    elif isinstance(item, tuple) and len(item) == 2:
                        # (op_name, params) tuple
                        program.append(item)
                programs.append(program)
        
        if not programs:
            return []
        
        # Use consolidator to create templates
        templates = self.consolidator.consolidate(programs, top_k=top_k)
        
        # Add to our template collection
        for template in templates:
            self.templates[template.signature] = template
            self.ttl_decay[template.signature] = self.default_ttl
            self.mdl_scores[template.signature] = 1.0 / (template.score + 1e-6)
        
        # Return template operations as the mined patterns
        return [template.get_primitive_ops() for template in templates]
