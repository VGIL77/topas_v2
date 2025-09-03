import torch, torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional, Any

class HierarchicalObjectSlots(nn.Module):
    """
    Hierarchical Object-centric slot attention with multi-level reasoning.
    
    Architecture:
    - Level 1 (Pixels): Fine-grained patches/pixel groups (64 slots)
    - Level 2 (Objects): Connected components/objects (16 slots) [ORIGINAL]  
    - Level 3 (Groups): Pattern clusters/symmetries (8 slots)
    - Level 4 (Scene): Global scene relationships (4 slots)
    
    Features:
    - Cross-level attention (bidirectional information flow)
    - Graph reasoning for spatial relationships
    - Symmetry detection (rotation, reflection, translation)
    - Compositional structure (part-whole relationships)
    - Backward compatible with existing ObjectSlots interface
    """
    
    def __init__(self, in_ch, K=16, slot_dim=192, num_iters=3, hierarchical=True):
        super().__init__()
        
        # Core parameters (maintain compatibility)
        self.K = K  # Main level (objects) slot count
        self.slot_dim = slot_dim
        self.in_ch = in_ch
        self.num_iters = num_iters
        self.hierarchical = hierarchical
        
        # Hierarchical slot counts
        self.K_pixels = 64      # Level 1: Fine-grained patches
        self.K_objects = K      # Level 2: Main objects (original)
        self.K_groups = 8       # Level 3: Pattern groups
        self.K_scene = 4        # Level 4: Scene-level
        
        # Initialize all slot levels
        if hierarchical:
            self._init_hierarchical_slots()
        else:
            # Fallback to original ObjectSlots behavior
            self._init_flat_slots()
            
        # Cross-level attention mechanisms
        if hierarchical:
            self._init_cross_level_attention()
            
        # Graph reasoning components
        self._init_graph_reasoning()
        
        # Symmetry detection
        self._init_symmetry_detection()
        
        # Compositional structure
        self._init_compositional_reasoning()
        
    def _init_hierarchical_slots(self):
        """Initialize hierarchical slot embeddings and attention mechanisms"""
        
        # Slot embeddings for each level
        self.pixel_slots = nn.Parameter(torch.randn(1, self.K_pixels, self.slot_dim) * 0.3)
        self.object_slots = nn.Parameter(torch.randn(1, self.K_objects, self.slot_dim) * 0.5)
        self.group_slots = nn.Parameter(torch.randn(1, self.K_groups, self.slot_dim) * 0.7)
        self.scene_slots = nn.Parameter(torch.randn(1, self.K_scene, self.slot_dim) * 1.0)
        
        # Initialize with diversity
        with torch.no_grad():
            for level_slots, level_name in [
                (self.pixel_slots, "pixel"), (self.object_slots, "object"),
                (self.group_slots, "group"), (self.scene_slots, "scene")
            ]:
                K_level = level_slots.shape[1]
                for k in range(K_level):
                    level_slots[0, k] += torch.randn(self.slot_dim) * 0.1 * k
        
        # Attention projections for each level
        self.levels_config = {
            'pixel': {'slots': self.pixel_slots, 'K': self.K_pixels},
            'object': {'slots': self.object_slots, 'K': self.K_objects},
            'group': {'slots': self.group_slots, 'K': self.K_groups},
            'scene': {'slots': self.scene_slots, 'K': self.K_scene}
        }
        
        # Shared attention projections (all levels use same input features)
        self.to_k = nn.Linear(self.in_ch, self.slot_dim, bias=False)
        self.to_v = nn.Linear(self.in_ch, self.slot_dim, bias=False)
        
        # Level-specific query projections
        self.to_q_pixel = nn.Linear(self.slot_dim, self.slot_dim, bias=False)
        self.to_q_object = nn.Linear(self.slot_dim, self.slot_dim, bias=False)
        self.to_q_group = nn.Linear(self.slot_dim, self.slot_dim, bias=False)
        self.to_q_scene = nn.Linear(self.slot_dim, self.slot_dim, bias=False)
        
        # Level-specific MLPs for slot updates
        self.pixel_mlp = self._make_slot_mlp()
        self.object_mlp = self._make_slot_mlp()
        self.group_mlp = self._make_slot_mlp()
        self.scene_mlp = self._make_slot_mlp()
        
        # Layer norms for each level
        self.norm_slots_pixel = nn.LayerNorm(self.slot_dim)
        self.norm_slots_object = nn.LayerNorm(self.slot_dim)
        self.norm_slots_group = nn.LayerNorm(self.slot_dim)
        self.norm_slots_scene = nn.LayerNorm(self.slot_dim)
        
        self.norm_inputs = nn.LayerNorm(self.slot_dim)
        
    def _init_flat_slots(self):
        """Initialize flat slots (original ObjectSlots behavior)"""
        self.slot_embed = nn.Parameter(torch.randn(1, self.K, self.slot_dim) * 0.5)
        
        with torch.no_grad():
            for k in range(self.K):
                self.slot_embed[0, k] += torch.randn(self.slot_dim) * 0.2 * k
        
        self.to_k = nn.Linear(self.in_ch, self.slot_dim, bias=False)
        self.to_v = nn.Linear(self.in_ch, self.slot_dim, bias=False)
        self.to_q = nn.Linear(self.slot_dim, self.slot_dim, bias=False)
        
        self.slot_mlp = self._make_slot_mlp()
        self.norm_slots = nn.LayerNorm(self.slot_dim)
        self.norm_inputs = nn.LayerNorm(self.slot_dim)
        
    def _make_slot_mlp(self):
        """Create slot update MLP"""
        return nn.Sequential(
            nn.LayerNorm(self.slot_dim),
            nn.Linear(self.slot_dim, self.slot_dim * 2),
            nn.GELU(),
            nn.Linear(self.slot_dim * 2, self.slot_dim)
        )
        
    def _init_cross_level_attention(self):
        """Initialize cross-level attention mechanisms"""
        
        # Bottom-up attention (pixel -> object -> group -> scene)
        self.pixel_to_object_attn = CrossLevelAttention(self.slot_dim)
        self.object_to_group_attn = CrossLevelAttention(self.slot_dim)
        self.group_to_scene_attn = CrossLevelAttention(self.slot_dim)
        
        # Top-down attention (scene -> group -> object -> pixel)
        self.scene_to_group_attn = CrossLevelAttention(self.slot_dim)
        self.group_to_object_attn = CrossLevelAttention(self.slot_dim)
        self.object_to_pixel_attn = CrossLevelAttention(self.slot_dim)
        
    def _init_graph_reasoning(self):
        """Initialize graph reasoning components"""
        
        # Spatial relationship encoder
        self.spatial_encoder = SpatialRelationEncoder(self.slot_dim)
        
        # Graph neural network for connectivity
        self.graph_reasoning = GraphAttentionNetwork(
            node_dim=self.slot_dim,
            edge_dim=32,
            num_heads=4,
            num_layers=2
        )
        
        # Adjacency matrix predictor
        self.adjacency_predictor = nn.Sequential(
            nn.Linear(self.slot_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def _init_symmetry_detection(self):
        """Initialize symmetry detection components"""
        
        # Rotation symmetry detector (90°, 180°, 270°)
        self.rotation_detector = SymmetryDetector(
            self.slot_dim, symmetry_type='rotation'
        )
        
        # Reflection symmetry detector (H, V, D1, D2)
        self.reflection_detector = SymmetryDetector(
            self.slot_dim, symmetry_type='reflection'
        )
        
        # Translation symmetry detector (repeating patterns)
        self.translation_detector = SymmetryDetector(
            self.slot_dim, symmetry_type='translation'
        )
        
        # Group theory operations
        self.group_ops = GroupOperations(self.slot_dim)
        
    def _init_compositional_reasoning(self):
        """Initialize compositional structure reasoning"""
        
        # Part-whole relationship detector
        self.part_whole_detector = nn.Sequential(
            nn.Linear(self.slot_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Hierarchy structure predictor
        self.hierarchy_predictor = HierarchyPredictor(self.slot_dim)
        
        # Compositional attention
        self.composition_attn = nn.MultiheadAttention(
            embed_dim=self.slot_dim,
            num_heads=8,
            batch_first=True
        )

    @property
    def out_dim(self):
        """Return output dimension (backward compatibility)"""
        return self.slot_dim
        
    def forward(self, feat):
        """
        Hierarchical forward pass with multi-level reasoning
        
        Args:
            feat: Input features [B, C, H, W]
            
        Returns:
            slot_vecs: Main object slots [B, K, slot_dim] (backward compatible)
            attention_weights: Object-level attention [B, K, H*W] (backward compatible)
            hierarchical_features: Dict with all hierarchical outputs
        """
        B, C, H, W = feat.shape
        
        if not self.hierarchical:
            # Fallback to original ObjectSlots behavior
            return self._forward_flat(feat)
            
        # Flatten spatial dimensions
        tokens = feat.view(B, C, H * W).transpose(1, 2)  # [B, H*W, C]
        
        # Project tokens to key-value space (shared across levels)
        k = self.to_k(tokens)  # [B, H*W, slot_dim]
        v = self.to_v(tokens)  # [B, H*W, slot_dim]
        
        # Initialize all slot levels
        pixel_slots = self.pixel_slots.expand(B, -1, -1)
        object_slots = self.object_slots.expand(B, -1, -1)
        group_slots = self.group_slots.expand(B, -1, -1)
        scene_slots = self.scene_slots.expand(B, -1, -1)
        
        # Scale for attention
        scale = self.slot_dim ** -0.5
        
        # Store attention weights and intermediate results
        all_attention = {}
        all_slots = {}
        
        # Iterative refinement with hierarchical reasoning
        for iteration in range(self.num_iters):
            
            # Process each level with cross-attention
            pixel_slots, pixel_attn = self._process_level(
                pixel_slots, k, v, self.to_q_pixel, self.pixel_mlp,
                self.norm_slots_pixel, scale, f"pixel_{iteration}"
            )
            
            object_slots, object_attn = self._process_level(
                object_slots, k, v, self.to_q_object, self.object_mlp,
                self.norm_slots_object, scale, f"object_{iteration}"
            )
            
            group_slots, group_attn = self._process_level(
                group_slots, k, v, self.to_q_group, self.group_mlp,
                self.norm_slots_group, scale, f"group_{iteration}"
            )
            
            scene_slots, scene_attn = self._process_level(
                scene_slots, k, v, self.to_q_scene, self.scene_mlp,
                self.norm_slots_scene, scale, f"scene_{iteration}"
            )
            
            # Cross-level information flow
            if iteration > 0:  # Allow first iteration to establish initial representations
                
                # Bottom-up flow (fine to coarse)
                object_slots = object_slots + self.pixel_to_object_attn(pixel_slots, object_slots)
                group_slots = group_slots + self.object_to_group_attn(object_slots, group_slots)
                scene_slots = scene_slots + self.group_to_scene_attn(group_slots, scene_slots)
                
                # Top-down flow (coarse to fine)
                group_slots = group_slots + self.scene_to_group_attn(scene_slots, group_slots)
                object_slots = object_slots + self.group_to_object_attn(group_slots, object_slots)
                pixel_slots = pixel_slots + self.object_to_pixel_attn(object_slots, pixel_slots)
            
            # Store attention maps
            all_attention[f"pixel_{iteration}"] = pixel_attn
            all_attention[f"object_{iteration}"] = object_attn
            all_attention[f"group_{iteration}"] = group_attn
            all_attention[f"scene_{iteration}"] = scene_attn
        
        # Store final slots
        all_slots['pixel'] = pixel_slots
        all_slots['object'] = object_slots
        all_slots['group'] = group_slots
        all_slots['scene'] = scene_slots
        
        # Graph reasoning on object level
        spatial_relations = self._compute_spatial_relations(object_slots, H, W)
        graph_enhanced_objects = self.graph_reasoning(object_slots, spatial_relations)
        
        # Symmetry detection across all levels
        symmetries = self._detect_symmetries(all_slots, H, W)
        
        # Compositional structure analysis
        compositions = self._analyze_compositions(all_slots)
        
        # Prepare hierarchical features output
        hierarchical_features = {
            'all_slots': all_slots,
            'all_attention': all_attention,
            'spatial_relations': spatial_relations,
            'symmetries': symmetries,
            'compositions': compositions,
            'graph_enhanced_objects': graph_enhanced_objects,
            'hierarchy_structure': self._build_hierarchy_structure(all_slots)
        }
        
        # Return backward-compatible format
        # Main outputs are object-level for compatibility
        return graph_enhanced_objects, object_attn, hierarchical_features
    
    def _forward_flat(self, feat):
        """Original ObjectSlots forward pass (backward compatibility)"""
        B, C, H, W = feat.shape
        
        tokens = feat.view(B, C, H * W).transpose(1, 2)
        k = self.to_k(tokens)
        v = self.to_v(tokens)
        
        slots = self.slot_embed.expand(B, -1, -1)
        attn_weights = None
        scale = self.slot_dim ** -0.5
        
        for _ in range(self.num_iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            
            q = self.to_q(slots)
            dots = torch.einsum('bkd,bnd->bkn', q, k) * scale
            
            temperature = 2.0
            dots = dots / temperature
            attn = F.softmax(dots, dim=-1)
            
            attn_competitive = F.softmax(dots.transpose(1, 2) * temperature, dim=-1).transpose(1, 2)
            attn = attn * attn_competitive
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
            
            attn_weights = attn
            updates = torch.einsum('bkn,bnd->bkd', attn, v)
            updates = self.norm_inputs(updates)
            slots = slots_prev + self.slot_mlp(updates)
        
        return slots, attn_weights
    
    def _process_level(self, slots, k, v, to_q, slot_mlp, norm_slots, scale, level_name):
        """Process a single hierarchical level with cross-attention"""
        
        slots_prev = slots
        slots = norm_slots(slots)
        
        # Query projection for this level
        q = to_q(slots)
        
        # Compute attention
        dots = torch.einsum('bkd,bnd->bkn', q, k) * scale
        
        # Temperature and competitive normalization
        temperature = 2.0
        dots = dots / temperature
        attn = F.softmax(dots, dim=-1)
        
        attn_competitive = F.softmax(dots.transpose(1, 2) * temperature, dim=-1).transpose(1, 2)
        attn = attn * attn_competitive
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Update slots
        updates = torch.einsum('bkn,bnd->bkd', attn, v)
        updates = self.norm_inputs(updates)
        slots = slots_prev + slot_mlp(updates)
        
        return slots, attn
    
    def _compute_spatial_relations(self, object_slots, H, W):
        """Compute spatial relationships between objects"""
        return self.spatial_encoder(object_slots, H, W)
    
    def _detect_symmetries(self, all_slots, H, W):
        """Detect symmetries across all hierarchical levels"""
        symmetries = {}
        
        for level_name, slots in all_slots.items():
            level_symmetries = {}
            
            # Rotation symmetries
            level_symmetries['rotation'] = self.rotation_detector(slots, H, W)
            
            # Reflection symmetries  
            level_symmetries['reflection'] = self.reflection_detector(slots, H, W)
            
            # Translation symmetries
            level_symmetries['translation'] = self.translation_detector(slots, H, W)
            
            symmetries[level_name] = level_symmetries
            
        return symmetries
    
    def _analyze_compositions(self, all_slots):
        """Analyze compositional structure between levels"""
        compositions = {}
        
        # Part-whole relationships between adjacent levels
        compositions['pixel_to_object'] = self._compute_part_whole(
            all_slots['pixel'], all_slots['object']
        )
        compositions['object_to_group'] = self._compute_part_whole(
            all_slots['object'], all_slots['group']
        )
        compositions['group_to_scene'] = self._compute_part_whole(
            all_slots['group'], all_slots['scene']
        )
        
        return compositions
    
    def _compute_part_whole(self, part_slots, whole_slots):
        """Compute part-whole relationships between two slot levels"""
        B, K_part, D = part_slots.shape
        B, K_whole, D = whole_slots.shape
        
        # Expand for pairwise comparison
        parts_expanded = part_slots.unsqueeze(2).expand(-1, -1, K_whole, -1)  # [B, K_part, K_whole, D]
        wholes_expanded = whole_slots.unsqueeze(1).expand(-1, K_part, -1, -1)  # [B, K_part, K_whole, D]
        
        # Concatenate for relationship prediction
        pairs = torch.cat([parts_expanded, wholes_expanded], dim=-1)  # [B, K_part, K_whole, 2*D]
        
        # Predict part-whole relationships
        relationships = self.part_whole_detector(pairs.view(-1, 2 * self.slot_dim))
        relationships = relationships.view(B, K_part, K_whole)
        
        return relationships
    
    def _build_hierarchy_structure(self, all_slots):
        """Build explicit hierarchy structure representation"""
        return self.hierarchy_predictor(all_slots)
    
    def extract_object_masks(self, grid: torch.Tensor, slot_attention: torch.Tensor, threshold: float = 0.1) -> List[torch.Tensor]:
        """
        Extract connected-component masks from slot attention weights
        
        Args:
            grid: Input grid [H, W] or [B, H, W]
            slot_attention: Attention weights [B, K, H*W] or [K, H*W]
            threshold: Attention threshold for mask creation
            
        Returns:
            List of object masks, each [H, W] with connected components
        """
        if grid.dim() == 3:
            grid = grid[0]  # Remove batch dimension
        if slot_attention.dim() == 3:
            slot_attention = slot_attention[0]  # Remove batch dimension
            
        H, W = grid.shape
        K = slot_attention.shape[0]
        
        # Reshape attention to spatial dimensions
        attn_spatial = slot_attention.view(K, H, W)  # [K, H, W]
        
        masks = []
        
        for slot_idx in range(K):
            # Get attention for this slot
            attn = attn_spatial[slot_idx]  # [H, W]
            
            # Threshold attention to get binary mask
            binary_mask = (attn > threshold).float()
            
            # Skip empty masks
            if binary_mask.sum() == 0:
                continue
            
            # Extract connected components within the attention mask
            object_masks = self._get_connected_components_from_mask(grid, binary_mask)
            masks.extend(object_masks)
        
        return masks
    
    def extract_object_features(self, grid: torch.Tensor, attention_weights: torch.Tensor) -> List[Dict[str, Any]]:
        """
        Compute rich object features from grid and attention weights
        
        Args:
            grid: Input grid [H, W] or [B, H, W]
            attention_weights: Attention weights [B, K, H*W] or [K, H*W]
            
        Returns:
            List of object feature dictionaries containing:
            - mask: object binary mask [H, W]
            - area: total area (number of pixels)
            - bbox: bounding box [y_min, x_min, y_max, x_max]
            - centroid: center of mass [y, x]
            - color_set: set of colors in the object
            - adjacency: list of adjacent object indices
            - shape_signature: geometric shape descriptor
            - symmetry: symmetry properties
            - holes: number of holes in the object
        """
        masks = self.extract_object_masks(grid, attention_weights)
        
        features = []
        for i, mask in enumerate(masks):
            obj_features = {
                'mask': mask,
                'area': mask.sum().item(),
                'bbox': self.compute_bbox(mask),
                'centroid': self.compute_centroid(mask),
                'color_set': self.get_colors(grid, mask),
                'adjacency': [],  # Will be filled later
                'shape_signature': self.compute_shape_signature(mask),
                'symmetry': self.detect_symmetry(mask),
                'holes': self.count_holes(mask)
            }
            features.append(obj_features)
        
        # Compute adjacency relationships
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                if self.objects_adjacent(features[i]['mask'], features[j]['mask']):
                    features[i]['adjacency'].append(j)
                    features[j]['adjacency'].append(i)
        
        return features
    
    def compute_bbox(self, mask: torch.Tensor) -> Tuple[int, int, int, int]:
        """Compute bounding box of mask"""
        if mask.sum() == 0:
            return (0, 0, 0, 0)
        
        y_coords, x_coords = torch.nonzero(mask, as_tuple=True)
        y_min, y_max = y_coords.min().item(), y_coords.max().item()
        x_min, x_max = x_coords.min().item(), x_coords.max().item()
        return (y_min, x_min, y_max, x_max)
    
    def compute_centroid(self, mask: torch.Tensor) -> Tuple[float, float]:
        """Compute center of mass of mask"""
        if mask.sum() == 0:
            return (0.0, 0.0)
        
        y_coords, x_coords = torch.nonzero(mask, as_tuple=True)
        centroid_y = y_coords.float().mean().item()
        centroid_x = x_coords.float().mean().item()
        return (centroid_y, centroid_x)
    
    def get_colors(self, grid: torch.Tensor, mask: torch.Tensor) -> set:
        """Get set of colors in masked region"""
        if grid.dim() == 3:
            grid = grid[0]
        
        masked_grid = grid * mask.long()
        colors = torch.unique(masked_grid[mask > 0])
        return set(colors.cpu().tolist())
    
    def compute_shape_signature(self, mask: torch.Tensor) -> Dict[str, float]:
        """Compute geometric shape descriptors"""
        if mask.sum() == 0:
            return {'aspect_ratio': 1.0, 'compactness': 0.0, 'convexity': 0.0, 'extent': 0.0}
        
        y_min, x_min, y_max, x_max = self.compute_bbox(mask)
        height = y_max - y_min + 1
        width = x_max - x_min + 1
        
        # Aspect ratio
        aspect_ratio = width / max(height, 1)
        
        # Compactness (circularity)
        area = mask.sum().item()
        perimeter = self._compute_perimeter(mask)
        compactness = 4 * math.pi * area / max(perimeter ** 2, 1) if perimeter > 0 else 0.0
        
        # Extent (proportion of bounding box filled)
        bbox_area = height * width
        extent = area / max(bbox_area, 1)
        
        # Convexity approximation (area / convex_hull_area)
        # Simple approximation using bbox area
        convexity = extent  # Simplified
        
        return {
            'aspect_ratio': aspect_ratio,
            'compactness': compactness,
            'convexity': convexity,
            'extent': extent
        }
    
    def _compute_perimeter(self, mask: torch.Tensor) -> int:
        """Compute perimeter of binary mask"""
        if mask.sum() == 0:
            return 0
        
        H, W = mask.shape
        perimeter = 0
        
        for i in range(H):
            for j in range(W):
                if mask[i, j] > 0:
                    # Check if pixel is on boundary
                    is_boundary = False
                    for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        ni, nj = i + di, j + dj
                        if ni < 0 or ni >= H or nj < 0 or nj >= W or mask[ni, nj] == 0:
                            is_boundary = True
                            break
                    if is_boundary:
                        perimeter += 1
        
        return perimeter
    
    def detect_symmetry(self, mask: torch.Tensor) -> Dict[str, bool]:
        """Detect symmetry properties of object"""
        if mask.sum() == 0:
            return {'horizontal': False, 'vertical': False, 'diagonal': False, 'rotational': False}
        
        # Extract object region
        y_min, x_min, y_max, x_max = self.compute_bbox(mask)
        obj_mask = mask[y_min:y_max+1, x_min:x_max+1]
        
        # Check horizontal symmetry
        horizontal_sym = torch.equal(obj_mask, torch.flip(obj_mask, dims=[1]))
        
        # Check vertical symmetry
        vertical_sym = torch.equal(obj_mask, torch.flip(obj_mask, dims=[0]))
        
        # Check diagonal symmetry (simplified)
        try:
            diagonal_sym = torch.equal(obj_mask, obj_mask.t()) if obj_mask.shape[0] == obj_mask.shape[1] else False
        except:
            diagonal_sym = False
        
        # Check 180-degree rotational symmetry
        try:
            rotational_sym = torch.equal(obj_mask, torch.rot90(obj_mask, k=2, dims=(0, 1)))
        except:
            rotational_sym = False
        
        return {
            'horizontal': horizontal_sym,
            'vertical': vertical_sym,
            'diagonal': diagonal_sym,
            'rotational': rotational_sym
        }
    
    def count_holes(self, mask: torch.Tensor) -> int:
        """Count number of holes in the object (simplified topology)"""
        if mask.sum() == 0:
            return 0
        
        # Simple hole counting: find connected components of background within object bbox
        y_min, x_min, y_max, x_max = self.compute_bbox(mask)
        obj_region = mask[y_min:y_max+1, x_min:x_max+1]
        
        # Background mask within object region
        bg_mask = (obj_region == 0).float()
        
        # Count connected components of background
        # Holes are background components that don't touch the boundary
        holes = 0
        visited = torch.zeros_like(bg_mask, dtype=torch.bool)
        H, W = bg_mask.shape
        
        for i in range(H):
            for j in range(W):
                if bg_mask[i, j] > 0 and not visited[i, j]:
                    # Flood fill this background component
                    component_pixels = []
                    stack = [(i, j)]
                    is_hole = True  # Assume hole until proven otherwise
                    
                    while stack:
                        ci, cj = stack.pop()
                        if ci < 0 or ci >= H or cj < 0 or cj >= W or visited[ci, cj] or bg_mask[ci, cj] == 0:
                            continue
                        
                        visited[ci, cj] = True
                        component_pixels.append((ci, cj))
                        
                        # Check if touching boundary (not a hole)
                        if ci == 0 or ci == H-1 or cj == 0 or cj == W-1:
                            is_hole = False
                        
                        # Add neighbors
                        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            stack.append((ci + di, cj + dj))
                    
                    if is_hole and len(component_pixels) > 0:
                        holes += 1
        
        return holes
    
    def objects_adjacent(self, mask1: torch.Tensor, mask2: torch.Tensor, max_dist: int = 1) -> bool:
        """Check if two objects are adjacent (within max_dist pixels)"""
        y1_coords, x1_coords = torch.nonzero(mask1, as_tuple=True)
        y2_coords, x2_coords = torch.nonzero(mask2, as_tuple=True)
        
        if len(y1_coords) == 0 or len(y2_coords) == 0:
            return False
        
        # Check minimum distance between any pixels
        coords1 = torch.stack([y1_coords, x1_coords], dim=1).float()
        coords2 = torch.stack([y2_coords, x2_coords], dim=1).float()
        
        distances = torch.cdist(coords1, coords2)
        min_distance = distances.min().item()
        
        return min_distance <= max_dist
    
    def compute_clean_cc_masks(self, grid: torch.Tensor) -> List[torch.Tensor]:
        """
        Connected components with noise filtering
        
        Args:
            grid: Input grid [H, W] or [B, H, W]
            
        Returns:
            List of clean object masks with small components filtered out
        """
        if grid.dim() == 3:
            grid = grid[0]
        
        H, W = grid.shape
        masks = []
        visited = torch.zeros_like(grid, dtype=torch.bool)
        
        # Find connected components for each non-zero color
        unique_colors = torch.unique(grid)
        unique_colors = unique_colors[unique_colors != 0]  # Exclude background
        
        for color in unique_colors:
            color_mask = (grid == color)
            
            for i in range(H):
                for j in range(W):
                    if color_mask[i, j] and not visited[i, j]:
                        # Flood-fill this component
                        component_mask = torch.zeros_like(grid, dtype=torch.float)
                        stack = [(i, j)]
                        component_pixels = []
                        
                        while stack:
                            ci, cj = stack.pop()
                            if (ci < 0 or ci >= H or cj < 0 or cj >= W or 
                                visited[ci, cj] or grid[ci, cj] != color):
                                continue
                            
                            visited[ci, cj] = True
                            component_mask[ci, cj] = 1.0
                            component_pixels.append((ci, cj))
                            
                            # Add 4-connected neighbors
                            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                                stack.append((ci + di, cj + dj))
                        
                        # Filter small components (< 3 pixels)
                        if len(component_pixels) >= 3:
                            masks.append(component_mask)
        
        # Merge adjacent same-color regions if needed
        merged_masks = self._merge_adjacent_same_color(masks, grid)
        
        return merged_masks
    
    def _merge_adjacent_same_color(self, masks: List[torch.Tensor], grid: torch.Tensor) -> List[torch.Tensor]:
        """Merge adjacent masks of the same color"""
        if not masks:
            return masks
        
        merged = []
        used = set()
        
        for i, mask1 in enumerate(masks):
            if i in used:
                continue
            
            # Find color of this mask
            color1 = self._get_mask_color(mask1, grid)
            merged_mask = mask1.clone()
            used.add(i)
            
            # Look for adjacent masks of same color
            changed = True
            while changed:
                changed = False
                for j, mask2 in enumerate(masks):
                    if j in used:
                        continue
                    
                    color2 = self._get_mask_color(mask2, grid)
                    if color1 == color2 and self.objects_adjacent(merged_mask, mask2, max_dist=1):
                        merged_mask = merged_mask + mask2
                        merged_mask = (merged_mask > 0).float()  # Ensure binary
                        used.add(j)
                        changed = True
            
            merged.append(merged_mask)
        
        return merged
    
    def _get_mask_color(self, mask: torch.Tensor, grid: torch.Tensor) -> int:
        """Get the dominant color in a masked region"""
        if mask.sum() == 0:
            return 0
        
        masked_values = grid[mask > 0]
        if len(masked_values) == 0:
            return 0
        
        # Return most common color
        unique, counts = torch.unique(masked_values, return_counts=True)
        most_common_idx = counts.argmax()
        return unique[most_common_idx].item()
    
    def _get_connected_components_from_mask(self, grid: torch.Tensor, attention_mask: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract connected components from a grid within an attention mask
        
        Args:
            grid: Input grid [H, W]
            attention_mask: Binary attention mask [H, W]
            
        Returns:
            List of component masks, each [H, W]
        """
        H, W = grid.shape
        visited = torch.zeros_like(grid, dtype=torch.bool)
        components = []
        
        # Find all connected components in the masked region
        for i in range(H):
            for j in range(W):
                if attention_mask[i, j] > 0 and not visited[i, j] and grid[i, j] != 0:
                    # Found a new component - flood fill
                    component_mask = torch.zeros_like(grid)
                    
                    stack = [(i, j)]
                    component_color = grid[i, j]
                    
                    while stack:
                        ci, cj = stack.pop()
                        if (ci < 0 or ci >= H or cj < 0 or cj >= W or 
                            visited[ci, cj] or 
                            attention_mask[ci, cj] == 0 or
                            grid[ci, cj] != component_color):
                            continue
                        
                        visited[ci, cj] = True
                        component_mask[ci, cj] = 1.0
                        
                        # Add 4-connected neighbors
                        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            stack.append((ci + di, cj + dj))
                    
                    # Only keep components with reasonable size
                    if component_mask.sum() > 0:
                        components.append(component_mask)
        
        return components
    
    def compute_object_spatial_relations(self, object_masks: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute spatial relationships between detected objects
        
        Args:
            object_masks: List of object masks, each [H, W]
            
        Returns:
            Dictionary containing spatial relation tensors
        """
        if not object_masks:
            return {}
        
        num_objects = len(object_masks)
        H, W = object_masks[0].shape
        
        # Compute centroids for each object
        centroids = []
        areas = []
        bboxes = []
        
        for mask in object_masks:
            # Centroid
            y_coords, x_coords = torch.nonzero(mask, as_tuple=True)
            if len(y_coords) > 0:
                centroid_y = y_coords.float().mean()
                centroid_x = x_coords.float().mean()
                centroids.append([centroid_y, centroid_x])
                
                # Area
                areas.append(mask.sum().item())
                
                # Bounding box
                y_min, y_max = y_coords.min().item(), y_coords.max().item()
                x_min, x_max = x_coords.min().item(), x_coords.max().item()
                bboxes.append([y_min, x_min, y_max, x_max])
            else:
                centroids.append([0.0, 0.0])
                areas.append(0.0)
                bboxes.append([0, 0, 0, 0])
        
        centroids = torch.tensor(centroids)
        areas = torch.tensor(areas)
        bboxes = torch.tensor(bboxes)
        
        # Compute pairwise relations
        relations = {}
        
        # Distance matrix
        if num_objects > 1:
            # Pairwise distances between centroids
            distances = torch.cdist(centroids, centroids)  # [N, N]
            relations['distances'] = distances
            
            # Relative positions (dx, dy)
            dx = centroids[:, 1].unsqueeze(1) - centroids[:, 1].unsqueeze(0)  # [N, N]
            dy = centroids[:, 0].unsqueeze(1) - centroids[:, 0].unsqueeze(0)  # [N, N]
            relations['relative_positions'] = torch.stack([dx, dy], dim=-1)  # [N, N, 2]
            
            # Area ratios
            area_ratios = areas.unsqueeze(1) / (areas.unsqueeze(0) + 1e-8)  # [N, N]
            relations['area_ratios'] = area_ratios
            
            # Adjacency (touching objects)
            adjacency = torch.zeros(num_objects, num_objects)
            for i in range(num_objects):
                for j in range(i + 1, num_objects):
                    if self._objects_adjacent(object_masks[i], object_masks[j]):
                        adjacency[i, j] = 1.0
                        adjacency[j, i] = 1.0
            relations['adjacency'] = adjacency
            
            # Containment relations
            containment = torch.zeros(num_objects, num_objects)
            for i in range(num_objects):
                for j in range(num_objects):
                    if i != j and self._object_contains(object_masks[i], object_masks[j]):
                        containment[i, j] = 1.0
            relations['containment'] = containment
        else:
            # Single object case
            relations['distances'] = torch.zeros(1, 1)
            relations['relative_positions'] = torch.zeros(1, 1, 2)
            relations['area_ratios'] = torch.ones(1, 1)
            relations['adjacency'] = torch.zeros(1, 1)
            relations['containment'] = torch.zeros(1, 1)
        
        relations['centroids'] = centroids
        relations['areas'] = areas
        relations['bboxes'] = bboxes
        
        return relations
    
    def _objects_adjacent(self, mask1: torch.Tensor, mask2: torch.Tensor, max_dist: int = 1) -> bool:
        """Check if two objects are adjacent (within max_dist pixels)"""
        # Get coordinates of each object
        y1, x1 = torch.nonzero(mask1, as_tuple=True)
        y2, x2 = torch.nonzero(mask2, as_tuple=True)
        
        if len(y1) == 0 or len(y2) == 0:
            return False
        
        # Check minimum distance between any pixels of the two objects
        coords1 = torch.stack([y1, x1], dim=1).float()  # [N1, 2]
        coords2 = torch.stack([y2, x2], dim=1).float()  # [N2, 2]
        
        # Compute pairwise distances
        distances = torch.cdist(coords1, coords2)  # [N1, N2]
        min_distance = distances.min().item()
        
        return min_distance <= max_dist
    
    def _object_contains(self, outer_mask: torch.Tensor, inner_mask: torch.Tensor) -> bool:
        """Check if outer object completely contains inner object"""
        # Inner object is contained if all its pixels are within outer object's bounding box
        # and there's significant overlap
        if inner_mask.sum() == 0:
            return False
        
        overlap = (outer_mask * inner_mask).sum()
        inner_area = inner_mask.sum()
        
        # Consider contained if >80% of inner object overlaps with outer
        return overlap / inner_area > 0.8


class CrossLevelAttention(nn.Module):
    """Cross-level attention for information flow between hierarchical levels"""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.scale = dim ** -0.5
        self.out_proj = nn.Linear(dim, dim)
        
    def forward(self, source_slots, target_slots):
        """
        Args:
            source_slots: Information source [B, K_source, D]
            target_slots: Information target [B, K_target, D]
        Returns:
            updated_targets: Enhanced target slots [B, K_target, D]
        """
        B, K_target, D = target_slots.shape
        B, K_source, D = source_slots.shape
        
        # Queries from targets, keys/values from sources
        q = self.to_q(target_slots)  # [B, K_target, D]
        k = self.to_k(source_slots)  # [B, K_source, D]
        v = self.to_v(source_slots)  # [B, K_source, D]
        
        # Cross-attention
        dots = torch.einsum('btd,bsd->bts', q, k) * self.scale  # [B, K_target, K_source]
        attn = F.softmax(dots, dim=-1)
        
        # Aggregate information
        updates = torch.einsum('bts,bsd->btd', attn, v)  # [B, K_target, D]
        
        return self.out_proj(updates)


class SpatialRelationEncoder(nn.Module):
    """Encode spatial relationships between objects"""
    
    def __init__(self, slot_dim):
        super().__init__()
        self.slot_dim = slot_dim
        
        # Position encoding for spatial relationships
        self.pos_encoder = nn.Sequential(
            nn.Linear(4, 32),  # (x1, y1, x2, y2) relative positions
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
    def forward(self, slots, H, W):
        """
        Args:
            slots: Object slots [B, K, D]
            H, W: Grid dimensions
        Returns:
            spatial_relations: [B, K, K, 32] pairwise spatial encodings
        """
        B, K, D = slots.shape
        
        # For now, use simple distance-based encoding
        # In practice, would use attention weights to infer positions
        positions = torch.arange(K, device=slots.device).float()
        positions = positions.view(1, K, 1).expand(B, K, 2)  # Simple 2D positions
        
        # Compute pairwise spatial relationships
        pos_diff = positions.unsqueeze(2) - positions.unsqueeze(1)  # [B, K, K, 2]
        
        # Add relative distances
        distances = torch.norm(pos_diff, dim=-1, keepdim=True)  # [B, K, K, 1]
        spatial_features = torch.cat([pos_diff, distances], dim=-1)  # [B, K, K, 3]
        
        # Pad to expected input size for pos_encoder
        padding = torch.zeros(B, K, K, 1, device=slots.device)
        spatial_features = torch.cat([spatial_features, padding], dim=-1)  # [B, K, K, 4]
        
        # Encode spatial relationships
        spatial_relations = self.pos_encoder(spatial_features)  # [B, K, K, 32]
        
        return spatial_relations


class GraphAttentionNetwork(nn.Module):
    """Graph attention network for object reasoning"""
    
    def __init__(self, node_dim, edge_dim, num_heads=4, num_layers=2):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        self.gat_layers = nn.ModuleList([
            GATLayer(node_dim, edge_dim, num_heads)
            for _ in range(num_layers)
        ])
        
    def forward(self, nodes, edges):
        """
        Args:
            nodes: Node features [B, K, node_dim]
            edges: Edge features [B, K, K, edge_dim]
        Returns:
            enhanced_nodes: [B, K, node_dim]
        """
        x = nodes
        
        for layer in self.gat_layers:
            x = layer(x, edges)
            
        return x


class GATLayer(nn.Module):
    """Single Graph Attention Layer"""
    
    def __init__(self, node_dim, edge_dim, num_heads):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_heads = num_heads
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=node_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Edge incorporation
        self.edge_proj = nn.Linear(edge_dim, node_dim)
        
        # Layer norm and feedforward
        self.norm1 = nn.LayerNorm(node_dim)
        self.norm2 = nn.LayerNorm(node_dim)
        
        self.ff = nn.Sequential(
            nn.Linear(node_dim, node_dim * 2),
            nn.ReLU(),
            nn.Linear(node_dim * 2, node_dim)
        )
        
    def forward(self, nodes, edges):
        """
        Args:
            nodes: [B, K, node_dim]
            edges: [B, K, K, edge_dim]  
        Returns:
            output: [B, K, node_dim]
        """
        B, K, node_dim = nodes.shape
        
        # Incorporate edge information
        edge_info = self.edge_proj(edges)  # [B, K, K, node_dim]
        edge_bias = edge_info.mean(dim=2)  # [B, K, node_dim]
        
        # Self-attention with residual
        attn_out, _ = self.attention(nodes, nodes, nodes)
        x = self.norm1(nodes + attn_out + edge_bias)
        
        # Feedforward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x


class SymmetryDetector(nn.Module):
    """Detect various types of symmetries in slot representations"""
    
    def __init__(self, slot_dim, symmetry_type='rotation'):
        super().__init__()
        self.slot_dim = slot_dim
        self.symmetry_type = symmetry_type
        
        if symmetry_type == 'rotation':
            # Detect 90°, 180°, 270° rotations
            self.detector = nn.Sequential(
                nn.Linear(slot_dim * 4, 128),  # Compare 4 rotated versions
                nn.ReLU(),
                nn.Linear(128, 3),  # 90°, 180°, 270°
                nn.Sigmoid()
            )
        elif symmetry_type == 'reflection':
            # Detect H, V, diagonal reflections
            self.detector = nn.Sequential(
                nn.Linear(slot_dim * 4, 128),  # Compare 4 reflected versions
                nn.ReLU(), 
                nn.Linear(128, 4),  # H, V, D1, D2
                nn.Sigmoid()
            )
        elif symmetry_type == 'translation':
            # Detect repeating patterns
            self.detector = nn.Sequential(
                nn.Linear(slot_dim * 2, 128),
                nn.ReLU(),
                nn.Linear(128, 1),  # Translation symmetry score
                nn.Sigmoid()
            )
            
    def forward(self, slots, H, W):
        """
        Args:
            slots: Slot representations [B, K, slot_dim]
            H, W: Grid dimensions
        Returns:
            symmetry_scores: Symmetry detection results
        """
        B, K, D = slots.shape
        
        if self.symmetry_type == 'rotation':
            # Simple rotation symmetry detection (placeholder)
            # In practice, would use proper rotation operations on attention maps
            slots_shifted = torch.roll(slots, shifts=1, dims=1)
            combined = torch.cat([slots, slots_shifted, slots, slots_shifted], dim=-1)
            combined_flat = combined.view(B * K, -1)
            return self.detector(combined_flat).view(B, K, -1)
            
        elif self.symmetry_type == 'reflection':
            # Simple reflection symmetry detection (placeholder)
            slots_flipped = torch.flip(slots, dims=[1])
            combined = torch.cat([slots, slots_flipped, slots, slots_flipped], dim=-1)
            combined_flat = combined.view(B * K, -1)
            return self.detector(combined_flat).view(B, K, -1)
            
        elif self.symmetry_type == 'translation':
            # Translation pattern detection
            slots_shifted = torch.roll(slots, shifts=1, dims=1)
            combined = torch.cat([slots, slots_shifted], dim=-1)
            combined_flat = combined.view(B * K, -1)
            return self.detector(combined_flat).view(B, K, -1)


class GroupOperations(nn.Module):
    """Group theory operations for symmetry reasoning"""
    
    def __init__(self, slot_dim):
        super().__init__()
        self.slot_dim = slot_dim
        
        # Group element encoders
        self.rotation_encoder = nn.Linear(4, slot_dim)  # 4 rotation matrices
        self.reflection_encoder = nn.Linear(4, slot_dim)  # 4 reflection matrices
        
    def forward(self, slots):
        """Apply group operations to slots"""
        # Placeholder for group theory operations
        return slots


class HierarchyPredictor(nn.Module):
    """Predict explicit hierarchical structure"""
    
    def __init__(self, slot_dim):
        super().__init__()
        self.slot_dim = slot_dim
        
        # Hierarchy structure prediction
        # Input is flattened hierarchy_features: [B, 4*slot_dim]
        self.structure_pred = nn.Sequential(
            nn.Linear(4 * slot_dim, 128),  # Fix: 4 levels * slot_dim
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)  # Hierarchy encoding
        )
        
    def forward(self, all_slots):
        """
        Args:
            all_slots: Dict of slot representations for each level
        Returns:
            hierarchy_structure: Explicit hierarchy representation
        """
        # Combine all slot levels
        combined = []
        for level_name in ['pixel', 'object', 'group', 'scene']:
            if level_name in all_slots:
                # Pool each level to fixed size
                pooled = all_slots[level_name].mean(dim=1)  # [B, slot_dim]
                combined.append(pooled)
        
        if combined:
            hierarchy_features = torch.stack(combined, dim=1)  # [B, 4, slot_dim]
            hierarchy_flat = hierarchy_features.view(hierarchy_features.shape[0], -1)
            structure = self.structure_pred(hierarchy_flat)
            return structure
        else:
            return None


# Backward compatibility alias
ObjectSlots = HierarchicalObjectSlots
