import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Any, Tuple, Optional


class RelGraph(nn.Module):
    """
    Relational graph module using multi-head attention and feed-forward layers
    to enrich slot features and increase similarity for symmetric slots.
    
    Architecture:
    - Stack of 2-4 transformer layers
    - Multi-head self-attention (4-8 heads) 
    - Feed-forward network with ReLU/GELU
    - Layer normalization and residual connections
    - Optional positional embeddings based on slot indices
    """
    
    def __init__(self, d=192, layers=6, heads=8, similarity_boost=0.3):
        super().__init__()
        self.d = d
        self.layers = min(max(layers, 2), 4)  # Clamp to 2-4 layers as specified
        self.heads = min(max(heads, 4), 8)    # Clamp to 4-8 heads as specified
        self.similarity_boost = similarity_boost
        
        # Ensure d is divisible by heads for proper attention
        assert d % self.heads == 0, f"Feature dimension {d} must be divisible by heads {self.heads}"
        self.head_dim = d // self.heads
        
        # Build transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(d=d, heads=self.heads, similarity_boost=similarity_boost) 
            for _ in range(self.layers)
        ])
        
        # Optional positional embeddings for slot indices
        # We'll use learnable position embeddings for up to 128 slots
        self.max_slots = 128
        self.pos_embedding = nn.Parameter(torch.randn(1, self.max_slots, d) * 0.02)
        
        # Fixed spatial projection (avoid creating modules during forward)
        self.spatial_proj = nn.Linear(9, d)
        
        # Final layer norm for stability
        self.final_norm = nn.LayerNorm(d)
        
    def forward(self, slots, object_masks=None, H=None, W=None):
        """
        Forward pass that enriches slot features through self-attention with spatial relations
        
        Args:
            slots: [B, K, d] slot feature vectors
            object_masks: Optional list of object masks for spatial computation
            H, W: Grid dimensions for spatial reasoning
            
        Returns:
            enriched_slots: [B, K, d] enriched slot features with spatial relations
            spatial_relations: Dictionary of spatial relationship features (if masks provided)
        """
        B, K, d = slots.shape
        assert d == self.d, f"Input feature dim {d} doesn't match model dim {self.d}"
        
        # Add positional embeddings based on slot indices
        if K <= self.max_slots:
            pos_emb = self.pos_embedding[:, :K, :]  # [1, K, d]
            x = slots + pos_emb  # [B, K, d]
        else:
            # For very large slot counts, skip positional embeddings
            x = slots
        
        # Compute spatial relations if object masks are provided
        spatial_relations = None
        if object_masks is not None and H is not None and W is not None:
            spatial_relations = self.compute_spatial_relations(slots, object_masks, H, W)
            
            # Incorporate spatial features into slot representations
            if 'spatial_features' in spatial_relations:
                spatial_features = spatial_relations['spatial_features']  # [B, K, spatial_dim]
                spatial_embedding = self.spatial_proj(spatial_features)  # [B, K, d]
                x = x + spatial_embedding * 0.1  # Small contribution to avoid overwhelming slot features
            
        # Apply transformer layers with residual connections
        for layer in self.transformer_layers:
            x = layer(x)  # Each layer handles residual internally
            
        # Final normalization for stability
        x = self.final_norm(x)
        
        if spatial_relations is not None:
            return x, spatial_relations
        return x
    
    def compute_spatial_relations(self, slots, object_masks, H, W):
        """
        Compute spatial relationships between objects based on masks and positions
        
        Args:
            slots: [B, K, d] slot features  
            object_masks: List of object masks, each [H, W]
            H, W: Grid dimensions
            
        Returns:
            Dictionary containing spatial relation features
        """
        if not object_masks:
            return {}
        
        B, K, d = slots.shape
        num_objects = min(len(object_masks), K)  # Match to available slots
        
        # Compute object properties
        centroids = []
        areas = []
        bboxes = []
        
        for i, mask in enumerate(object_masks[:num_objects]):
            # Centroid computation
            y_coords, x_coords = torch.nonzero(mask, as_tuple=True)
            if len(y_coords) > 0:
                centroid_y = y_coords.float().mean()
                centroid_x = x_coords.float().mean()
                centroids.append([centroid_y.item(), centroid_x.item()])
                
                # Area
                areas.append(mask.sum().item())
                
                # Bounding box
                y_min, y_max = y_coords.min().item(), y_coords.max().item()
                x_min, x_max = x_coords.min().item(), x_coords.max().item()
                bboxes.append([y_min, x_min, y_max, x_max])
            else:
                centroids.append([H/2, W/2])  # Default center
                areas.append(0.0)
                bboxes.append([0, 0, H-1, W-1])
        
        # Pad to match slot count
        while len(centroids) < K:
            centroids.append([H/2, W/2])
            areas.append(0.0)
            bboxes.append([0, 0, H-1, W-1])
        
        centroids = torch.tensor(centroids[:K], device=slots.device)  # [K, 2]
        areas = torch.tensor(areas[:K], device=slots.device)  # [K]
        bboxes = torch.tensor(bboxes[:K], device=slots.device)  # [K, 4]
        
        relations = {}
        
        if K > 1:
            # Pairwise distances
            distances = torch.cdist(centroids.unsqueeze(0), centroids.unsqueeze(0))[0]  # [K, K]
            relations['distances'] = distances
            
            # Relative positions
            dx = centroids[:, 1].unsqueeze(1) - centroids[:, 1].unsqueeze(0)  # [K, K]
            dy = centroids[:, 0].unsqueeze(1) - centroids[:, 0].unsqueeze(0)  # [K, K]
            relations['dx'] = dx
            relations['dy'] = dy
            
            # Area ratios
            area_ratios = areas.unsqueeze(1) / (areas.unsqueeze(0) + 1e-8)  # [K, K]
            relations['area_ratios'] = area_ratios
            
            # Adjacency matrix (objects within distance threshold)
            adjacency = (distances < 2.0).float()  # Objects within 2 pixels
            adjacency.fill_diagonal_(0)  # Remove self-connections
            relations['adjacency'] = adjacency
            
            # Containment (based on bounding box inclusion)
            containment = torch.zeros(K, K, device=slots.device)
            for i in range(K):
                for j in range(K):
                    if i != j:
                        # Check if object j is contained in object i's bounding box
                        bbox_i = bboxes[i]  # [y_min, x_min, y_max, x_max]
                        bbox_j = bboxes[j]
                        
                        if (bbox_j[0] >= bbox_i[0] and bbox_j[1] >= bbox_i[1] and
                            bbox_j[2] <= bbox_i[2] and bbox_j[3] <= bbox_i[3] and
                            areas[i] > areas[j]):
                            containment[i, j] = 1.0
            relations['containment'] = containment
            
            # Create spatial feature vectors for each slot
            spatial_features = []
            for k in range(K):
                # Features for slot k: [centroid_x, centroid_y, area, bbox_width, bbox_height,
                #                      avg_distance_to_others, num_adjacent, is_container, is_contained]
                centroid_x, centroid_y = centroids[k]
                area = areas[k]
                bbox = bboxes[k]
                bbox_width = bbox[3] - bbox[1] + 1  # x_max - x_min + 1
                bbox_height = bbox[2] - bbox[0] + 1  # y_max - y_min + 1
                
                # Average distance to other objects
                other_distances = distances[k]
                other_distances[k] = 0  # Exclude self
                avg_distance = other_distances.sum() / (K - 1) if K > 1 else 0
                
                # Number of adjacent objects
                num_adjacent = adjacency[k].sum()
                
                # Container/contained status
                is_container = containment[k].sum()  # How many objects this contains
                is_contained = containment[:, k].sum()  # How many objects contain this
                
                features = torch.tensor([
                    centroid_x / W,  # Normalized coordinates
                    centroid_y / H,
                    area / (H * W),  # Normalized area
                    bbox_width / W,  # Normalized dimensions
                    bbox_height / H,
                    avg_distance / max(H, W),  # Normalized distance
                    num_adjacent / K,  # Normalized adjacency count
                    is_container / K,  # Normalized container status
                    is_contained / K,  # Normalized contained status
                ], device=slots.device)
                
                spatial_features.append(features)
            
            spatial_features = torch.stack(spatial_features).unsqueeze(0).expand(B, -1, -1)  # [B, K, 9]
            relations['spatial_features'] = spatial_features
        else:
            # Single object case
            relations['distances'] = torch.zeros(1, 1, device=slots.device)
            relations['dx'] = torch.zeros(1, 1, device=slots.device)
            relations['dy'] = torch.zeros(1, 1, device=slots.device)
            relations['area_ratios'] = torch.ones(1, 1, device=slots.device)
            relations['adjacency'] = torch.zeros(1, 1, device=slots.device)
            relations['containment'] = torch.zeros(1, 1, device=slots.device)
            
            # Single object spatial features
            centroid_x, centroid_y = centroids[0]
            area = areas[0]
            bbox = bboxes[0]
            bbox_width = bbox[3] - bbox[1] + 1
            bbox_height = bbox[2] - bbox[0] + 1
            
            features = torch.tensor([
                centroid_x / W, centroid_y / H, area / (H * W),
                bbox_width / W, bbox_height / H,
                0.0, 0.0, 0.0, 0.0  # No relations for single object
            ], device=slots.device)
            
            spatial_features = features.unsqueeze(0).unsqueeze(0).expand(B, K, -1)  # [B, K, 9]
            relations['spatial_features'] = spatial_features
        
        relations['centroids'] = centroids.unsqueeze(0).expand(B, -1, -1)  # [B, K, 2]
        relations['areas'] = areas.unsqueeze(0).expand(B, -1)  # [B, K]
        relations['bboxes'] = bboxes.unsqueeze(0).expand(B, -1, -1)  # [B, K, 4]
        
        return relations


class TransformerLayer(nn.Module):
    """
    Single transformer layer with multi-head attention and feed-forward network
    """
    
    def __init__(self, d, heads, ff_ratio=4, dropout=0.1, similarity_boost=0.2):
        super().__init__()
        self.d = d
        self.heads = heads
        
        # Multi-head self-attention with similarity boost
        self.self_attn = MultiHeadAttention(d, heads, dropout, similarity_boost)
        self.norm1 = nn.LayerNorm(d)
        
        # Feed-forward network
        self.ff = FeedForward(d, ff_ratio, dropout)
        self.norm2 = nn.LayerNorm(d)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass with pre-norm transformer architecture
        
        Args:
            x: [B, K, d] input features
            
        Returns:
            output: [B, K, d] enriched features
        """
        # Multi-head self-attention with pre-norm and residual
        norm_x = self.norm1(x)
        attn_out = self.self_attn(norm_x, norm_x, norm_x)  # Self-attention
        x = x + self.dropout(attn_out)
        
        # Feed-forward with pre-norm and residual
        norm_x = self.norm2(x)
        ff_out = self.ff(norm_x)
        x = x + self.dropout(ff_out)
        
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism designed to increase similarity
    between related/symmetric slots
    """
    
    def __init__(self, d, heads, dropout=0.1, similarity_boost=0.2):
        super().__init__()
        self.d = d
        self.heads = heads
        self.head_dim = d // heads
        self.similarity_boost = similarity_boost
        
        assert d % heads == 0, f"d={d} must be divisible by heads={heads}"
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.out_proj = nn.Linear(d, d)
        
        # Additional projection for similarity computation
        self.sim_proj = nn.Linear(d, d // 4, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, query, key, value, mask=None):
        """
        Multi-head attention forward pass with similarity enhancement
        
        Args:
            query: [B, K, d] query vectors
            key: [B, K, d] key vectors  
            value: [B, K, d] value vectors
            mask: Optional attention mask
            
        Returns:
            output: [B, K, d] attention output
        """
        B, K, d = query.shape
        
        # Linear projections and reshape for multi-head
        Q = self.q_proj(query).view(B, K, self.heads, self.head_dim).transpose(1, 2)  # [B, heads, K, head_dim]
        K_proj = self.k_proj(key).view(B, K, self.heads, self.head_dim).transpose(1, 2)    # [B, heads, K, head_dim]
        V = self.v_proj(value).view(B, K, self.heads, self.head_dim).transpose(1, 2)  # [B, heads, K, head_dim]
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K_proj.transpose(-2, -1)) * self.scale  # [B, heads, K, K]
        
        # Compute similarity bonus for related slots
        if self.similarity_boost > 0.0:
            # Project to similarity space for more robust similarity computation
            sim_features = self.sim_proj(query)  # [B, K, d//4]
            sim_features_norm = F.normalize(sim_features, dim=-1)  # [B, K, d//4]
            
            # Compute cosine similarity matrix
            similarity_matrix = torch.bmm(sim_features_norm, sim_features_norm.transpose(-2, -1))  # [B, K, K]
            
            # Create similarity bonus (higher for more similar slots)
            similarity_bonus = self.similarity_boost * similarity_matrix.unsqueeze(1)  # [B, 1, K, K]
            similarity_bonus = similarity_bonus.expand(-1, self.heads, -1, -1)  # [B, heads, K, K]
            
            # Add similarity bonus to attention scores
            attn_scores = attn_scores + similarity_bonus
        
        # Apply mask if provided
        if mask is not None:
            attn_scores.masked_fill_(mask == 0, float('-inf'))
            
        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, heads, K, K]
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # [B, heads, K, head_dim]
        
        # Concatenate heads and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, K, d)  # [B, K, d]
        output = self.out_proj(attn_output)
        
        return output


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network
    """
    
    def __init__(self, d, ff_ratio=4, dropout=0.1, activation='gelu'):
        super().__init__()
        self.d = d
        self.ff_dim = d * ff_ratio
        
        self.linear1 = nn.Linear(d, self.ff_dim)
        self.linear2 = nn.Linear(self.ff_dim, d)
        self.dropout = nn.Dropout(dropout)
        
        # Choose activation function
        if activation.lower() == 'gelu':
            self.activation = F.gelu
        elif activation.lower() == 'relu':
            self.activation = F.relu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
            
    def forward(self, x):
        """
        Feed-forward network forward pass
        
        Args:
            x: [B, K, d] input features
            
        Returns:
            output: [B, K, d] processed features
        """
        # Two linear layers with activation and dropout
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        
        return x


# ============================================================================
# OBJECT RELATION PREDICTOR - ACTIVE RELATION CLASSIFICATION
# ============================================================================

class ObjectRelationPredictor(nn.Module):
    """Predict relationships between objects using their features"""
    
    def __init__(self, feature_dim: int = 64, hidden_dim: int = 128, num_relation_types: int = 8):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_relation_types = num_relation_types
        
        # Feature encoder for object properties
        self.feature_encoder = nn.Sequential(
            nn.Linear(16, feature_dim),  # Input: area, bbox(4), centroid(2), shape_sig(4), symmetries(4), holes(1)
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU()
        )
        
        # Pairwise relationship predictor
        self.relation_predictor = nn.Sequential(
            nn.Linear(feature_dim * 2 + 8, hidden_dim),  # 2 object features + spatial features
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_relation_types),
            nn.Sigmoid()  # Probabilities for each relation type
        )
        
        # Spatial relationship encoder
        self.spatial_encoder = nn.Sequential(
            nn.Linear(8, 32),  # dx, dy, distance, angle, size_ratio, area_ratio, overlap, boundary_dist
            nn.ReLU(),
            nn.Linear(32, 8)
        )
        
        # Self-attention for global object context
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=4,
            batch_first=True
        )
        
    def encode_object_features(self, objects: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Encode object features into fixed-size vectors
        
        Args:
            objects: List of object feature dictionaries
            
        Returns:
            Object feature tensor [N, feature_dim]
        """
        if not objects:
            return torch.empty(0, self.feature_dim)
        
        features = []
        for obj in objects:
            # Extract numerical features
            area = obj['area']
            bbox = obj['bbox']  # (y_min, x_min, y_max, x_max)
            centroid = obj['centroid']  # (y, x)
            shape_sig = obj['shape_signature']  # {'aspect_ratio', 'compactness', 'extent', ...}
            symmetry = obj['symmetry']  # {'horizontal', 'vertical', 'rotational', ...}
            holes = obj['holes']
            
            # Create feature vector
            feat_vec = [
                float(area),
                float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]),  # bbox
                float(centroid[0]), float(centroid[1]),  # centroid
                float(shape_sig.get('aspect_ratio', 1.0)),
                float(shape_sig.get('compactness', 0.0)),
                float(shape_sig.get('extent', 0.0)),
                float(shape_sig.get('convexity', 0.0)),  # If available
                float(symmetry.get('horizontal', False)),
                float(symmetry.get('vertical', False)),
                float(symmetry.get('rotational', False)),
                float(symmetry.get('diagonal', False)),  # If available
                float(holes)
            ]
            
            features.append(feat_vec)
        
        # Convert to tensor and encode
        feature_tensor = torch.tensor(features, dtype=torch.float32)
        encoded = self.feature_encoder(feature_tensor)
        
        # Apply self-attention for global context
        if len(encoded) > 1:
            attended, _ = self.attention(encoded.unsqueeze(0), encoded.unsqueeze(0), encoded.unsqueeze(0))
            encoded = attended.squeeze(0)
        
        return encoded
    
    def compute_spatial_features(self, obj1: Dict[str, Any], obj2: Dict[str, Any]) -> torch.Tensor:
        """
        Compute spatial relationship features between two objects
        
        Args:
            obj1, obj2: Object feature dictionaries
            
        Returns:
            Spatial feature vector [8]
        """
        # Extract centroids and bboxes
        c1 = obj1['centroid']
        c2 = obj2['centroid']
        bbox1 = obj1['bbox']
        bbox2 = obj2['bbox']
        area1 = obj1['area']
        area2 = obj2['area']
        
        # Compute spatial relationships
        dx = c2[1] - c1[1]  # x difference
        dy = c2[0] - c1[0]  # y difference
        distance = math.sqrt(dx*dx + dy*dy)
        angle = math.atan2(dy, dx) if distance > 0 else 0.0
        
        # Size relationships
        size_ratio = area2 / max(area1, 1.0)
        area_ratio = min(area1, area2) / max(area1, area2)
        
        # Overlap and boundary distance (simplified)
        overlap = self._compute_overlap(bbox1, bbox2)
        boundary_dist = self._compute_boundary_distance(bbox1, bbox2)
        
        spatial_features = torch.tensor([
            dx, dy, distance, angle, size_ratio, area_ratio, overlap, boundary_dist
        ], dtype=torch.float32)
        
        return self.spatial_encoder(spatial_features)
    
    def _compute_overlap(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Compute overlap ratio between two bounding boxes"""
        y1_min, x1_min, y1_max, x1_max = bbox1
        y2_min, x2_min, y2_max, x2_max = bbox2
        
        # Intersection
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        intersection = y_overlap * x_overlap
        
        # Union
        area1 = (y1_max - y1_min) * (x1_max - x1_min)
        area2 = (y2_max - y2_min) * (x2_max - x2_min)
        union = area1 + area2 - intersection
        
        return intersection / max(union, 1.0)
    
    def _compute_boundary_distance(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Compute minimum distance between object boundaries"""
        y1_min, x1_min, y1_max, x1_max = bbox1
        y2_min, x2_min, y2_max, x2_max = bbox2
        
        # Check if bboxes overlap
        if (y1_min <= y2_max and y1_max >= y2_min and 
            x1_min <= x2_max and x1_max >= x2_min):
            return 0.0  # Overlapping
        
        # Compute minimum distance
        y_dist = max(0, max(y1_min - y2_max, y2_min - y1_max))
        x_dist = max(0, max(x1_min - x2_max, x2_min - x1_max))
        
        return math.sqrt(y_dist*y_dist + x_dist*x_dist)
    
    def forward(self, obj1_features: torch.Tensor, obj2_features: torch.Tensor, 
                spatial_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict relationships between two objects
        
        Args:
            obj1_features: First object features [feature_dim]
            obj2_features: Second object features [feature_dim]
            spatial_features: Spatial relationship features [8]
            
        Returns:
            Dictionary of relationship probabilities
        """
        # Concatenate features
        combined_features = torch.cat([obj1_features, obj2_features, spatial_features], dim=-1)
        
        # Predict relationships
        relation_probs = self.relation_predictor(combined_features)
        
        # Map to named relationships
        relations = {
            'touching': relation_probs[0],
            'contained': relation_probs[1],
            'containing': relation_probs[2],
            'aligned_horizontal': relation_probs[3],
            'aligned_vertical': relation_probs[4],
            'same_shape': relation_probs[5],
            'same_color': relation_probs[6],
            'symmetric': relation_probs[7]
        }
        
        return relations
    
    def predict_all_relations(self, objects: List[Dict[str, Any]]) -> Dict[Tuple[int, int], Dict[str, torch.Tensor]]:
        """
        Predict relationships between all pairs of objects
        
        Args:
            objects: List of object feature dictionaries
            
        Returns:
            Dictionary mapping (i, j) pairs to relationship predictions
        """
        if len(objects) < 2:
            return {}
        
        # Encode all object features
        encoded_features = self.encode_object_features(objects)
        
        relations = {}
        
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                # Compute spatial features
                spatial_feats = self.compute_spatial_features(objects[i], objects[j])
                
                # Predict relationships
                rel_probs = self.forward(
                    encoded_features[i], 
                    encoded_features[j], 
                    spatial_feats
                )
                
                relations[(i, j)] = rel_probs
                
                # Add reverse relationships with some adjustments
                reverse_rels = rel_probs.copy()
                # Swap containment relationships
                reverse_rels['contained'] = rel_probs['containing']
                reverse_rels['containing'] = rel_probs['contained']
                relations[(j, i)] = reverse_rels
        
        return relations
    
    def build_relation_graph(self, objects: List[Dict[str, Any]], threshold: float = 0.5) -> Dict[str, List[Tuple[int, int]]]:
        """
        Build relationship graph with edges above threshold
        
        Args:
            objects: List of object feature dictionaries
            threshold: Minimum probability to include relationship
            
        Returns:
            Dictionary mapping relation types to lists of (i, j) pairs
        """
        all_relations = self.predict_all_relations(objects)
        
        graph = {
            'touching': [],
            'contained': [],
            'containing': [],
            'aligned_horizontal': [],
            'aligned_vertical': [],
            'same_shape': [],
            'same_color': [],
            'symmetric': []
        }
        
        for (i, j), relations in all_relations.items():
            for rel_type, prob in relations.items():
                if prob > threshold:
                    graph[rel_type].append((i, j))
        
        return graph


class ObjectRelationLoss(nn.Module):
    """Loss functions for object relationship prediction training"""
    
    def __init__(self, relation_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        
        self.relation_weights = relation_weights or {
            'touching': 1.0,
            'contained': 1.0,
            'containing': 1.0,
            'aligned_horizontal': 0.8,
            'aligned_vertical': 0.8,
            'same_shape': 0.6,
            'same_color': 1.2,
            'symmetric': 0.7
        }
    
    def relation_prediction_loss(self, pred_relations: Dict[str, torch.Tensor], 
                                true_relations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute weighted BCE loss for relationship predictions"""
        total_loss = 0.0
        
        for rel_type in pred_relations:
            if rel_type in true_relations:
                weight = self.relation_weights.get(rel_type, 1.0)
                loss = F.binary_cross_entropy(pred_relations[rel_type], true_relations[rel_type])
                total_loss += weight * loss
        
        return total_loss
    
    def consistency_loss(self, relations: Dict[Tuple[int, int], Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Enforce consistency constraints between relationships"""
        consistency_loss = 0.0
        
        for (i, j), rels_ij in relations.items():
            # Check if reverse relationship exists
            if (j, i) in relations:
                rels_ji = relations[(j, i)]
                
                # Containment consistency: if i contains j, then j is contained in i
                if 'containing' in rels_ij and 'contained' in rels_ji:
                    consistency_loss += F.mse_loss(rels_ij['containing'], rels_ji['contained'])
                
                # Symmetry consistency: symmetric relation should be mutual
                if 'symmetric' in rels_ij and 'symmetric' in rels_ji:
                    consistency_loss += F.mse_loss(rels_ij['symmetric'], rels_ji['symmetric'])
        
        return consistency_loss


def create_object_relation_predictor(feature_dim: int = 64, hidden_dim: int = 128) -> ObjectRelationPredictor:
    """Factory function to create object relation predictor"""
    return ObjectRelationPredictor(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_relation_types=8
    )