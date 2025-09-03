#!/usr/bin/env python3
"""
Template Retrieval System
k-NN retrieval over template library for retrieval-augmented reasoning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple
import logging
from dataclasses import dataclass
import pickle
import os

logger = logging.getLogger(__name__)


@dataclass
class Template:
    """Represents a program template with metadata"""
    program: List[str]  # DSL operations sequence
    embeddings: torch.Tensor  # Encoded template features
    success_rate: float  # Historical success rate
    theme: str  # Visual theme (symmetry, counting, etc.)
    difficulty: float  # Complexity score
    usage_count: int  # How often this template was used
    last_used: int  # Last epoch when used
    
    def __post_init__(self):
        if self.usage_count == 0:
            self.usage_count = 1  # Avoid division by zero


class TaskEncoder(nn.Module):
    """Encodes ARC tasks for template retrieval"""
    
    def __init__(self, grid_dim: int = 1024, embedding_dim: int = 256):
        super().__init__()
        self.grid_dim = grid_dim
        self.embedding_dim = embedding_dim
        
        # Grid encoder
        self.grid_encoder = nn.Sequential(
            nn.Linear(grid_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Feature encoders
        self.size_encoder = nn.Linear(4, 32)  # H_in, W_in, H_out, W_out
        self.histogram_encoder = nn.Linear(20, 32)  # Input + output color histograms
        self.symmetry_encoder = nn.Linear(8, 32)   # Symmetry features
        self.theme_encoder = nn.Linear(16, 32)     # Theme classification features
        
        # Final projection
        self.projection = nn.Linear(256 + 32*4, embedding_dim)
        
    def forward(self, task_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode task features into embedding space.
        
        Args:
            task_features: Dictionary containing:
                - grid: Flattened grid features [batch_size, grid_dim]
                - size: Size features [batch_size, 4]
                - histogram: Color histograms [batch_size, 20]
                - symmetry: Symmetry features [batch_size, 8] 
                - theme: Theme features [batch_size, 16]
        
        Returns:
            Task embeddings [batch_size, embedding_dim]
        """
        # Encode each feature type
        grid_emb = self.grid_encoder(task_features['grid'])
        size_emb = self.size_encoder(task_features['size'])
        hist_emb = self.histogram_encoder(task_features['histogram'])
        sym_emb = self.symmetry_encoder(task_features['symmetry'])
        theme_emb = self.theme_encoder(task_features['theme'])
        
        # Concatenate and project
        combined = torch.cat([grid_emb, size_emb, hist_emb, sym_emb, theme_emb], dim=-1)
        embedding = self.projection(combined)
        
        return F.normalize(embedding, p=2, dim=-1)  # L2 normalize


class TemplateLibrary:
    """Maintains library of program templates with fast k-NN retrieval"""
    
    def __init__(self, 
                 max_templates: int = 10000,
                 embedding_dim: int = 256,
                 cache_dir: Optional[str] = None):
        self.max_templates = max_templates
        self.embedding_dim = embedding_dim
        self.cache_dir = cache_dir
        
        self.templates: List[Template] = []
        self.embeddings_matrix: Optional[torch.Tensor] = None
        self.is_dirty = False  # Track if embeddings need rebuilding
        
        # Load from cache if available
        if cache_dir and os.path.exists(os.path.join(cache_dir, 'template_library.pkl')):
            self.load_from_cache()
    
    def add_template(self, 
                    program: List[str],
                    embedding: torch.Tensor,
                    success_rate: float = 0.5,
                    theme: str = "unknown",
                    difficulty: float = 1.0) -> None:
        """Add a new template to the library"""
        template = Template(
            program=program,
            embeddings=embedding.detach().clone(),
            success_rate=success_rate,
            theme=theme,
            difficulty=difficulty,
            usage_count=0,
            last_used=0
        )
        
        self.templates.append(template)
        self.is_dirty = True
        
        # Remove oldest templates if exceeding max size
        if len(self.templates) > self.max_templates:
            # Sort by usage and success rate
            self.templates.sort(key=lambda t: t.usage_count * t.success_rate)
            self.templates = self.templates[-self.max_templates:]
    
    def build_embeddings_matrix(self):
        """Rebuild the embeddings matrix for fast retrieval"""
        if not self.templates:
            self.embeddings_matrix = None
            return
        
        embeddings = torch.stack([t.embeddings for t in self.templates])
        self.embeddings_matrix = embeddings
        self.is_dirty = False
    
    def retrieve_templates(self, 
                         query_embedding: torch.Tensor,
                         k: int = 5,
                         theme_filter: Optional[str] = None,
                         min_success_rate: float = 0.0) -> List[Tuple[Template, float]]:
        """
        Retrieve k most similar templates.
        
        Args:
            query_embedding: Query embedding [embedding_dim]
            k: Number of templates to retrieve
            theme_filter: Optional theme filter
            min_success_rate: Minimum success rate threshold
            
        Returns:
            List of (template, similarity_score) pairs
        """
        if not self.templates:
            return []
        
        # Rebuild embeddings matrix if needed
        if self.is_dirty or self.embeddings_matrix is None:
            self.build_embeddings_matrix()
        
        if self.embeddings_matrix is None:
            return []
        
        # Filter templates
        valid_indices = []
        for i, template in enumerate(self.templates):
            if template.success_rate >= min_success_rate:
                if theme_filter is None or template.theme == theme_filter:
                    valid_indices.append(i)
        
        if not valid_indices:
            return []
        
        # Compute similarities
        valid_embeddings = self.embeddings_matrix[valid_indices]
        query_embedding = F.normalize(query_embedding.unsqueeze(0), p=2, dim=-1)
        similarities = torch.mm(query_embedding, valid_embeddings.t()).squeeze(0)
        
        # Get top-k
        top_k = min(k, len(valid_indices))
        top_similarities, top_indices = similarities.topk(top_k)
        
        results = []
        for sim, idx in zip(top_similarities, top_indices):
            template_idx = valid_indices[idx]
            template = self.templates[template_idx]
            results.append((template, float(sim)))
        
        return results
    
    def update_template_stats(self, 
                            template: Template,
                            success: bool,
                            epoch: int):
        """Update template usage statistics"""
        template.usage_count += 1
        template.last_used = epoch
        
        # Update success rate with exponential moving average
        alpha = 0.1
        new_success = 1.0 if success else 0.0
        template.success_rate = (1 - alpha) * template.success_rate + alpha * new_success
    
    def save_to_cache(self):
        """Save template library to disk"""
        if not self.cache_dir:
            return
        
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_path = os.path.join(self.cache_dir, 'template_library.pkl')
        
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'templates': self.templates,
                'max_templates': self.max_templates,
                'embedding_dim': self.embedding_dim
            }, f)
    
    def load_from_cache(self):
        """Load template library from disk"""
        if not self.cache_dir:
            return
        
        cache_path = os.path.join(self.cache_dir, 'template_library.pkl')
        if not os.path.exists(cache_path):
            return
        
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            
            self.templates = data['templates']
            self.max_templates = data['max_templates']
            self.embedding_dim = data['embedding_dim']
            self.is_dirty = True  # Need to rebuild embeddings matrix
            
            logger.info(f"Loaded {len(self.templates)} templates from cache")
            
        except Exception as e:
            logger.warning(f"Failed to load template cache: {e}")
    
    def get_statistics(self) -> Dict[str, float]:
        """Get library statistics"""
        if not self.templates:
            return {}
        
        success_rates = [t.success_rate for t in self.templates]
        usage_counts = [t.usage_count for t in self.templates]
        difficulties = [t.difficulty for t in self.templates]
        
        return {
            'num_templates': len(self.templates),
            'avg_success_rate': np.mean(success_rates),
            'avg_usage_count': np.mean(usage_counts),
            'avg_difficulty': np.mean(difficulties),
            'max_success_rate': max(success_rates),
            'min_success_rate': min(success_rates)
        }


class TemplateRetrieval:
    """Main template retrieval system"""
    
    def __init__(self, 
                 embedding_dim: int = 256,
                 cache_dir: Optional[str] = None):
        self.embedding_dim = embedding_dim
        
        # Initialize components
        self.task_encoder = TaskEncoder(embedding_dim=embedding_dim)
        self.template_library = TemplateLibrary(
            embedding_dim=embedding_dim, 
            cache_dir=cache_dir
        )
        
        # Retrieval statistics
        self.retrieval_stats = {
            'total_retrievals': 0,
            'successful_retrievals': 0,
            'template_hits': 0
        }
    
    def encode_task(self, task_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode task for retrieval"""
        return self.task_encoder(task_features)
    
    def retrieve_templates(self, 
                         task_features: Dict[str, torch.Tensor],
                         k: int = 5,
                         theme_filter: Optional[str] = None) -> List[Tuple[Template, float]]:
        """
        Retrieve k nearest templates for given task.
        
        Args:
            task_features: Task feature dictionary
            k: Number of templates to retrieve
            theme_filter: Optional theme filter
            
        Returns:
            List of (template, similarity) pairs
        """
        self.retrieval_stats['total_retrievals'] += 1
        
        # Encode task
        query_embedding = self.encode_task(task_features)
        if len(query_embedding.shape) > 1:
            query_embedding = query_embedding[0]  # Take first in batch
        
        # Retrieve templates
        templates = self.template_library.retrieve_templates(
            query_embedding, k=k, theme_filter=theme_filter
        )
        
        if templates:
            self.retrieval_stats['successful_retrievals'] += 1
        
        return templates
    
    def inject_as_priors(self, 
                        templates: List[Tuple[Template, float]],
                        policy_net: nn.Module,
                        boost_factor: float = 2.0) -> None:
        """
        Initialize policy network with template priors.
        
        Boosts the probability of operations from retrieved templates.
        
        Args:
            templates: Retrieved templates with similarities
            policy_net: Policy network to modify
            boost_factor: How much to boost template operations
        """
        if not templates:
            return
        
        # Extract operations from templates
        all_ops = []
        weights = []
        
        for template, similarity in templates:
            for op in template.program:
                all_ops.append(op)
                # Weight by similarity and success rate
                weight = similarity * template.success_rate * boost_factor
                weights.append(weight)
        
        # Apply boosts to policy network
        if hasattr(policy_net, 'boost_ops'):
            policy_net.boost_ops(all_ops, weights)
        elif hasattr(policy_net, 'apply_template_priors'):
            policy_net.apply_template_priors(all_ops, weights)
        else:
            logger.warning("Policy network doesn't support template boosting")
    
    def add_successful_program(self, 
                              program: List[str],
                              task_features: Dict[str, torch.Tensor],
                              theme: str = "unknown",
                              difficulty: float = 1.0) -> None:
        """Add a successful program as a new template"""
        # Encode the task
        embedding = self.encode_task(task_features)
        if len(embedding.shape) > 1:
            embedding = embedding[0]  # Take first in batch
        
        # Add to library
        self.template_library.add_template(
            program=program,
            embedding=embedding,
            success_rate=1.0,  # New template starts with high success rate
            theme=theme,
            difficulty=difficulty
        )
        
        logger.debug(f"Added template: {program[:3]}... (theme: {theme})")
    
    def update_template_feedback(self, 
                               template: Template,
                               success: bool,
                               epoch: int):
        """Update template with feedback from usage"""
        self.template_library.update_template_stats(template, success, epoch)
        
        if success:
            self.retrieval_stats['template_hits'] += 1
    
    def get_hit_rate(self) -> float:
        """Get template hit rate"""
        total = self.retrieval_stats['total_retrievals']
        hits = self.retrieval_stats['template_hits']
        return hits / total if total > 0 else 0.0
    
    def save_library(self):
        """Save template library to disk"""
        self.template_library.save_to_cache()
    
    def get_statistics(self) -> Dict[str, float]:
        """Get comprehensive statistics"""
        stats = self.template_library.get_statistics()
        stats.update({
            'hit_rate': self.get_hit_rate(),
            'retrieval_success_rate': (
                self.retrieval_stats['successful_retrievals'] / 
                max(self.retrieval_stats['total_retrievals'], 1)
            )
        })
        stats.update(self.retrieval_stats)
        return stats


class ThemeClassifier(nn.Module):
    """Classifies task themes for better template retrieval"""
    
    THEMES = [
        'symmetry', 'counting', 'pattern_completion', 'object_detection',
        'color_mapping', 'spatial_transformation', 'logical_operation',
        'shape_manipulation', 'size_scaling', 'connectivity', 'unknown'
    ]
    
    def __init__(self, input_dim: int = 1024):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, len(self.THEMES))
        )
    
    def forward(self, task_features: torch.Tensor) -> torch.Tensor:
        """Classify task theme"""
        return self.classifier(task_features)
    
    def predict_theme(self, task_features: torch.Tensor) -> str:
        """Predict most likely theme"""
        logits = self.forward(task_features)
        predicted_idx = logits.argmax(dim=-1)
        return self.THEMES[predicted_idx]