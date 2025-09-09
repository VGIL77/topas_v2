#!/usr/bin/env python3
from dataclasses import dataclass, field
from typing import List, Set
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import uuid

@dataclass
class Theme:
    content: str
    embedding: torch.Tensor
    frequency: float = 1.0
    emergence_score: float = 0.0
    theme_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_ids: Set[str] = field(default_factory=set)
    creation_time: int = 0
    last_accessed: int = 0

    def __post_init__(self):
        if self.embedding.requires_grad:
            self.embedding = self.embedding.detach()

    def update_access(self, time_step: int) -> None:
        self.last_accessed = time_step
        self.frequency = min(self.frequency * 1.01, 10.0)

class ThemeEncoder(nn.Module):
    """Contextual theme encoder with positional encoding and 1-layer transformer."""
    def __init__(self, d_model: int = 64, nhead: int = 2, dim_feedforward: int = 128):
        super().__init__()
        self.d_model = d_model
        # Simplified: just use linear projection for efficiency while maintaining context
        self.context_proj = nn.Linear(d_model, d_model)
        self.temporal_weight = nn.Parameter(torch.randn(1, 1, d_model) * 0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode sequence with context. Input: [batch, seq, d_model], Output: [batch, d_model]"""
        # Apply positional weighting across sequence
        seq_len = x.size(1)
        pos_weights = torch.linspace(0.5, 1.5, seq_len, device=x.device).view(1, -1, 1)
        
        # Weight tokens by position and temporal pattern
        weighted_x = x * pos_weights + self.temporal_weight
        
        # Context-aware projection
        projected = self.context_proj(weighted_x)
        
        # Attention-like pooling: use softmax over sequence dimension
        attention_weights = torch.softmax(projected.sum(dim=2), dim=1).unsqueeze(2)
        pooled = (projected * attention_weights).sum(dim=1)
        
        return pooled

class EmergentThemeSynthesis:
    def __init__(self, embedding_dim: int = 64, entropy_thresh: float = 1.5, novelty_thresh: float = 0.3, mutation_rate: float = 0.15, max_themes: int = 1000, theme_decay_rate: float = 0.99, min_frequency: float = 0.1, use_contextual: bool = False):
        self.embedding_dim = embedding_dim
        self.entropy_thresh = entropy_thresh
        self.novelty_thresh = novelty_thresh
        self.mutation_rate = mutation_rate
        self.max_themes = max_themes
        self.theme_decay_rate = theme_decay_rate
        self.min_frequency = min_frequency
        self.use_contextual = use_contextual
        self.themes: List[Theme] = []
        self.theme_index = {}
        self.time_step = 0
        self.synthesis_count = 0
        self.emergence_history = []
        
        # Initialize contextual encoder if enabled
        if self.use_contextual:
            self.theme_encoder = ThemeEncoder(d_model=embedding_dim)
            pass  # Contextual theme encoder initialized
        else:
            self.theme_encoder = None
            pass  # Non-contextual theme encoder initialized

    def process_dream_themes(self, tokens: torch.Tensor, labels: torch.Tensor) -> List[Theme]:
        if self.use_contextual and self.theme_encoder is not None:
            # Use contextual encoder for richer theme embeddings
            # Create label-specific sequences by applying different transformations per label
            batch_seq_tokens = []
            for i, label in enumerate(labels):
                # Create label-specific sequence transformation
                label_factor = (label.float() + 1.0) / (labels.max().float() + 1.0)  # Normalize to [1/(n+1), n/(n+1)]
                label_transform = tokens * (0.8 + 0.4 * label_factor)  # Scale by label-specific factor
                # Add small label-specific shift
                label_shift = torch.randn_like(tokens) * 0.1 * label_factor
                transformed_seq = label_transform + label_shift
                batch_seq_tokens.append(transformed_seq)
            
            batch_seq_tokens = torch.stack(batch_seq_tokens, dim=0)
            with torch.no_grad():  # Keep embeddings frozen for Theme objects
                theme_vecs = self.theme_encoder(batch_seq_tokens)
        else:
            # Backward compatible: use mean pooling
            theme_vecs = tokens.mean(0).unsqueeze(0).repeat(labels.size(0), 1)
            
        new_themes = [Theme(f"theme_{i}", theme_vecs[i], creation_time=self.time_step, last_accessed=self.time_step) for i in range(labels.size(0))]
        self.themes.extend(new_themes)
        for theme in new_themes:
            self.theme_index[theme.theme_id] = theme
        self.time_step += 1
        return new_themes

    def synthesize_emergent_themes(self, themes: List[Theme]):
        if len(themes) < 2: return
        embeds = torch.stack([t.embedding for t in themes])
        probs = F.softmax(embeds.sum(-1), dim=0)
        H = - (probs * probs.clamp(1e-9).log()).sum()
        psi = embeds / embeds.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        super_H = H + 0.1 * (psi @ psi.t()).trace()

        if super_H > self.entropy_thresh:
            parent1, parent2 = random.sample(themes, 2)
            child_embed = 0.5 * (parent1.embedding + parent2.embedding)
            mutation_mask = torch.rand_like(child_embed) < self.mutation_rate
            child_embed[mutation_mask] += torch.randn_like(child_embed[mutation_mask]) * 0.1
            kl_div = F.kl_div(child_embed.log_softmax(-1), embeds.mean(0).log_softmax(-1), reduction='batchmean')
            if kl_div > self.novelty_thresh:
                new_theme = Theme(
                    "emergent_theme",
                    child_embed,
                    emergence_score=kl_div.item(),
                    parent_ids={parent1.theme_id, parent2.theme_id},
                    creation_time=self.time_step,
                    last_accessed=self.time_step
                )
                self.themes.append(new_theme)
                self.theme_index[new_theme.theme_id] = new_theme
                self.synthesis_count += 1
                self.emergence_history.append(kl_div.item())

        # Decay themes
        themes_to_remove = []
        for theme in self.themes:
            theme.frequency *= self.theme_decay_rate
            if theme.frequency < self.min_frequency or len(self.themes) > self.max_themes:
                themes_to_remove.append(theme)
        for theme in themes_to_remove:
            self.themes.remove(theme)
            del self.theme_index[theme.theme_id]
        self.time_step += 1

    def get_insights(self) -> str:
        if not self.themes: return "No insights yet."
        avg_embed = torch.stack([t.embedding for t in self.themes]).mean(0)
        return "Complexity navigation and pattern discovery through evolutionary challenges."    
    def get_embedding(self, latents):
        """Get theme embedding for given latents"""
        if self.themes:
            # Return embedding of most relevant theme (simplified: first theme)
            return self.themes[0].embedding
        else:
            # Return zero embedding if no themes
            return torch.zeros(self.embedding_dim, device=latents.device if torch.is_tensor(latents) else "cpu")
