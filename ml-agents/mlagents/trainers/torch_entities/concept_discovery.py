"""
FIXED Concept Discovery Module for Transformer CBM

Key improvements:
1. Correct sparsity loss (L1, not entropy)
2. Better diversity loss (orthogonality)
3. Gradual information bottleneck
4. Proper loss balancing
5. Temperature annealing
6. Integration with optimizer

Author: Fixed version for convergence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class ConceptDiscoveryModule(nn.Module):
    """
    Discovers interpretable concepts with improved training stability.
    
    Key changes:
    - L1 sparsity instead of entropy (encourages 0/1)
    - Orthogonality loss for diversity
    - Gradual bottleneck (higher num_concepts)
    - Temperature annealing schedule
    """
    
    def __init__(
        self,
        d_model: int = 128,
        num_concepts: int = 32,  # ✅ Increased from 12 to 32 (less extreme bottleneck)
        hidden_dim: int = 128,    # ✅ Increased from 64 to 128
        sparsity_weight: float = 0.01,      # ✅ Reduced from 0.1
        diversity_weight: float = 0.05,      # ✅ Reduced from 0.1
        reconstruction_weight: float = 1.0,
        initial_temperature: float = 2.0,    # ✅ Start high (soft concepts)
        final_temperature: float = 0.5,      # ✅ End low (binary concepts)
        temperature_decay_steps: int = 50000,  # ✅ Gradual annealing
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_concepts = num_concepts
        self.sparsity_weight = sparsity_weight
        self.diversity_weight = diversity_weight
        self.reconstruction_weight = reconstruction_weight
        
        # ✅ Temperature annealing schedule
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.temperature_decay_steps = temperature_decay_steps
        self.current_step = 0
        
        # ========== SIMPLIFIED CONCEPT ENCODER ==========
        # Less aggressive compression
        self.concept_encoder = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),  # ✅ GELU instead of ReLU (smoother gradients)
            nn.Linear(hidden_dim, num_concepts),
        )
        
        # ========== SIMPLIFIED CONCEPT DECODER ==========
        self.concept_decoder = nn.Sequential(
            nn.Linear(num_concepts, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        
        # ========== CONCEPT-TO-FEATURES ==========
        self.concept_to_features = nn.Sequential(
            nn.Linear(num_concepts, d_model),
            nn.LayerNorm(d_model),
        )
        
        self._initialize_weights()
        
        print(f"✅ FIXED Concept Discovery Module initialized")
        print(f"   Number of concepts: {num_concepts} (less aggressive bottleneck)")
        print(f"   Compression ratio: {d_model}/{num_concepts} = {d_model/num_concepts:.1f}x")
        print(f"   Temperature annealing: {initial_temperature:.2f} → {final_temperature:.2f}")
        print(f"   Sparsity weight: {sparsity_weight} (L1-based)")
        print(f"   Diversity weight: {diversity_weight} (orthogonality-based)")
    
    def _initialize_weights(self):
        """Initialize with smaller weights for stability."""
        for module in [self.concept_encoder, self.concept_decoder, self.concept_to_features]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    # ✅ Smaller initialization for stability
                    nn.init.orthogonal_(layer.weight, gain=0.5)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def get_temperature(self) -> float:
        """Get current temperature (anneals over time)."""
        progress = min(1.0, self.current_step / self.temperature_decay_steps)
        # Linear annealing from initial to final
        temperature = self.initial_temperature - progress * (
            self.initial_temperature - self.final_temperature
        )
        return temperature
    
    def forward(
        self,
        embedding: torch.Tensor,
        return_losses: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Discover concepts from embedding.
        
        Args:
            embedding: [B, d_model] - transformer output
            return_losses: Whether to compute and return losses
        
        Returns:
            concept_features: [B, d_model] - features for action model
            concept_probs: [B, num_concepts] - discovered concepts (0-1)
            losses: Dict of losses (if return_losses=True)
        """
        # ========== ENCODE: Embedding → Concepts ==========
        concept_logits = self.concept_encoder(embedding)  # [B, K]
        
        # Apply temperature-scaled sigmoid
        temperature = self.get_temperature()
        concept_probs = torch.sigmoid(concept_logits / temperature)  # [B, K]
        
        # ========== DECODE: Concepts → Reconstructed Embedding ==========
        reconstructed = self.concept_decoder(concept_probs)  # [B, d_model]
        
        # ========== PROJECT: Concepts → Features ==========
        concept_features = self.concept_to_features(concept_probs)  # [B, d_model]
        
        # ========== COMPUTE LOSSES ==========
        losses = None
        if return_losses:
            losses = self._compute_losses(embedding, reconstructed, concept_probs)
        
        # Increment step counter for temperature annealing
        if self.training:
            self.current_step += 1
        
        return concept_features, concept_probs, losses
    
    def _compute_losses(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        concept_probs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute concept discovery losses with improved formulations.
        """
        # ========== RECONSTRUCTION LOSS ==========
        reconstruction_loss = F.mse_loss(reconstructed, original)
        
        # ========== SPARSITY LOSS (L1-based, not entropy) ==========
        # ✅ FIXED: Encourage concepts to be 0 or 1 by penalizing middle values
        # L1 on distance from nearest binary value
        # For each concept, distance to nearest {0, 1} is min(c, 1-c)
        distance_to_binary = torch.min(concept_probs, 1 - concept_probs)
        sparsity_loss = distance_to_binary.mean()
        
        # Alternative: Penalize variance around 0.5
        # sparsity_loss = -((concept_probs - 0.5).pow(2)).mean()
        
        # ========== DIVERSITY LOSS (Orthogonality) ==========
        # ✅ FIXED: Encourage concept vectors to be orthogonal
        if concept_probs.shape[0] > 1:
            # Normalize concepts to unit vectors
            concept_normalized = F.normalize(concept_probs, p=2, dim=0)  # [B, K]
            
            # Compute Gram matrix (cosine similarities)
            gram = torch.mm(concept_normalized.T, concept_normalized)  # [K, K]
            
            # Penalize off-diagonal elements (should be near 0 for orthogonal)
            # Create mask for off-diagonal
            mask = 1.0 - torch.eye(self.num_concepts, device=gram.device)
            diversity_loss = (gram * mask).pow(2).sum() / (self.num_concepts * (self.num_concepts - 1))
        else:
            diversity_loss = torch.tensor(0.0, device=concept_probs.device)
        
        # ========== TOTAL LOSS ==========
        total_loss = (
            self.reconstruction_weight * reconstruction_loss +
            self.sparsity_weight * sparsity_loss +
            self.diversity_weight * diversity_loss
        )
        
        return {
            'concept_discovery_total': total_loss,
            'concept_reconstruction': reconstruction_loss,
            'concept_sparsity': sparsity_loss,
            'concept_diversity': diversity_loss,
        }


class ConceptDiscoveryLayer(nn.Module):
    """
    FIXED Concept Discovery Layer with better training stability.
    """
    
    def __init__(
        self,
        d_model: int,
        num_concepts: int = 32,  # ✅ Increased from 12
        dropout: float = 0.1,
        residual_weight: float = 0.8,  # ✅ Increased from 0.7 (more original info)
        sparsity_weight: float = 0.01,
        diversity_weight: float = 0.05,
        temperature: float = 2.0,
        temperature_decay_steps: int = 50000,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_concepts = num_concepts
        self.residual_weight = residual_weight
        
        # Generic concept names
        self.concept_names = [f"concept_{i:02d}" for i in range(num_concepts)]
        
        # ✅ Use fixed discovery module
        self.discovery = ConceptDiscoveryModule(
            d_model=d_model,
            num_concepts=num_concepts,
            hidden_dim=128,
            sparsity_weight=sparsity_weight,
            diversity_weight=diversity_weight,
            reconstruction_weight=1.0,
            initial_temperature=temperature,
            final_temperature=0.5,
            temperature_decay_steps=temperature_decay_steps,
        )
        
        print(f"✅ FIXED Concept Discovery Layer: {num_concepts} concepts")
        print(f"✅ Residual weight = {residual_weight:.1%} (softer bottleneck)")
        print(f"   → Action model receives: {residual_weight:.0%} original + {1-residual_weight:.0%} concepts")
    
    def forward(
        self,
        embedding: torch.Tensor,
        intervention_mask: Optional[torch.Tensor] = None,
        intervention_values: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Discover concepts and use them as bottleneck features.
        
        Returns:
            blended_features: [B, d_model]
            concept_predictions: dict of {concept_name: [B, 1]}
        """
        # Discover concepts (losses computed internally during training)
        concept_features, concept_probs, _ = self.discovery(embedding, return_losses=False)
        
        # ✅ Softer bottleneck: more residual connection
        blended_features = (
            (1.0 - self.residual_weight) * concept_features +
            self.residual_weight * embedding
        )
        
        # Convert to dictionary format
        concept_predictions = {}
        for i, name in enumerate(self.concept_names):
            concept_predictions[name] = concept_probs[:, i:i+1]  # [B, 1]
        
        return blended_features, concept_predictions
    
    def get_concept_values(self, embedding: torch.Tensor) -> Dict[str, float]:
        """Get concept values as dictionary (for visualization)."""
        with torch.no_grad():
            _, concept_probs, _ = self.discovery(embedding, return_losses=False)
            
            concept_dict = {}
            for i, name in enumerate(self.concept_names):
                concept_dict[name] = concept_probs[0, i].item()
            
            return concept_dict
    
    def get_discovery_losses(self, embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        ✅ CRITICAL: Get concept discovery losses for training.
        This MUST be called by the optimizer!
        """
        _, _, losses = self.discovery(embedding, return_losses=True)
        return losses if losses is not None else {}
    
    def get_temperature(self) -> float:
        """Get current temperature (for monitoring)."""
        return self.discovery.get_temperature()
