"""
Fully ONNX-compatible Transformer Actor for ML-Agents with TEMPORAL SEQUENCE Support.

Key Changes from Original:
- NEW: TemporalTransformerBody processes sequences of observations (L > 1)
- CBM (Concept Bottleneck Model) with direct concept bottleneck (no complex aggregation)
- LSTM removed - transformer handles temporal reasoning
- Shared Critic support added
- Context-aware memory handling (inference vs training)

Author: Temporal sequence version with proper CBM bottleneck
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union

from mlagents.trainers.buffer import AgentBuffer
from mlagents.trainers.torch_entities.agent_action import AgentAction
from mlagents.trainers.torch_entities.action_model import ActionModel
from mlagents_envs.base_env import ActionSpec, ObservationSpec
from mlagents.trainers.settings import NetworkSettings
from mlagents.trainers.torch_entities.networks import Actor, Critic
from mlagents.trainers.torch_entities.decoders import ValueHeads
from mlagents.trainers.torch_entities.model_serialization import exporting_to_onnx


# ============================================================================
# CONCEPT BOTTLENECK MODEL (CBM)
# ============================================================================

class ConceptDefinitions:
    """Define human-interpretable concepts for the drone navigation task."""
    
    # Navigation concepts
    DISTANCE_TO_TARGET = "distance_to_target"
    DIRECTION_TO_TARGET = "direction_alignment"
    VELOCITY_MAGNITUDE = "speed"
    VELOCITY_DIRECTION = "moving_towards_target"
    
    # Obstacle concepts
    OBSTACLE_FRONT = "obstacle_in_front"
    OBSTACLE_LEFT = "obstacle_on_left"
    OBSTACLE_RIGHT = "obstacle_on_right"
    MIN_OBSTACLE_DISTANCE = "closest_obstacle_distance"
    PATH_BLOCKED = "direct_path_blocked"
    
    # Environmental concepts
    WIND_STRENGTH = "wind_force_magnitude"
    WIND_ALIGNMENT = "wind_helps_or_hinders"
    
    # Behavioral concepts
    STUCK_STATE = "appears_stuck"
    EXPLORATION_MODE = "exploring_vs_exploiting"
    DETOUR_NEEDED = "needs_detour"
    
    @classmethod
    def get_all_concepts(cls) -> List[str]:
        return [
            cls.DISTANCE_TO_TARGET, cls.DIRECTION_TO_TARGET,
            cls.VELOCITY_MAGNITUDE, cls.VELOCITY_DIRECTION,
            cls.OBSTACLE_FRONT, cls.OBSTACLE_LEFT, cls.OBSTACLE_RIGHT,
            cls.MIN_OBSTACLE_DISTANCE, cls.PATH_BLOCKED,
            cls.WIND_STRENGTH, cls.WIND_ALIGNMENT,
            cls.STUCK_STATE, cls.EXPLORATION_MODE, cls.DETOUR_NEEDED,
        ]
    
    @classmethod
    def get_concept_type(cls, concept: str) -> str:
        """Return whether concept is binary or continuous."""
        binary_concepts = {
            cls.VELOCITY_DIRECTION, cls.OBSTACLE_FRONT,
            cls.OBSTACLE_LEFT, cls.OBSTACLE_RIGHT,
            cls.PATH_BLOCKED, cls.STUCK_STATE, cls.DETOUR_NEEDED,
        }
        return "binary" if concept in binary_concepts else "continuous"


class ConceptBottleneckLayer(nn.Module):
    """
    Proper Concept Bottleneck Layer for interpretable RL.
    
    Key change: Uses simple linear projection instead of complex aggregation
    to preserve interpretability of the bottleneck.
    """
    
    def __init__(
        self,
        d_model: int,
        num_concepts: int = None,
        concept_names: List[str] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.concept_names = concept_names or ConceptDefinitions.get_all_concepts()
        self.num_concepts = num_concepts or len(self.concept_names)
        self.d_model = d_model
        
        print(f"[OK] CBM: Predicting {self.num_concepts} interpretable concepts")
        
        # Concept prediction heads (output raw logits)
        self.concept_predictors = nn.ModuleDict()
        for concept_name in self.concept_names:
            self.concept_predictors[concept_name] = nn.Sequential(
                nn.Linear(d_model, 32),  # 128 → 32 (smaller!)
                nn.ReLU(),
                nn.Linear(32, 1),
            )
        
        # ✅ KEY CHANGE: Simple linear projection preserves interpretability
        # Maps concept vector [B, num_concepts] → [B, d_model]
        self.concept_to_features = nn.Sequential(
            nn.Linear(self.num_concepts, d_model),
            nn.LayerNorm(d_model),  # Just normalization, no complex transform
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize based on expected concept distributions."""
        # Concept-specific bias initialization
        concept_biases = {
            "obstacle_in_front": -1.0,      # Start pessimistic (more likely to say "no obstacle")
            "obstacle_on_left": -1.0,
            "obstacle_on_right": -1.0,
            "direct_path_blocked": -1.5,    # Even rarer
            "appears_stuck": -2.0,          # Very rare
            "needs_detour": -1.5,
            "moving_towards_target": 0.5,   # Slightly optimistic
            # Continuous concepts start at 0 (maps to 0.5 probability)
            "distance_to_target": 0.0,
            "direction_alignment": 0.0,
            "speed": 0.0,
            "closest_obstacle_distance": 0.0,
            "wind_force_magnitude": 0.0,
            "wind_helps_or_hinders": 0.0,
            "exploring_vs_exploiting": 0.0,
        }
    
        for concept_name, module in self.concept_predictors.items():
            for i, layer in enumerate(module):
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=1.0)  # Normal initialization
                
                    # Only set bias for final layer
                    if i == len(module) - 2:  # Second to last layer (before final output)
                        nn.init.zeros_(layer.bias)
                    elif i == len(module) - 1:  # Final layer
                        # Set bias based on concept
                        bias_value = concept_biases.get(concept_name, 0.0)
                        nn.init.constant_(layer.bias, bias_value)
    
    def forward(
        self, 
        embedding: torch.Tensor,
        intervention_mask: Optional[torch.Tensor] = None,
        intervention_values: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Predict concepts and use them as bottleneck features.
        
        Args:
            embedding: [B, d_model] - transformer output
            intervention_mask: [B, num_concepts] - which concepts to override
            intervention_values: [B, num_concepts] - override values (logits)
        
        Returns:
            concept_features: [B, d_model] - features derived from concepts
            concept_predictions: dict of {concept_name: [B, 1]} (RAW LOGITS)
        """
        concept_predictions = {}
        concept_logits_list = []
        
        # Step 1: Predict each concept independently
        for i, concept_name in enumerate(self.concept_names):
            logits = self.concept_predictors[concept_name](embedding)
            
            # Allow manual intervention (for testing interpretability)
            if intervention_mask is not None and intervention_values is not None:
                mask = intervention_mask[:, i:i+1]
                logits = torch.where(mask > 0.5, intervention_values[:, i:i+1], logits)
            
            concept_predictions[concept_name] = logits
            concept_logits_list.append(logits)
        
        # Step 2: Convert logits to probabilities
        concept_logits = torch.cat(concept_logits_list, dim=1)
        concept_probs = torch.sigmoid(concept_logits)
        
        # Step 3: ✅ Use concepts as bottleneck (single linear layer)
        # This is the KEY for interpretability - action model sees ONLY concepts
        concept_features = self.concept_to_features(concept_probs)
        
        return concept_features, concept_predictions
    
    def get_concept_values(self, embedding: torch.Tensor) -> Dict[str, float]:
        """
        Get concept values as human-readable dictionary.
        Useful for visualization and debugging.
        """
        with torch.no_grad():
            concept_dict = {}
            for concept_name in self.concept_names:
                logits = self.concept_predictors[concept_name](embedding)
                prob = torch.sigmoid(logits).item()
                concept_dict[concept_name] = prob
            return concept_dict


# ============================================================================
# TEMPORAL TRANSFORMER BODY
# ============================================================================

class TemporalTransformerBody(nn.Module):
    """
    Transformer that processes TEMPORAL SEQUENCES of observations.
    
    Optimized memory handling:
    - Distinguishes between inference (parallel envs) and training (buffer sampling)
    - Only logs warnings during inference when agents reset
    - Silent during training (expected batch size variations)
    """
    
    def __init__(
        self,
        observation_specs: List[ObservationSpec],
        d_model: int = 128,
        n_head: int = 4,
        n_layer: int = 2,
        ff_mult: int = 2,
        dropout: float = 0.1,
        memory_size: int = 0,
        capture_internals: bool = False,
        use_concept_bottleneck: bool = False,
        concept_names: List[str] = None,
    ):
        super().__init__()
        assert d_model % n_head == 0, f"d_model must be divisible by n_head"
        
        self.obs_dim = int(sum(int(np.prod(spec.shape)) for spec in observation_specs))
        self.d_model = d_model
        self.n_head = n_head
        self.n_layer = n_layer
        self.head_dim = d_model // n_head
        self.capture_internals = capture_internals
        self.use_concept_bottleneck = use_concept_bottleneck
        
        # Calculate sequence length from memory size
        self.memory_size = memory_size
        if memory_size > 0:
            self.sequence_length = memory_size // self.obs_dim
            if self.sequence_length < 2:
                self.sequence_length = 8
                self.memory_size = self.obs_dim * self.sequence_length
                print(f"[WARN] Adjusted memory_size to {self.memory_size} for sequence_length={self.sequence_length}")
        else:
            self.sequence_length = 1
        
        print(f"[OK] TemporalTransformer initialized: sequence_length={self.sequence_length}")
        
        self.internal_states = {}

        depth_scale = (6 * n_layer) ** -0.25
        # Input projection
        self.input_proj = nn.Linear(self.obs_dim, d_model)
        nn.init.orthogonal_(self.input_proj.weight, gain=np.sqrt(2))
        nn.init.zeros_(self.input_proj.bias)
        
        # Learnable temporal positional encoding
        self.temporal_pos_encoding = nn.Parameter(
            torch.randn(1, self.sequence_length, d_model) * (d_model ** -0.5)
        )

        self.attn_scales = nn.ParameterList([
            nn.Parameter(torch.ones(1) * 0.1) for _ in range(n_layer)
        ])
        self.ffn_scales = nn.ParameterList([
            nn.Parameter(torch.ones(1) * 0.1) for _ in range(n_layer)
        ])
        
        # Transformer layers
        self.qkv_layers = nn.ModuleList()
        self.attn_out_layers = nn.ModuleList()

        for i in range(n_layer):
            # QKV projection
            qkv = nn.Linear(d_model, d_model * 3)
            nn.init.xavier_uniform_(qkv.weight, gain=depth_scale)  # Depth-scaled
            nn.init.zeros_(qkv.bias)
            self.qkv_layers.append(qkv)
    
            # Attention output projection
            attn_out = nn.Linear(d_model, d_model)
            nn.init.xavier_uniform_(attn_out.weight, gain=depth_scale)  # Depth-scaled
            nn.init.zeros_(attn_out.bias)
            self.attn_out_layers.append(attn_out)
        
        self.attn_dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layer)])
        self.attn_out_dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layer)])
        self.ffn_dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layer)])
        
        # Feed-forward networks
        self.ffn_layers = nn.ModuleList()
        for i in range(n_layer):
            ffn = nn.Sequential(
                nn.Linear(d_model, d_model * ff_mult),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * ff_mult, d_model),
            )
            # Depth-scaled initialization
            nn.init.xavier_uniform_(ffn[0].weight, gain=depth_scale)
            nn.init.zeros_(ffn[0].bias)
            nn.init.xavier_uniform_(ffn[3].weight, gain=depth_scale)
            nn.init.zeros_(ffn[3].bias)
            self.ffn_layers.append(ffn)
        
        # Layer normalization
        self.norm1_layers = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layer)])
        self.norm2_layers = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layer)])
        self.final_norm = nn.LayerNorm(d_model)
        
        # Concept bottleneck
        if self.use_concept_bottleneck:
            self.concept_bottleneck = ConceptBottleneckLayer(
                d_model=d_model,
                concept_names=concept_names,
                dropout=dropout,
            )
            self.last_concept_predictions = {}
        
        self.encoding_size = d_model
        
        # Context tracking for smarter logging
        self._inference_batch_threshold = 32
        self._agent_reset_count = 0

    def _update_observation_buffer(
        self, 
        current_obs: torch.Tensor, 
        obs_buffer: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Maintain sliding window of recent observations."""
        B = current_obs.shape[0]
        
        if obs_buffer is None or obs_buffer.shape[0] != B:
            obs_buffer = current_obs.unsqueeze(1).repeat(1, self.sequence_length, 1)
        else:
            obs_buffer = torch.cat([
                obs_buffer[:, 1:, :],
                current_obs.unsqueeze(1)
            ], dim=1)
        
        return obs_buffer

    def forward(
        self,
        inputs: List[torch.Tensor],
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
        intervention_mask: Optional[torch.Tensor] = None,
        intervention_values: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        if self.capture_internals:
            self.internal_states = {}

        # Flatten and concatenate observations
        flat = [x.view(x.shape[0], -1).to(dtype=torch.float32) for x in inputs]
        current_obs = torch.cat(flat, dim=1)
        B = current_obs.shape[0]
        
        # Detect context: inference vs training
        is_inference_context = B <= self._inference_batch_threshold

        # ===== HANDLE MEMORY BUFFER =====
        if self.memory_size > 0:
            if memories is None or memories.numel() == 0:
                obs_buffer = current_obs.unsqueeze(1).repeat(1, self.sequence_length, 1)
                obs_sequence = obs_buffer
                memories_out = obs_buffer.reshape(B, -1)
            else:
                # Handle 3D tensor.
                # Two different shapes arrive depending on context:
                #   ONNX export/inference: Unity sends [B, 1, memory_size] → squeeze dim 1
                #   Training (Python):     previous step returned [1, B, memory_size] → squeeze dim 0
                # We use exporting_to_onnx.is_exporting() so the ONNX graph always
                # bakes in "squeeze axis 1", which is valid for any batch size at
                # runtime.  The training path uses squeeze(0) as before.
                if len(memories.shape) == 3:
                    if exporting_to_onnx.is_exporting():
                        memories = memories.squeeze(1)  # [B, 1, mem] → [B, mem]
                    else:
                        memories = memories.squeeze(0)  # [1, B, mem] → [B, mem]
                
                mem_batch_size = memories.shape[0]
                mem_size = memories.shape[-1]
                
                # Validate memory size
                if mem_size != self.memory_size:
                    if is_inference_context:
                        print(f"[WARN] Memory size mismatch: expected {self.memory_size}, got {mem_size}")
                    obs_buffer = current_obs.unsqueeze(1).repeat(1, self.sequence_length, 1)
                    obs_sequence = obs_buffer
                    memories_out = obs_buffer.reshape(B, -1)
                
                # Context-aware batch mismatch handling
                elif mem_batch_size != B:
                    # Only log during inference
                    if is_inference_context and not self.training:
                        self._agent_reset_count += 1
                        if self._agent_reset_count % 100 == 1:
                            print(f"[INFO] Agent resets: {self._agent_reset_count} ({mem_batch_size}->{B})")
                    
                    # Adjust memory
                    if mem_batch_size < B:
                        padding = torch.zeros(
                            (B - mem_batch_size, self.memory_size),
                            dtype=memories.dtype,
                            device=memories.device
                        )
                        memories = torch.cat([memories, padding], dim=0)
                    else:
                        memories = memories[:B]
                    
                    obs_buffer = memories.reshape(B, self.sequence_length, self.obs_dim)
                    obs_buffer = self._update_observation_buffer(current_obs, obs_buffer)
                    obs_sequence = obs_buffer
                    memories_out = obs_buffer.reshape(B, -1)
                
                else:
                    # Perfect match
                    obs_buffer = memories.reshape(B, self.sequence_length, self.obs_dim)
                    obs_buffer = self._update_observation_buffer(current_obs, obs_buffer)
                    obs_sequence = obs_buffer
                    memories_out = obs_buffer.reshape(B, -1)
        else:
            obs_sequence = current_obs.unsqueeze(1)
            memories_out = None

        # ===== TRANSFORMER PROCESSING =====
        L = obs_sequence.shape[1]
        x = self.input_proj(obs_sequence)

        if L == self.sequence_length:
            x = x + self.temporal_pos_encoding
        else:
            x = x + self.temporal_pos_encoding[:, :L, :]

        D = self.d_model

        for i in range(self.n_layer):
            x_norm = self.norm1_layers[i](x)
            qkv = self.qkv_layers[i](x_norm)
            qkv = qkv.reshape(B, L, 3, self.n_head, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        
            scale = self.head_dim ** -0.5
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn_weights = torch.softmax(attn, dim=-1)
            attn_weights_dropped = self.attn_dropout[i](attn_weights)
        
            out = torch.matmul(attn_weights_dropped, v)
            out = out.transpose(1, 2).reshape(B, L, D)
            out = self.attn_out_layers[i](out)
            out = self.attn_out_dropout[i](out)
            x = x + self.attn_scales[i] * out

            x_norm = self.norm2_layers[i](x)
            ffn_out = self.ffn_layers[i](x_norm)
            ffn_out = self.ffn_dropout[i](ffn_out)
            x = x + self.ffn_scales[i] * ffn_out

        x = self.final_norm(x)
        encoding = x[:, -1, :]

        # ===== CONCEPT BOTTLENECK =====
        concept_predictions = None
        if self.use_concept_bottleneck:
            # ✅ KEY: encoding is replaced by concept-based features
            encoding, concept_predictions = self.concept_bottleneck(
                encoding,
                intervention_mask=intervention_mask,
                intervention_values=intervention_values,
            )
            self.last_concept_predictions = concept_predictions

        # Format output
        # MemoryOutputApplier reads recurrent_out as [batch, 1, memory_size] (3D).
        # During training: unsqueeze(0) → [1, B, memory_size] (sequence wrapper).
        # During ONNX export: unsqueeze(1) → [B, 1, memory_size] so Sentis Width()
        # returns the correct memory_size and [agentIndex, 0, j] indexing works.
        if memories_out is not None:
            if exporting_to_onnx.is_exporting():
                memories_out = memories_out.unsqueeze(1)  # [B, 1, memory_size]
            else:
                memories_out = memories_out.unsqueeze(0)  # [1, B, memory_size]

        return encoding, memories_out

    def get_internal_states(self) -> Dict[str, Any]:
        return self.internal_states

    def update_normalization(self, buffer: AgentBuffer) -> None:
        return
    
    def get_last_concepts(self) -> Dict[str, torch.Tensor]:
        if hasattr(self, 'last_concept_predictions'):
            return self.last_concept_predictions
        return {}


# ============================================================================
# TRANSFORMER ACTOR
# ============================================================================

class TransformerActor(nn.Module, Actor, Critic):
    """Transformer-based actor network with Concept Bottleneck Model."""
    
    MODEL_EXPORT_VERSION = 3

    def __init__(
        self,
        observation_specs: List[ObservationSpec],
        network_settings: NetworkSettings,
        action_spec: ActionSpec,
        d_model: int = 128,
        n_head: int = 4,
        n_layer: int = 2,
        ff_mult: int = 2,
        dropout: float = 0.1,
        memory_size: int = 0,
        tanh_squash: bool = True,
        conditional_sigma: bool = False,
        capture_internals: bool = False,
        use_concept_bottleneck: bool = False,
        concept_names: List[str] = None,
        stream_names: List[str] = None,
    ):
        super().__init__()
        self.action_spec = action_spec
        self.use_concept_bottleneck = use_concept_bottleneck
        self.stream_names = stream_names
        
        # ML-Agents version parameters
        self.version_number = torch.nn.Parameter(
            torch.Tensor([self.MODEL_EXPORT_VERSION]), requires_grad=False
        )
        self.is_continuous_int_deprecated = torch.nn.Parameter(
            torch.Tensor([int(self.action_spec.is_continuous())]), requires_grad=False
        )
        self.continuous_act_size_vector = torch.nn.Parameter(
            torch.Tensor([int(self.action_spec.continuous_size)]), requires_grad=False
        )
        self.discrete_act_size_vector = torch.nn.Parameter(
            torch.Tensor([self.action_spec.discrete_branches]), requires_grad=False
        )
        self.act_size_vector_deprecated = torch.nn.Parameter(
            torch.Tensor(
                [self.action_spec.continuous_size + sum(self.action_spec.discrete_branches)]
            ),
            requires_grad=False,
        )

        # Create temporal transformer body
        self.network_body = TemporalTransformerBody(
            observation_specs,
            d_model=d_model,
            n_head=n_head,
            n_layer=n_layer,
            ff_mult=ff_mult,
            dropout=dropout,
            memory_size=memory_size,
            capture_internals=capture_internals,
            use_concept_bottleneck=use_concept_bottleneck,
            concept_names=concept_names,
        )
        
        self.encoding_size = self.network_body.encoding_size
        self._memory_size = self.network_body.memory_size
        
        print(f"[OK] TransformerActor: Final memory_size = {self._memory_size}")
        
        self.memory_size_vector = torch.nn.Parameter(
            torch.Tensor([int(self._memory_size)]), requires_grad=False
        )

        # Action head
        self.action_model = ActionModel(
            self.encoding_size,
            action_spec,
            conditional_sigma=conditional_sigma,
            tanh_squash=tanh_squash,
            deterministic=network_settings.deterministic,
        )
        
        # Value heads for shared critic
        if stream_names is not None:
            print(f"[OK] Creating value heads for shared critic: {stream_names}")
            self.value_heads = ValueHeads(stream_names, self.encoding_size)
        else:
            self.value_heads = None

        self._n_head = n_head
        self._export_obs_specs = observation_specs
        self.expect_flattened_obs_for_onnx = True

    @property
    def memory_size(self) -> int:
        return self._memory_size

    def update_normalization(self, buffer: AgentBuffer) -> None:
        self.network_body.update_normalization(buffer)

    def get_action_and_stats(
        self,
        inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
        intervention_mask: Optional[torch.Tensor] = None,
        intervention_values: Optional[torch.Tensor] = None,
    ) -> Tuple[AgentAction, Dict[str, Any], torch.Tensor]:
        encoding, memories = self.network_body(
            inputs, memories, sequence_length,
            intervention_mask, intervention_values
        )
        action, log_probs, entropies = self.action_model(encoding, masks)
        run_out: Dict[str, Any] = {
            "env_action": action.to_action_tuple(clip=self.action_model.clip_action),
            "log_probs": log_probs,
            "entropy": entropies,
        }
        
        if self.use_concept_bottleneck:
            concepts = self.network_body.get_last_concepts()
            if concepts:
                run_out["concepts"] = concepts
        
        return action, run_out, memories

    def get_stats(
        self,
        inputs: List[torch.Tensor],
        actions: AgentAction,
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Dict[str, Any]:
        encoding, _ = self.network_body(inputs, memories, sequence_length)
        log_probs, entropies = self.action_model.evaluate(encoding, masks, actions)
        
        stats = {"log_probs": log_probs, "entropy": entropies}
        
        if self.use_concept_bottleneck:
            concepts = self.network_body.get_last_concepts()
            if concepts:
                stats["concepts"] = concepts
        
        return stats

    # =========================================================================
    # CRITIC INTERFACE
    # =========================================================================
    
    def critic_pass(
        self,
        inputs: List[torch.Tensor],
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        encoding, memories_out = self.network_body(inputs, memories, sequence_length)
        
        if self.value_heads is not None:
            value_outputs = self.value_heads(encoding)
        else:
            raise RuntimeError(
                "TransformerActor.critic_pass called but no value_heads exist."
            )
        
        return value_outputs, memories_out

    def get_internal_states(self) -> Dict[str, Any]:
        return self.network_body.get_internal_states()

    def forward(
        self,
        inputs: Union[List[torch.Tensor], torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
    ) -> Tuple[Union[int, torch.Tensor], ...]:
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
    
        encoding, memories_out = self.network_body(inputs, memories, sequence_length=1)
        (
            cont_action_out,
            disc_action_out,
            action_out_deprecated,
            deterministic_cont_action_out,
            deterministic_disc_action_out,
        ) = self.action_model.get_action_out(encoding, masks)

        export_out = [self.version_number, self.memory_size_vector]
        if self.action_spec.continuous_size > 0:
            export_out += [
                cont_action_out,
                self.continuous_act_size_vector,
                deterministic_cont_action_out,
            ]
        if self.action_spec.discrete_size > 0:
            export_out += [
                disc_action_out,
                self.discrete_act_size_vector,
                deterministic_disc_action_out,
            ]
        if self.network_body.memory_size > 0:
            export_out += [memories_out]
        
        return tuple(export_out)
    
    def get_concept_explanations(self, threshold: float = 0.5) -> List[str]:
        """Get natural language explanations of current concepts."""
        if not self.use_concept_bottleneck:
            return []
        
        concepts = self.network_body.get_last_concepts()
        if not concepts:
            return []
        
        explanations = []
        
        for concept_name, pred_logits in concepts.items():
            logit_val = pred_logits.mean().item() if pred_logits.numel() > 1 else pred_logits.item()
            prob_value = torch.sigmoid(torch.tensor(logit_val)).item()
            
            if concept_name == ConceptDefinitions.DISTANCE_TO_TARGET:
                if prob_value < 0.33:
                    explanations.append("very close to target")
                elif prob_value > 0.66:
                    explanations.append("far from target")
                    
            elif concept_name == ConceptDefinitions.OBSTACLE_FRONT and prob_value > threshold:
                explanations.append("obstacle detected ahead")
                
            elif concept_name == ConceptDefinitions.PATH_BLOCKED and prob_value > threshold:
                explanations.append("direct path is blocked")
                
            elif concept_name == ConceptDefinitions.STUCK_STATE and prob_value > threshold:
                explanations.append("appears to be stuck")
                
            elif concept_name == ConceptDefinitions.DETOUR_NEEDED and prob_value > threshold:
                explanations.append("detour maneuver needed")
                
            elif concept_name == ConceptDefinitions.WIND_ALIGNMENT:
                if prob_value < 0.33:
                    explanations.append("wind is hindering movement")
                elif prob_value > 0.66:
                    explanations.append("wind is helping movement")
        
        return explanations
    
    def get_concept_predictions(self) -> Dict[str, torch.Tensor]:
        """Get raw concept predictions for analysis."""
        return self.network_body.get_last_concepts()
