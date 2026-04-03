from typing import Dict, cast, Tuple
import attr
import numpy as np
import copy

from mlagents.torch_utils import torch, default_device

from mlagents.trainers.buffer import AgentBuffer, BufferKey, RewardSignalUtil

from mlagents_envs.timers import timed
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.settings import (
    TrainerSettings,
    OnPolicyHyperparamSettings,
    ScheduleType,
)
from mlagents.trainers.torch_entities.networks import ValueNetwork
from mlagents.trainers.torch_entities.agent_action import AgentAction
from mlagents.trainers.torch_entities.action_log_probs import ActionLogProbs
from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents.trainers.trajectory import ObsUtil


@attr.s(auto_attribs=True)
class PPOSettings(OnPolicyHyperparamSettings):
    beta: float = 5.0e-3
    epsilon: float = 0.2
    lambd: float = 0.95
    num_epoch: int = 3
    shared_critic: bool = False
    learning_rate_schedule: ScheduleType = ScheduleType.LINEAR
    beta_schedule: ScheduleType = ScheduleType.LINEAR
    epsilon_schedule: ScheduleType = ScheduleType.LINEAR

    # Transformer parameters
    d_model: int = 128
    n_head: int = 4
    n_layer: int = 1
    ff_mult: int = 2
    dropout: float = 0.1
    tanh_squash: bool = True
    capture_internals: bool = False
    sequence_length: int = 64

    # Critic warmup: freeze actor for N steps, only train critic
    # (fixes performance dip when fine-tuning from merged checkpoints)
    critic_warmup_steps: int = 0

    # Staged unfreezing: freeze attention layers for N steps,
    # let FFN/LayerNorm/scales/action_head calibrate first
    attn_freeze_steps: int = 0

    # CBM-specific parameters
    use_concept_bottleneck: bool = False
    concept_loss_weight: float = 1.0
    concept_diversity_weight: float = 0.1
    warmup_concept_steps: int = 10000


class TorchPPOOptimizer(TorchOptimizer):
    def __init__(self, policy: TorchPolicy, trainer_settings: TrainerSettings):
        super().__init__(policy, trainer_settings)
        reward_signal_configs = trainer_settings.reward_signals
        reward_signal_names = [key.value for key, _ in reward_signal_configs.items()]

        self.hyperparameters: PPOSettings = cast(
            PPOSettings, trainer_settings.hyperparameters
        )

        params = list(self.policy.actor.parameters())
        
        if self.hyperparameters.shared_critic:
            print("✅ Using shared critic (actor network)")
            self._critic = policy.actor
        else:
            print("Creating SEPARATE ValueNetwork (standard critic)...")
            
            # Deep copy network settings to avoid modifying original
            network_settings_copy = copy.deepcopy(trainer_settings.network_settings)
            
            # Get actor's actual memory size
            actor_memory_size = policy.actor.memory_size
            print(f"📏 Actor memory_size: {actor_memory_size}")
            
            if actor_memory_size > 0:
                # Ensure memory settings exist
                if network_settings_copy.memory is None:
                    from mlagents.trainers.settings import NetworkSettings
                    network_settings_copy.memory = NetworkSettings.MemorySettings()
                    print("  Created new MemorySettings for critic")
                
                # CRITICAL: Set memory_size to match actor
                network_settings_copy.memory.memory_size = actor_memory_size
                
                print(f"  Set critic memory_size to: {actor_memory_size}")
                print(f"  Critic sequence_length: {network_settings_copy.memory.sequence_length}")
            else:
                print("  Actor has no memory, critic will have no memory")
            
            # Create standard ValueNetwork with synchronized settings
            self._critic = ValueNetwork(
                reward_signal_names,
                policy.behavior_spec.observation_specs,
                network_settings=network_settings_copy,
            )
            self._critic.to(default_device())
            params += list(self._critic.parameters())
            
            print(f"✅ ValueNetwork created successfully")
            print(f"  Critic memory_size: {self._critic.memory_size}")
            print(f"  Match status: {'MATCH ✓' if self._critic.memory_size == actor_memory_size else 'MISMATCH ✗'}\n")

        # Learning rate schedulers
        self.decay_learning_rate = ModelUtils.DecayedValue(
            self.hyperparameters.learning_rate_schedule,
            self.hyperparameters.learning_rate,
            1e-10,
            self.trainer_settings.max_steps,
        )
        self.decay_epsilon = ModelUtils.DecayedValue(
            self.hyperparameters.epsilon_schedule,
            self.hyperparameters.epsilon,
            0.1,
            self.trainer_settings.max_steps,
        )
        self.decay_beta = ModelUtils.DecayedValue(
            self.hyperparameters.beta_schedule,
            self.hyperparameters.beta,
            1e-5,
            self.trainer_settings.max_steps,
        )
        
        # ✅ NEW: Concept loss warmup scheduler
        if self.hyperparameters.use_concept_bottleneck:
            self.concept_loss_schedule = ModelUtils.DecayedValue(
                ScheduleType.LINEAR,
                0.1 * self.hyperparameters.concept_loss_weight,  # Start at 10% immediately
                self.hyperparameters.concept_loss_weight,  # Ramp to 100%
                2000,  # Much faster warmup
            )
            print(f"🎓 CBM warmup schedule: 0.0 → {self.hyperparameters.concept_loss_weight} over {self.hyperparameters.warmup_concept_steps} steps")

        self.optimizer = torch.optim.Adam(
            params, lr=self.trainer_settings.hyperparameters.learning_rate
        )
        
        self.stats_name_to_update_name = {
            "Losses/Value Loss": "value_loss",
            "Losses/Policy Loss": "policy_loss",
        }

        self.stream_names = list(self.reward_signals.keys())
        
        # Track concept training statistics
        self._concept_stats = {
            "total_updates": 0,
            "concept_loss_sum": 0.0,
            "concept_accuracy_sum": 0.0,
        }
        
        # ✅ NEW: Track gradient norms for debugging
        self._grad_norm_history = []

    @property
    def critic(self):
        return self._critic

    @timed
    def update(self, batch: AgentBuffer, num_sequences: int) -> Dict[str, float]:
        """
        Performs update on model.
        :param batch: Batch of experiences.
        :param num_sequences: Number of sequences to process.
        :return: Results of update.
        """
        # Get decayed parameters
        decay_lr = self.decay_learning_rate.get_value(self.policy.get_current_step())
        decay_eps = self.decay_epsilon.get_value(self.policy.get_current_step())
        decay_bet = self.decay_beta.get_value(self.policy.get_current_step())

        # ── Staged unfreezing: freeze qkv_layers + attn_out_layers ──
        attn_freeze_steps = self.hyperparameters.attn_freeze_steps
        if attn_freeze_steps > 0:
            current_step = self.policy.get_current_step()
            for name, param in self.policy.actor.named_parameters():
                if "qkv_layers" in name or "attn_out_layers" in name:
                    param.requires_grad = current_step >= attn_freeze_steps
            if current_step == attn_freeze_steps:
                print(f"[STAGED UNFREEZE] step {current_step} — attention layers unfrozen")
        
        returns = {}
        old_values = {}
        for name in self.reward_signals:
            old_values[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.value_estimates_key(name)]
            )
            returns[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.returns_key(name)]
            )

        n_obs = len(self.policy.behavior_spec.observation_specs)
        current_obs = ObsUtil.from_buffer(batch, n_obs)
        # Convert to tensors
        current_obs = [ModelUtils.list_to_tensor(obs) for obs in current_obs]

        act_masks = ModelUtils.list_to_tensor(batch[BufferKey.ACTION_MASK])
        actions = AgentAction.from_buffer(batch)

        memories = [
            ModelUtils.list_to_tensor(batch[BufferKey.MEMORY][i])
            for i in range(0, len(batch[BufferKey.MEMORY]), self.policy.sequence_length)
        ]
        if len(memories) > 0:
            memories = torch.stack(memories).unsqueeze(0)

        # Get value memories
        if self.hyperparameters.shared_critic:
            value_memories = memories  # ✅ Use same memory as actor
        else:
            value_memories = [
                ModelUtils.list_to_tensor(batch[BufferKey.CRITIC_MEMORY][i])
                for i in range(0, len(batch[BufferKey.CRITIC_MEMORY]), self.policy.sequence_length)
            ]
            if len(value_memories) > 0:
                value_memories = torch.stack(value_memories).unsqueeze(0)

        run_out = self.policy.actor.get_stats(
            current_obs,
            actions,
            masks=act_masks,
            memories=memories,
            sequence_length=self.policy.sequence_length,
        )

        log_probs = run_out["log_probs"]
        entropy = run_out["entropy"]
        
        # Extract concept predictions if CBM is enabled
        concept_predictions = run_out.get("concepts", None)

        values, _ = self.critic.critic_pass(
            current_obs,
            memories=value_memories,
            sequence_length=self.policy.sequence_length,
        )
        
        old_log_probs = ActionLogProbs.from_buffer(batch).flatten()
        log_probs = log_probs.flatten()
        loss_masks = ModelUtils.list_to_tensor(batch[BufferKey.MASKS], dtype=torch.bool)
        
        # =====================================================================
        # STANDARD PPO LOSSES
        # =====================================================================
        value_loss = ModelUtils.trust_region_value_loss(
            values, old_values, returns, decay_eps, loss_masks
        )
        policy_loss = ModelUtils.trust_region_policy_loss(
            ModelUtils.list_to_tensor(batch[BufferKey.ADVANTAGES]),
            log_probs,
            old_log_probs,
            loss_masks,
            decay_eps,
        )
        
        # Base loss (standard PPO)
        current_step = self.policy.get_current_step()
        # warmup_steps = self.hyperparameters.critic_warmup_steps
        # in_warmup = warmup_steps > 0 and current_step < warmup_steps
        # if in_warmup:
        #     print("in warmup")
        #     # Critic warmup: only train value network, freeze actor
        #     loss = 0.5 * value_loss
        # else:
        in_warmup = False
        loss = (
            policy_loss
            + 0.5 * value_loss
            - decay_bet * ModelUtils.masked_mean(entropy, loss_masks)
        )
        
        # =====================================================================
        # ✅ IMPROVED: Concept Loss with Gradual Warmup
        # =====================================================================
        concept_loss = torch.tensor(0.0, device=default_device())
        diversity_loss = torch.tensor(0.0, device=default_device())
        concept_accuracy = 0.0
        concept_weight = 0.0
        
        if (self.hyperparameters.use_concept_bottleneck and 
            concept_predictions is not None):
            
            current_step = self.policy.get_current_step()
            
            # ✅ Get scheduled concept loss weight (gradual ramp-up)
            concept_weight = self.concept_loss_schedule.get_value(current_step)
            
            # Only compute concept loss after some initial training
            if current_step >= 0:  # Small initial buffer for stability
                
                # Generate concept labels from observations
                concept_labels = self._generate_concept_labels(
                    batch, current_obs, n_obs
                )
                
                if concept_labels:
                    # Compute concept prediction loss
                    concept_loss, concept_accuracy = self._compute_concept_loss(
                        concept_predictions, concept_labels, loss_masks
                    )
                    
                    # Compute diversity loss (encourage diverse concept activations)
                    #diversity_loss = self._compute_diversity_loss(concept_predictions)
                    diversity_loss = torch.tensor(0.0, device=default_device())
                    
                    # ✅ Check for NaN before adding to loss
                    if not torch.isnan(concept_loss) and not torch.isnan(diversity_loss):
                        # Add to total loss with scheduled weight
                        loss = (
                            loss + 
                            concept_weight * concept_loss
                        )
                        
                        # Update statistics
                        self._concept_stats["total_updates"] += 1
                        self._concept_stats["concept_loss_sum"] += concept_loss.item()
                        self._concept_stats["concept_accuracy_sum"] += concept_accuracy
                    else:
                        print(f"⚠️  NaN detected in concept losses at step {current_step}, skipping concept update")
        
        # =====================================================================
        # ✅ IMPROVED: Gradient Computation with Better Clipping
        # =====================================================================
        
        # Set optimizer learning rate
        ModelUtils.update_learning_rate(self.optimizer, decay_lr)
        self.optimizer.zero_grad()
        
        # ✅ Check loss before backward
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️  Invalid loss detected at step {self.policy.get_current_step()}: {loss.item()}")
            print(f"  Policy Loss: {policy_loss.item()}")
            print(f"  Value Loss: {value_loss.item()}")
            print(f"  Concept Loss: {concept_loss.item()}")
            # Skip this update
            return {
                "Losses/Policy Loss": policy_loss.item() if not torch.isnan(policy_loss) else 0.0,
                "Losses/Value Loss": value_loss.item() if not torch.isnan(value_loss) else 0.0,
            }
        
        loss.backward()
        
        # ✅ IMPROVED: Gradient clipping on ALL parameters
        all_params = list(self.policy.actor.parameters())
        if not self.hyperparameters.shared_critic:
            all_params += list(self._critic.parameters())
        
        # Compute gradient norm before clipping (for monitoring)
        grad_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=float('inf'))
        self._grad_norm_history.append(grad_norm.item())
        if len(self._grad_norm_history) > 1000:
            self._grad_norm_history.pop(0)
        
        # ✅ Adaptive gradient clipping based on recent history
        if len(self._grad_norm_history) > 10:
            avg_grad_norm = np.mean(self._grad_norm_history[-100:])
            std_grad_norm = np.std(self._grad_norm_history[-100:])
            # Clip at mean + 2*std (allows for natural variation)
            max_grad_norm = min(5.0, avg_grad_norm + 2.0 * std_grad_norm)
        else:
            max_grad_norm = 2.0  # More permissive initial value
        
        # Apply clipping
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=max_grad_norm)
        
        self.optimizer.step()
        
        # =====================================================================
        # Prepare Update Statistics
        # =====================================================================
        # # Log warmup transitions
        # if in_warmup and current_step % 100000 == 0:
        #     print(f"  [CRITIC WARMUP] step {current_step}/{warmup_steps} — actor frozen, only critic training")
        # if warmup_steps > 0 and not in_warmup and current_step < warmup_steps + 50000 and current_step % 10000 == 0:
        #     if current_step == warmup_steps or (current_step - warmup_steps) < 10000:
        #         print(f"  [CRITIC WARMUP DONE] step {current_step} — actor unfrozen, full PPO training")

        update_stats = {
            "Losses/Policy Loss": torch.abs(policy_loss).item(),
            "Losses/Value Loss": value_loss.item(),
            "Policy/Learning Rate": decay_lr,
            "Policy/Epsilon": decay_eps,
            "Policy/Beta": decay_bet,
            "Policy/Gradient Norm": grad_norm.item(),
            "Policy/Critic Warmup": 1.0 if in_warmup else 0.0,
        }
        
        # Add concept statistics
        if self.hyperparameters.use_concept_bottleneck:
            update_stats["Concepts/Loss Weight"] = concept_weight
            
            if concept_loss.item() > 0:
                update_stats["Losses/Concept Loss"] = concept_loss.item()
                update_stats["Losses/Diversity Loss"] = diversity_loss.item()
                update_stats["Concepts/Accuracy"] = concept_accuracy
                
                # Add average concept accuracy over time
                if self._concept_stats["total_updates"] > 0:
                    avg_accuracy = (
                        self._concept_stats["concept_accuracy_sum"] / 
                        self._concept_stats["total_updates"]
                    )
                    update_stats["Concepts/Average Accuracy"] = avg_accuracy
        
        # ✅ Periodic detailed logging
        if self.policy.get_current_step() % 10000 == 0:
            print(f"\n📊 Training Stats @ Step {self.policy.get_current_step()}")
            print(f"  Policy Loss: {policy_loss.item():.4f}")
            print(f"  Value Loss: {value_loss.item():.4f}")
            if concept_weight > 0:
                print(f"  Concept Weight: {concept_weight:.4f}")
                print(f"  Concept Loss: {concept_loss.item():.4f}")
                print(f"  Concept Accuracy: {concept_accuracy:.4f}")
            print(f"  Gradient Norm: {grad_norm.item():.4f} (clipped at {max_grad_norm:.4f})")
            print(f"  Learning Rate: {decay_lr:.6f}")

        return update_stats
    
    # =========================================================================
    # CONCEPT LEARNING METHODS
    # =========================================================================
    
    def _generate_concept_labels(
        self,
        batch,
        observations: list,
        n_obs: int
    ) -> Dict[str, torch.Tensor]:
        """
        Build CBM supervision targets from the current mini-batch.

        Ray Perception Format (from Unity ML-Agents):
        Each ray produces (numDetectableTags + 2) values:
          [0 ... numDetectableTags-1]: One-hot encoding of which tag was hit (all 0s if no tag hit)
          [numDetectableTags]:         Miss flag (1.0 = missed everything, 0.0 = hit something)
          [numDetectableTags + 1]:     Hit fraction (normalized distance, 1.0 = max distance or miss)
    
        Example configurations:
          - 0 tags: [miss, hit_fraction]  → 2 values per ray
          - 1 tag:  [tag0_hit, miss, hit_fraction]  → 3 values per ray
          - 2 tags: [tag0_hit, tag1_hit, miss, hit_fraction]  → 4 values per ray
    
        Observations structure:
          - observations[0]: RayPerception [batch, ray_obs_size]
          - observations[1]: Vector [batch, 12]
              [0:3]  agent local pos (normalized)
              [3:6]  target local pos (normalized)
              [6:9]  wind vector (normalized)
              [9:12] agent velocity (normalized)
        """
        from mlagents.trainers.torch_entities.transformer_actor import ConceptDefinitions

        if not observations or len(observations) < 2:
            return {}

        device = observations[0].device
        B = observations[0].shape[0]

        # === CONSTANTS ===
        RAY_LENGTH = 100.0
        MAP_HALF_XZ = 500.0

        # Thresholds
        OBSTACLE_NEAR_THRESHOLD = 30.0
        PATH_BLOCKED_THRESHOLD = 20.0
        STUCK_SPEED_THRESHOLD = 0.1
        STUCK_DISTANCE_THRESHOLD = 15.0

        # ========================================================================
        # ✅ AUTOMATIC RAY FORMAT DETECTION
        # ========================================================================
        actual_ray_size = observations[0].shape[1]
        expected_vec_size = 12

        # Detect format: ray_obs_size = num_rays × (num_tags + 2)
        # Try different num_tags values
        detected_format = None
    
        for num_tags in range(0, 20):  # Support 0-19 tags
            per_ray = num_tags + 2
            if actual_ray_size % per_ray == 0:
                num_rays = actual_ray_size // per_ray
                if num_rays >= 1:
                    detected_format = {
                        'num_tags': num_tags,
                        'per_ray': per_ray,
                        'num_rays': num_rays,
                    }
                    break
    
        if detected_format is None:
            print(f"❌ Could not detect ray format for size {actual_ray_size}!")
            print(f"   Ray size must be divisible by (num_tags + 2)")
            return {}
    
        NUM_TAGS = detected_format['num_tags']
        PER_RAY = detected_format['per_ray']
        NUM_RAYS = detected_format['num_rays']
    
        # ✅ Identify indices for miss and hit_fraction
        MISS_INDEX = NUM_TAGS          # Position of miss flag
        HIT_FRAC_INDEX = NUM_TAGS + 1  # Position of hit fraction
    
        # Log format detection (only once)
        if not hasattr(self, '_ray_format_logged'):
            print(f"\n{'='*70}")
            print(f"🔍 RAY PERCEPTION FORMAT DETECTED (Unity ML-Agents)")
            print(f"{'='*70}")
            print(f"  Ray Observation Size:  {actual_ray_size}")
            print(f"  Number of Tags:        {NUM_TAGS}")
            print(f"  Values Per Ray:        {PER_RAY} (tags + miss + hit_fraction)")
            print(f"  Number of Rays:        {NUM_RAYS}")
            if NUM_TAGS > 0:
                print(f"  Tag encoding:          [0:{NUM_TAGS-1}] (one-hot)")
            print(f"  Miss flag index:       [{MISS_INDEX}] (1=miss, 0=hit)")
            print(f"  Hit fraction index:    [{HIT_FRAC_INDEX}] (0-1 normalized)")
            print(f"{'='*70}\n")
            self._ray_format_logged = True

        # Verify vector observations
        if observations[1].shape[1] != expected_vec_size:
            print(f"⚠️  Vector observation size mismatch! Expected {expected_vec_size}, got {observations[1].shape[1]}")

        # Containers
        concepts: Dict[str, list] = {k: [] for k in ConceptDefinitions.get_all_concepts()}

        for i in range(B):
            # ======== VECTOR OBSERVATIONS ========
            vec = observations[1][i].detach().cpu().numpy()
    
            # Parse vector observations
            ax, ay, az = vec[0:3]
            tx, ty, tz = vec[3:6]
            wx, wy, wz = vec[6:9]
            vx, vy, vz = vec[9:12]

            # Denormalize
            agent = np.array([ax * MAP_HALF_XZ, ay * 50.0, az * MAP_HALF_XZ], dtype=np.float32)
            targt = np.array([tx * MAP_HALF_XZ, ty * 50.0, tz * MAP_HALF_XZ], dtype=np.float32)
            wind = np.array([wx, wy, wz], dtype=np.float32)
            vel = np.array([vx, vy, vz], dtype=np.float32)

            # Compute derived quantities
            to_target = targt - agent
            tt_norm = np.linalg.norm(to_target) + 1e-8
            vel_mag = np.linalg.norm(vel) + 1e-8
            wind_mag = np.linalg.norm(wind) + 1e-8

            distance_norm = float(np.clip(
                1.0 - np.exp(-tt_norm / (MAP_HALF_XZ * 0.5)),
                0.0,
                1.0
            ))
    
            speed_norm = float(np.clip(vel_mag, 0.0, 1.0))
            speed_norm = 0.9 * speed_norm + 0.1 * 0.5
    
            moving_towards = float(1.0 if np.dot(vel / vel_mag, to_target / tt_norm) > 0.0 else 0.0)
    
            direction_alignment = float(np.clip(
                0.5 * (np.dot(to_target / tt_norm, vel / vel_mag) + 1.0),
                0.0,
                1.0
            ))
    
            wind_align_01 = float(np.clip(
                0.5 * (np.dot(wind / wind_mag, to_target / tt_norm) + 1.0),
                0.0,
                1.0
            ))

            # ========================================================================
            # ✅ RAY PARSING (Unity ML-Agents Official Format)
            # ========================================================================
            ray_flat = observations[0][i].detach().cpu().numpy()
            rays = ray_flat.reshape(NUM_RAYS, PER_RAY)

            # Extract components according to Unity format:
            # rays[:, 0:NUM_TAGS]      = one-hot tag encoding
            # rays[:, MISS_INDEX]      = miss flag (1=miss, 0=hit)
            # rays[:, HIT_FRAC_INDEX]  = hit fraction (0-1)
        
            miss_flags = rays[:, MISS_INDEX]      # 1.0 = missed, 0.0 = hit something
            hit_fracs = rays[:, HIT_FRAC_INDEX]   # Normalized distance (0-1)
        
            # ✅ Detect which rays hit obstacles
            # A ray hit an obstacle if:
            #   1. miss_flag = 0 (hit something)
            #   2. If tags exist, check if first tag (assumed to be geometry) was hit
        
            if NUM_TAGS == 0:
                # No tags: Use miss flag only
                # miss=0 means ray hit something (we assume it's an obstacle)
                hit_obstacle = (miss_flags < 0.5)
            
            elif NUM_TAGS == 1:
                # ✅ CORRECTED: 1 tag = "Target"
                # Infer obstacles: hit something but NOT the target
                target_hits = rays[:, 0]  # First (and only) tag is Target
                hit_something = (miss_flags < 0.5)  # miss=0 means hit something
    
                hit_target = (target_hits > 0.5)  # Explicitly hit target
                hit_obstacle = hit_something & (~hit_target)  # Hit something that's NOT target
            
            else:
                # 2+ tags: Assume first tag is "Geometry" (obstacles)
                # Common setup: tag0=Geometry, tag1=Target, etc.
                tag0_hits = rays[:, 0]  # First tag (Geometry)
                hit_obstacle = (tag0_hits > 0.5)
        
            # Convert hit fractions to actual distances
            dists = hit_fracs * RAY_LENGTH
        
            # ✅ Distance array: Use actual distance where obstacle hit, else max distance
            obs_dists = np.where(hit_obstacle, dists, RAY_LENGTH)

            # ========================================================================
            # ✅ RAY SECTORING (works for any odd number of rays)
            # ========================================================================
            if NUM_RAYS % 2 == 0:
                # Even number of rays: use middle two as "front"
                center_left = NUM_RAYS // 2 - 1
                center_right = NUM_RAYS // 2
                front_min = min(obs_dists[center_left], obs_dists[center_right])
                left_slice = slice(0, center_left)
                right_slice = slice(center_right + 1, NUM_RAYS)
            else:
                # Odd number of rays: use middle ray as "front"
                center = NUM_RAYS // 2
                front_min = float(obs_dists[center])
                left_slice = slice(0, center)
                right_slice = slice(center + 1, NUM_RAYS)

            # Compute sectoral minimums
            left_min = float(np.min(obs_dists[left_slice])) if left_slice.start < left_slice.stop else RAY_LENGTH
            right_min = float(np.min(obs_dists[right_slice])) if right_slice.start < right_slice.stop else RAY_LENGTH
            any_min = float(np.min(obs_dists))

            # Path blocked
            path_blocked = bool(front_min < PATH_BLOCKED_THRESHOLD)

            # ======== FILL CONCEPT LABELS ========
    
            # Obstacle concepts
            concepts[ConceptDefinitions.OBSTACLE_FRONT].append(
                1.0 if front_min < OBSTACLE_NEAR_THRESHOLD else 0.0
            )
            concepts[ConceptDefinitions.OBSTACLE_LEFT].append(
                1.0 if left_min < OBSTACLE_NEAR_THRESHOLD else 0.0
            )
            concepts[ConceptDefinitions.OBSTACLE_RIGHT].append(
                1.0 if right_min < OBSTACLE_NEAR_THRESHOLD else 0.0
            )
            concepts[ConceptDefinitions.PATH_BLOCKED].append(
                1.0 if path_blocked else 0.0
            )
    
            concepts[ConceptDefinitions.MIN_OBSTACLE_DISTANCE].append(
                min(any_min / RAY_LENGTH, 1.0)
            )

            # Navigation concepts
            concepts[ConceptDefinitions.DISTANCE_TO_TARGET].append(distance_norm)
            concepts[ConceptDefinitions.VELOCITY_MAGNITUDE].append(speed_norm)
            concepts[ConceptDefinitions.VELOCITY_DIRECTION].append(moving_towards)
            concepts[ConceptDefinitions.DIRECTION_TO_TARGET].append(direction_alignment)
    
            # Environmental concepts
            concepts[ConceptDefinitions.WIND_STRENGTH].append(
                float(np.clip(wind_mag, 0.0, 1.0))
            )
            concepts[ConceptDefinitions.WIND_ALIGNMENT].append(
                float(np.clip(wind_align_01, 0.0, 1.0))
            )

            # Behavioral concepts
            is_stuck = (
                speed_norm < STUCK_SPEED_THRESHOLD and 
                distance_norm > 0.15 and
                any_min < STUCK_DISTANCE_THRESHOLD
            )
            concepts[ConceptDefinitions.STUCK_STATE].append(1.0 if is_stuck else 0.0)
    
            needs_detour = path_blocked and distance_norm > 0.2
            concepts[ConceptDefinitions.DETOUR_NEEDED].append(1.0 if needs_detour else 0.0)
    
            concepts[ConceptDefinitions.EXPLORATION_MODE].append(
                float(np.clip(distance_norm, 0.0, 1.0))
            )

        # Convert to tensors
        out: Dict[str, torch.Tensor] = {}
        for k, vals in concepts.items():
            if not vals:
                continue
            t = torch.tensor(vals, dtype=torch.float32, device=device).unsqueeze(1)
            out[k] = torch.clamp(t, 0.0, 1.0)

        return out
    
    def _compute_concept_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
        loss_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """
        Computes CBM auxiliary loss.
        - Binary concepts: BCEWithLogitsLoss on raw logits.
        - Continuous concepts: MSE on sigmoid(logits) vs targets in [0,1].
        - Applies loss_masks (1=use, 0=ignore) per-sample.
        Returns:
          (total_concept_loss: torch.Tensor, avg_binary_accuracy: float)
        """

        # Configure losses
        bce_logits = torch.nn.BCEWithLogitsLoss(reduction="none")
        mse = torch.nn.MSELoss(reduction="none")

        # Binary concepts
        BINARY_CONCEPTS = {
            "moving_towards_target",
            "obstacle_in_front",
            "obstacle_on_left",
            "obstacle_on_right",
            "direct_path_blocked",
            "appears_stuck",
            "needs_detour",
        }

        if not predictions:
            device = labels[next(iter(labels))].device if labels else torch.device("cpu")
            return torch.tensor(0.0, device=device), 0.0

        device = next(iter(predictions.values())).device

        if loss_masks is None:
            any_pred = next(iter(predictions.values()))
            loss_masks = torch.ones_like(any_pred).view(-1)
        else:
            loss_masks = loss_masks.view(-1).to(device)
        loss_masks = (loss_masks > 0.5).float()

        # Aggregate losses and accuracy
        total_loss = 0.0
        used_concepts = 0

        bin_acc_sum = 0.0
        bin_acc_count = 0

        for name, pred_logits in predictions.items():
            if name not in labels:
                continue

            # Flatten to [B]
            p = pred_logits.view(-1).to(device)
            t = labels[name].view(-1).to(device)

            # Sanitize labels to [0,1]
            t = torch.nan_to_num(t, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

            # Ensure mask matches
            m = loss_masks
            if m.numel() != p.numel():
                m = torch.ones_like(p)

            # ✅ Check for NaN in predictions or labels
            if torch.isnan(p).any() or torch.isnan(t).any():
                print(f"⚠️  NaN detected in concept {name}, skipping")
                continue
            
            # ✅ Check for inf values
            if torch.isinf(p).any():
                print(f"⚠️  Inf detected in concept {name} predictions, clamping")
                p = torch.clamp(p, -10.0, 10.0)

            if name in BINARY_CONCEPTS:
                # BCE on logits
                vec_loss = bce_logits(p, t)
            
                # Accuracy on probabilities
                with torch.no_grad():
                    prob = torch.sigmoid(p)
                    pred_bin = (prob >= 0.5).float()
                    valid = m > 0.5
                    if valid.any():
                        acc = (pred_bin[valid].eq(t[valid].round()).float().mean()).item()
                        bin_acc_sum += acc
                        bin_acc_count += 1
            else:
                # Continuous: MSE on sigmoid(logit) vs target
                p_cont = torch.sigmoid(p).clamp(0.01, 0.99)
                vec_loss = mse(p_cont, t)

            # Masked mean for this concept
            if m.sum() > 0:
                concept_loss = (vec_loss * m).sum() / (m.sum() + 1e-8)
            
                # Check if loss is NaN
                if torch.isnan(concept_loss):
                    print(f"⚠️  NaN loss for concept {name}, skipping")
                    continue
                
                total_loss = total_loss + concept_loss
                used_concepts += 1

        if used_concepts == 0:
            return torch.tensor(0.0, device=device), 0.0

        avg_loss = total_loss / used_concepts
        avg_bin_acc = (bin_acc_sum / max(bin_acc_count, 1)) if bin_acc_count > 0 else 0.0

        return avg_loss, float(avg_bin_acc)
    

    def get_modules(self):
        modules = {
            "Optimizer:value_optimizer": self.optimizer,
            "Optimizer:critic": self._critic,
        }
        for reward_provider in self.reward_signals.values():
            modules.update(reward_provider.get_modules())
        return modules
