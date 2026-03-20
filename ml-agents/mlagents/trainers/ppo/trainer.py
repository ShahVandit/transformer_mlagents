# # Unity ML-Agents Toolkit
# ## ML-Agent Learning (PPO)
# Contains an implementation of PPO as described in: https://arxiv.org/abs/1707.06347

from typing import cast, Type, Union, Dict, Any

import numpy as np
import copy

from mlagents_envs.base_env import BehaviorSpec
from mlagents_envs.logging_util import get_logger
from mlagents.trainers.buffer import BufferKey, RewardSignalUtil
from mlagents.trainers.trainer.on_policy_trainer import OnPolicyTrainer
from mlagents.trainers.policy.policy import Policy
from mlagents.trainers.trainer.trainer_utils import get_gae
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.ppo.optimizer_torch import TorchPPOOptimizer, PPOSettings
from mlagents.trainers.trajectory import Trajectory
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents.trainers.settings import TrainerSettings

from mlagents.trainers.torch_entities.networks import SimpleActor, SharedActorCritic
from mlagents.trainers.torch_entities.transformer_actor import  TransformerActor #JW ADDED

logger = get_logger(__name__)

TRAINER_NAME = "ppo"


class PPOTrainer(OnPolicyTrainer):
    """The PPOTrainer is an implementation of the PPO algorithm."""

    def __init__(
        self,
        behavior_name: str,
        reward_buff_cap: int,
        trainer_settings: TrainerSettings,
        training: bool,
        load: bool,
        seed: int,
        artifact_path: str,
    ):
        """
        Responsible for collecting experiences and training PPO model.
        :param behavior_name: The name of the behavior associated with trainer config
        :param reward_buff_cap: Max reward history to track in the reward buffer
        :param trainer_settings: The parameters for the trainer.
        :param training: Whether the trainer is set for training.
        :param load: Whether the model should be loaded.
        :param seed: The seed the model will be initialized with
        :param artifact_path: The directory within which to store artifacts from this trainer.
        """
        super().__init__(
            behavior_name,
            reward_buff_cap,
            trainer_settings,
            training,
            load,
            seed,
            artifact_path,
        )
        self.hyperparameters: PPOSettings = cast(
            PPOSettings, self.trainer_settings.hyperparameters
        )
        self.seed = seed
        self.shared_critic = self.hyperparameters.shared_critic
        self.policy: TorchPolicy = None  # type: ignore

    def _process_trajectory(self, trajectory: Trajectory) -> None:
        """
        Takes a trajectory and processes it, putting it into the update buffer.
        Processing involves calculating value and advantage targets for model updating step.
        :param trajectory: The Trajectory tuple containing the steps to be processed.
        """
        super()._process_trajectory(trajectory)
        agent_id = trajectory.agent_id  # All the agents should have the same ID

        agent_buffer_trajectory = trajectory.to_agentbuffer()
        # Check if we used group rewards, warn if so.
        self._warn_if_group_reward(agent_buffer_trajectory)

        # Update the normalization
        if self.is_training:
            self.policy.actor.update_normalization(agent_buffer_trajectory)
            self.optimizer.critic.update_normalization(agent_buffer_trajectory)

        # Get all value estimates
        (
            value_estimates,
            value_next,
            value_memories,
        ) = self.optimizer.get_trajectory_value_estimates(
            agent_buffer_trajectory,
            trajectory.next_obs,
            trajectory.done_reached and not trajectory.interrupted,
        )
        if value_memories is not None:
            agent_buffer_trajectory[BufferKey.CRITIC_MEMORY].set(value_memories)

        for name, v in value_estimates.items():
            agent_buffer_trajectory[RewardSignalUtil.value_estimates_key(name)].extend(
                v
            )
            self._stats_reporter.add_stat(
                f"Policy/{self.optimizer.reward_signals[name].name.capitalize()} Value Estimate",
                np.mean(v),
            )

        # Evaluate all reward functions
        self.collected_rewards["environment"][agent_id] += np.sum(
            agent_buffer_trajectory[BufferKey.ENVIRONMENT_REWARDS]
        )
        for name, reward_signal in self.optimizer.reward_signals.items():
            evaluate_result = (
                reward_signal.evaluate(agent_buffer_trajectory) * reward_signal.strength
            )
            agent_buffer_trajectory[RewardSignalUtil.rewards_key(name)].extend(
                evaluate_result
            )
            # Report the reward signals
            self.collected_rewards[name][agent_id] += np.sum(evaluate_result)

        # Compute GAE and returns
        tmp_advantages = []
        tmp_returns = []
        for name in self.optimizer.reward_signals:
            bootstrap_value = value_next[name]

            local_rewards = agent_buffer_trajectory[
                RewardSignalUtil.rewards_key(name)
            ].get_batch()
            local_value_estimates = agent_buffer_trajectory[
                RewardSignalUtil.value_estimates_key(name)
            ].get_batch()

            local_advantage = get_gae(
                rewards=local_rewards,
                value_estimates=local_value_estimates,
                value_next=bootstrap_value,
                gamma=self.optimizer.reward_signals[name].gamma,
                lambd=self.hyperparameters.lambd,
            )
            local_return = local_advantage + local_value_estimates
            # This is later use as target for the different value estimates
            agent_buffer_trajectory[RewardSignalUtil.returns_key(name)].set(
                local_return
            )
            agent_buffer_trajectory[RewardSignalUtil.advantage_key(name)].set(
                local_advantage
            )
            tmp_advantages.append(local_advantage)
            tmp_returns.append(local_return)

        # Get global advantages
        global_advantages = list(
            np.mean(np.array(tmp_advantages, dtype=np.float32), axis=0)
        )
        global_returns = list(np.mean(np.array(tmp_returns, dtype=np.float32), axis=0))
        agent_buffer_trajectory[BufferKey.ADVANTAGES].set(global_advantages)
        agent_buffer_trajectory[BufferKey.DISCOUNTED_RETURNS].set(global_returns)

        self._append_to_update_buffer(agent_buffer_trajectory)

        # If this was a terminal trajectory, append stats and reset reward collection
        if trajectory.done_reached:
            self._update_end_episode_stats(agent_id, self.optimizer)

    def create_optimizer(self) -> TorchOptimizer:
        return TorchPPOOptimizer(  # type: ignore
            cast(TorchPolicy, self.policy), self.trainer_settings  # type: ignore
        )  # type: ignore

  # In trainer.py - FIND this method and UPDATE the transformer section:

    def create_policy(
        self, parsed_behavior_id: BehaviorIdentifiers, behavior_spec: BehaviorSpec
    ) -> TorchPolicy:
        """
        Creates a policy with a PyTorch backend and PPO hyperparameters
        """
        ns = self.trainer_settings.network_settings
        use_tfm = bool(getattr(ns, "use_transformer", False))

        # === TRANSFORMER PATH ===
        if use_tfm:
            hp = self.hyperparameters
    
            # Calculate memory_size from sequence_length
            memory_size = 0
            sequence_length = hp.sequence_length
    
            if sequence_length > 1:
                obs_size = sum(
                    int(np.prod(spec.shape)) 
                    for spec in behavior_spec.observation_specs
                )
                memory_size = obs_size * sequence_length
        
                print(f"\n{'='*60}")
                print(f"TEMPORAL TRANSFORMER CONFIGURATION")
                print(f"{'='*60}")
                print(f"  Sequence Length:    {sequence_length} timesteps")
                print(f"  Observation Size:   {obs_size}")
                print(f"  Memory Size:        {memory_size} (auto-calculated)")
                print(f"  d_model:            {getattr(hp, 'd_model', 128)}")
                print(f"  n_heads:            {getattr(hp, 'n_head', 4)}")
                print(f"  n_layers:           {getattr(hp, 'n_layer', 2)}")
                print(f"  Concept Bottleneck: {getattr(hp, 'use_concept_bottleneck', False)}")
                print(f"  Shared Critic:      {getattr(hp, 'shared_critic', False)}")
                print(f"{'='*60}\n")
            else:
                print(f"\n⚠️  Transformer with sequence_length=1 (no temporal memory)")
    
            actor_cls = TransformerActor
            actor_kwargs = dict(
                d_model=getattr(hp, "d_model", 128),
                n_head=getattr(hp, "n_head", 4),
                n_layer=getattr(hp, "n_layer", 2),
                ff_mult=getattr(hp, "ff_mult", 2),
                dropout=getattr(hp, "dropout", 0.1),
                tanh_squash=getattr(hp, "tanh_squash", True),
                capture_internals=getattr(hp, "capture_internals", False),
                memory_size=memory_size,
                use_concept_bottleneck=getattr(hp, "use_concept_bottleneck", False),
                # concept_names removed - concepts discovered automatically!
            )

            if bool(getattr(hp, "shared_critic", False)):
                reward_signal_configs = self.trainer_settings.reward_signals
                reward_signal_names = [key.value for key, _ in reward_signal_configs.items()]
                actor_kwargs["stream_names"] = reward_signal_names
                print(f"✅ Shared critic enabled with streams: {reward_signal_names}")
        
            # ✅ CRITICAL: Create custom network_settings for TorchPolicy
            # This prevents Policy.__init__ from reading wrong memory settings
            custom_network_settings = copy.deepcopy(self.trainer_settings.network_settings)
        
            # ✅ FIX: Create temporary memory settings to make Policy think it's recurrent
            if memory_size > 0:
                from mlagents.trainers.settings import NetworkSettings
                custom_network_settings.memory = NetworkSettings.MemorySettings(
                    sequence_length=sequence_length,
                    memory_size=memory_size
                )
    
            policy = TorchPolicy(
                self.seed,
                behavior_spec,
                custom_network_settings,  # ← Use modified settings!
                actor_cls,
                actor_kwargs,
            )
        
            # ✅ Verify the settings stuck
            print(f"\n{'='*60}")
            print(f"POLICY VERIFICATION")
            print(f"{'='*60}")
            print(f"  policy.m_size:          {policy.m_size}")
            print(f"  policy.sequence_length: {policy.sequence_length}")
            print(f"  policy.use_recurrent:   {policy.use_recurrent}")
            print(f"  actor.memory_size:      {policy.actor.memory_size}")
            print(f"{'='*60}\n")
    
            if bool(getattr(hp, "shared_critic", False)):
                policy.critic = policy.actor
    
            return policy 

        # === NON-TRANSFORMER PATH (unchanged) ===
        actor_cls2 = SimpleActor
        actor_kwargs2 = {"conditional_sigma": False, "tanh_squash": False}

        if self.shared_critic:
            reward_signal_configs = self.trainer_settings.reward_signals
            reward_signal_names = [key.value for key, _ in reward_signal_configs.items()]
            actor_cls2 = SharedActorCritic
            actor_kwargs2.update({"stream_names": reward_signal_names})

        policy = TorchPolicy(
            self.seed,
            behavior_spec,
            self.trainer_settings.network_settings,
            actor_cls2,
            actor_kwargs2,
        )
        return policy



    def get_policy(self, name_behavior_id: str) -> Policy:
        """
        Gets policy from trainer associated with name_behavior_id
        :param name_behavior_id: full identifier of policy
        """

        return self.policy

    @staticmethod
    def get_trainer_name() -> str:
        return TRAINER_NAME
