import torch
import copy
from typing import Callable, Optional

from benchmarl.environments import VmasTask
from benchmarl.utils import DEVICE_TYPING
from tensordict import TensorDictBase
from torch.distributions.constraints import positive
from torchrl.envs import EnvBase, VmasEnv
from vmas.simulator.core import Agent, Landmark, Line, Sphere, World, Action
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils
from vmas.simulator import rendering  # comment if using headless mode (e.g., using server)


class Scenario(BaseScenario):
    """
    Defines the Multi-Agent Pursuit-Evasion scenario with Early Termination.

    In this scenario, a team of pursuers (adversaries) learns to capture a team of
    evaders (good agents). The key features include:
    - Early Termination: Agents that are captured or have captured a target become
      inactive and are moved to a designated 'heaven' location.
    - Hierarchical Reward: A combination of sparse capture rewards, dense team-level
      progress rewards (using the Sinkhorn algorithm for optimal transport cost),
      and individual shaping rewards/penalties.
    - Hybrid Evader Policy: Evaders employ a non-learning, rule-based policy that
      switches between emergency evasion (using Artificial Potential Fields) and
      guided exploration (using Levy flights).
    """
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        num_good_agents = kwargs.pop("num_good_agents", 4)
        num_adversaries = kwargs.pop("num_adversaries", 5)
        num_landmarks = kwargs.pop("num_landmarks", 2)
        self.shape_agent_rew = kwargs.pop("shape_agent_rew", False)
        self.shape_adversary_rew = kwargs.pop("shape_adversary_rew", True)
        self.agents_share_rew = kwargs.pop("agents_share_rew", False)
        self.adversaries_share_rew = kwargs.pop("adversaries_share_rew", True)
        self.observe_same_team = kwargs.pop("observe_same_team", True)
        self.observe_pos = kwargs.pop("observe_pos", True)
        self.observe_vel = kwargs.pop("observe_vel", True)
        self.bound = kwargs.pop("bound", 1.3)
        self.respawn_at_catch = kwargs.pop("respawn_at_catch", False)
        self.max_speed_adv = kwargs.pop("max_speed_adv", 0.4)
        self.max_speed_good = kwargs.pop("max_speed_good", 0.4)

        # Termination coefficients
        self.adv_heaven_reward = kwargs.pop("adv_heaven_reward", 20.0)  # one time reward for a successful capture
        self.agent_heaven_penalty = kwargs.pop("agent_heaven_penalty", -10.0)  # one time penalty for being captured
        self.heaven_size = 0.1  # 'heaven' location is plot as a circle
        self.heaven_position = torch.tensor([0.0, self.bound + self.heaven_size * 2], device=device,
                                            dtype=torch.float32)

        # reward shaping coefficients
        self.boundary_penalty_coef = kwargs.pop("boundary_penalty_coef", 0.15)
        self.collision_penalty_coef = kwargs.pop("collision_penalty_coef", 0.5)
        self.obstacle_penalty_coef = kwargs.pop("obstacle_penalty_coef", 0.2)
        self.distance_reward_coef = kwargs.pop("distance_reward_coef", 0.3)
        self.smoothing_distance_coef = kwargs.pop("smoothing_distance_coef", 0.7)
        self.oscillation_penalty_coef = kwargs.pop("oscillation_penalty_coef", 0.01)
        self.oscillation_vel_threshold = kwargs.pop("oscillation_vel_threshold", 0.1)
        self.closing_speed_reward_coef = kwargs.pop("closing_speed_reward_coef", 0.15)
        self.time_reward_coef = kwargs.pop("time_reward_coef", 0.5)
        self.time_pressure_coef = kwargs.pop("time_pressure_coef", 0.05)
        self.not_catch_penalty = kwargs.pop("not_catch_penalty", -0.0)
        self.max_steps = kwargs.pop("max_steps", 100)
        self.current_step = torch.zeros(batch_dim, device=device, dtype=torch.int32)
        self.scale_collision = 2
        self.min_dist_clamp = kwargs.pop("min_dist_clamp", 1.0)

        # Evader policy 1 (APF) coefficients
        self.adv_repulsion_coef_close = kwargs.pop("adv_repulsion_coef_close", 0.5)  # Repulsion coefficient from adversaries at close range
        self.adv_repulsion_coef_far = kwargs.pop("adv_repulsion_coef_far", 0.00)  # Repulsion coefficient from adversaries at far range
        self.adv_safe_dist_extra = kwargs.pop("adv_safe_dist_extra", 0.4)  # defined what distance is "close" for adversaries
        self.landmark_repulsion_coef = kwargs.pop("landmark_repulsion_coef", 0.01)  # Repulsion coefficient from landmarks
        self.landmark_safe_dist_extra = kwargs.pop("landmark_safe_dist_extra", 0.25)
        self.boundary_repulsion_coef = kwargs.pop("boundary_repulsion_coef", 0.01)  # Repulsion coefficient from boundaries
        self.boundary_safe_dist_factor = kwargs.pop("boundary_safe_dist_factor", 2.0)
        self.teammate_repulsion_coef = kwargs.pop("teammate_repulsion_coef", 0.01)  # Repulsion coefficient from good agents
        self.teammate_safe_dist = kwargs.pop("teammate_safe_dist", 0.25)

        # Evader policy 2 (Levy) coefficients
        self.apf_trigger_distance_adv = kwargs.pop("apf_trigger_distance_adv", 0.4)  # Distance at which evaders switch to APF mode when near a pursuer
        self.guidance_force_coef = kwargs.pop("guidance_force_coef", 0.3)  # guidance force = "guidance_force_coef"*"max_speed_good"
        self.max_acceleration = kwargs.pop("max_acceleration", 0.2)  # max_acceleration = "max_acceleration"*"max_speed_adv"
        self.direction_change_interval = kwargs.pop("direction_change_interval", 30)
        self.levy_alpha = kwargs.pop("levy_alpha", 1.5)  # Levy exponent (typically 1 < alpha <= 2)
        # This scale parameter affects the step size of the Levy flight
        self.levy_scale = kwargs.pop("levy_scale", 0.1)  # Scale parameter for Levy steps
        self.target_velocity = None
        self.direction_timer = None
        self.rng_levy = torch.Generator(device=device)

        ScenarioUtils.check_kwargs_consumed(kwargs)

        self.visualize_semidims = False

        world = World(
            batch_dim=batch_dim,
            device=device,
            x_semidim=self.bound,
            y_semidim=self.bound,
            substeps=10,
            collision_force=500,
        )
        # set any world properties first
        num_agents = num_adversaries + num_good_agents
        self.adversary_radius = 0.075

        # Add agents
        for i in range(num_agents):
            adversary = True if i < num_adversaries else False
            name = f"adversary_{i}" if adversary else f"agent_{i - num_adversaries}"
            agent = Agent(
                name=name,
                render_action=True,
                collide=True,
                shape=Sphere(radius=self.adversary_radius if adversary else 0.05),
                u_multiplier=1.0 if adversary else 1.0,
                max_speed=self.max_speed_adv if adversary else self.max_speed_good,
                color=Color.RED if adversary else Color.GREEN,
                adversary=adversary,
            )

            # agent active if not collide with other team
            agent.active = torch.ones(batch_dim, device=device, dtype=torch.bool)
            # attribute to record whether heaven reward has been given in a single episode
            agent.heaven_reward_given = torch.zeros(batch_dim, device=device, dtype=torch.bool)

            # add previous velocity for adversaries (for oscillation penalty)
            if adversary:
                agent.prev_vel = torch.zeros((batch_dim, world.dim_p), device=device, dtype=torch.float32)

            world.add_agent(agent)
        # Add landmarks
        for i in range(num_landmarks):
            landmark = Landmark(
                name=f"landmark {i}",
                collide=True,
                shape=Sphere(radius=0.2),
                color=Color.BLACK,
            )
            world.add_landmark(landmark)

        return world

    def reset_world_at(self, env_index: int = None):
        """
        Resets the world state for one or all environments in the batch.

        This function handles two scenarios based on the `env_index` parameter:
        1.  Full Reset (`env_index` is None): Resets all environments. This is
            typically called at the beginning of a training run. It re-initializes
            all states, including agent positions and the evaders' Levy flight targets.
        2.  Partial Reset (`env_index` is not None): Resets a single environment
            that has finished its episode. This is crucial for vectorized training in
            BenchMARL framework to continue uninterrupted.

        Args:
            env_index: The index of the environment to reset. If None, resets all.
        """
        num_good_agents = len(self.good_agents())

        if env_index is None:  # Full reset for all environments

            # For full reset, initialize for the whole batch
            self.prev_assignment_cost = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.float32)

            if num_good_agents > 0:
                if self.target_velocity is None or \
                        self.target_velocity.shape[0] != self.world.batch_dim or \
                        self.target_velocity.shape[1] != num_good_agents:
                    # Initialize or re-initialize if batch_dim or num_good_agents changed
                    self.target_velocity = torch.zeros(
                        (self.world.batch_dim, num_good_agents, self.world.dim_p),
                        device=self.world.device,
                        dtype=torch.float32,
                    )
                    self.direction_timer = torch.zeros(
                        (self.world.batch_dim, num_good_agents),
                        device=self.world.device,
                        dtype=torch.int32,
                    )

                # Initialize the evaders' long-term Levy flight velocity targets
                num_to_init = self.world.batch_dim * num_good_agents
                step_lengths = self._sample_levy_steps(num_to_init, self.world.device)
                theta = torch.rand(num_to_init, generator=self.rng_levy, device=self.world.device) * 2 * torch.pi

                # Reshape the 1D velocity vectors and assign them to the 2D target tensor
                self.target_velocity[..., 0] = (step_lengths * torch.cos(theta)).view(self.world.batch_dim,
                                                                                      num_good_agents)
                self.target_velocity[..., 1] = (step_lengths * torch.sin(theta)).view(self.world.batch_dim,
                                                                                      num_good_agents)

                self.direction_timer[:] = 0  # The direction change timer must be reset

            else:  # No good agents
                self.target_velocity = None
                self.direction_timer = None

            self.current_step[:] = 0  # Moved here for full reset scope

        else:  # Partial reset for a specific environment
            if self.target_velocity is not None:
                # Assign a new random Levy flight target for the specific environment
                num_to_init = num_good_agents
                step_lengths = self._sample_levy_steps(num_to_init, self.world.device)
                theta = torch.rand(num_to_init, generator=self.rng_levy, device=self.world.device) * 2 * torch.pi
                self.target_velocity[env_index, :, 0] = step_lengths * torch.cos(theta)
                self.target_velocity[env_index, :, 1] = step_lengths * torch.sin(theta)

            if self.direction_timer is not None:
                self.direction_timer[env_index] = 0

            self.current_step[env_index] = 0

            # For partial reset
            self.prev_assignment_cost[env_index] = 0.0

        # Reset agent-specific parameters
        for agent in self.world.agents:

            if not hasattr(agent, 'rew'):
                agent.rew = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.float32)

            # Reset agent state for the specific environment index
            if env_index is not None:
                agent.active[env_index] = True
                agent.heaven_reward_given[env_index] = False

                # Reset pursuer-specific attributes
                agent.rew[env_index] = 0.0  # Reset reward accumulator
                if agent.adversary:
                    agent.prev_vel[env_index] = 0.0

            else:
                agent.active[:] = True
                agent.heaven_reward_given[:] = False

                # Reset pursuer-specific attributes
                agent.rew[:] = 0.0  # Reset reward accumulator
                if agent.adversary:
                    agent.prev_vel[:] = 0.0

            # Randomly initialize agent positions within the environment bounds in env_index
            agent.set_pos(
                torch.zeros(
                    (
                        (1, self.world.dim_p)
                        if env_index is not None
                        else (self.world.batch_dim, self.world.dim_p)
                    ),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -self.bound,
                    self.bound,
                ),
                batch_index=env_index,
            )

        for landmark in self.world.landmarks:
            landmark.set_pos(
                torch.zeros(
                    (
                        (1, self.world.dim_p)
                        if env_index is not None
                        else (self.world.batch_dim, self.world.dim_p)
                    ),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -(self.bound - 0.1),
                    self.bound - 0.1,
                ),
                batch_index=env_index,
            )

    def is_collision(self, agent1: Agent, agent2: Agent):
        delta_pos = agent1.state.pos - agent2.state.pos
        dist = torch.linalg.vector_norm(delta_pos, dim=-1)
        dist_min = (agent1.shape.radius + agent2.shape.radius) * self.scale_collision
        return (dist < dist_min) & agent1.active & agent2.active  # ignore collisions between inactive agents (because they are in same heaven loc)

    # detect all collisions in all batch world
    def detect_collisions(self):
        adversaries = self.adversaries()
        good_agents = self.good_agents()

        # Obtain all agents' positions/active status
        adv_pos = torch.stack([a.state.pos for a in adversaries], dim=1)  # (batch_dim, n_adv, 2)
        good_pos = torch.stack([a.state.pos for a in good_agents], dim=1)  # (batch_dim, n_good, 2)
        adv_active = torch.stack([a.active for a in adversaries], dim=1)  # (batch_dim, n_adv)
        good_active = torch.stack([a.active for a in good_agents], dim=1)  # (batch_dim, n_good)

        # calculate the distance between each pair of agents
        delta_pos = adv_pos.unsqueeze(2) - good_pos.unsqueeze(1)  # (batch_dim, n_adv, n_good, 2)
        dist = torch.linalg.vector_norm(delta_pos, dim=-1)  # (batch_dim, n_adv, n_good)
        dist_min = (adversaries[0].shape.radius + good_agents[0].shape.radius) * self.scale_collision  # homo agents

        # Just consider active agents' collision
        active_mask = adv_active.unsqueeze(2) & good_active.unsqueeze(1)  # (batch_dim, n_adv, n_good)
        collisions = (dist < dist_min) & active_mask  # (batch_dim, n_adv, n_good)

        return collisions  # collision matrix

    # process collision
    def process_collisions(self, collisions):
        """
        Processes raw collision data to determine unique capture events.

        This function resolves ambiguities when multiple pursuers might collide with
        the same evader in a single timestep. It implements a greedy assignment
        protocol: each pursuer that is involved in one or more collisions "claims"
        the single closest evader it is colliding with. An evader can only be
        claimed once per timestep.

        If a unique capture is confirmed, this function will:
        1.  Update the `active` status of the involved agents to `False`.
        2.  Move the agents to the pre-defined 'heaven' location.
        3.  Apply a one-time capture reward (for the pursuer) or penalty (for the evader).

        Args:
            collisions: A boolean matrix tensor of shape (batch, n_pursuers, n_evaders) indicating potential collisions.
            (noted that collisions is feed by the return of detect_collisions())

        Returns:
            A boolean value of `True` if at least one successful capture occurred,
            otherwise `False`. This is used to conditionally apply other rewards.
        """
        adversaries = self.adversaries()
        good_agents = self.good_agents()
        n_adv = len(adversaries)
        n_good = len(good_agents)

        if not collisions.any() or n_good==0 or n_adv==0:
            return False  # Early exit if no collisions or agents exist

        # Step 1. Initialization
        # Prepare tensors to track unique captures within this timestep
        adv_pos = torch.stack([a.state.pos for a in adversaries], dim=1)  # (batch, n_adv, 2)
        good_pos = torch.stack([a.state.pos for a in good_agents], dim=1)  # (batch, n_good, 2)

        # Flag to mark which evaders have been caught in this step to prevent double-counting
        good_agent_caught_this_step = torch.zeros(
            (self.world.batch_dim, n_good), device=self.world.device, dtype=torch.bool
        )
        # Flag to mark which pursuers made a successful catch in this step to prevent double-counting
        adv_caught_this_step = torch.zeros(
            (self.world.batch_dim, n_adv), device=self.world.device, dtype=torch.bool
        )


        # Step 2. Resolve collisions and assign unique captures
        # Iterate through each pursuer to determine if it made a unique catch
        for i, adv in enumerate(adversaries):
            # Get all evaders this pursuer is colliding with in each batch environment
            collided_with_adv_i = collisions[:, i, :]

            # Skip if this pursuer has no collisions in any batch environment
            if not collided_with_adv_i.any():
                continue

            # Calculate distances from this pursuer to all evaders
            distances_to_good = torch.linalg.vector_norm(adv.state.pos.unsqueeze(1) - good_pos, dim=-1)

            # Mask non-colliding evaders by setting their distance to +infinity
            # This ensures that argmin will only select from valid collision targets
            distances_to_good[~collided_with_adv_i] = float('inf')

            # Find the index of the single closest evader this pursuer is colliding with
            closest_good_idx = torch.argmin(distances_to_good, dim=1)

            # Identify the batch environments where a valid (non-infinite distance) catch occurred
            adv_made_a_catch_mask = torch.isfinite(
                distances_to_good.gather(1, closest_good_idx.unsqueeze(1)).squeeze(1))

            batch_indices = torch.where(adv_made_a_catch_mask)[0]
            if batch_indices.numel() == 0:
                continue

            # For these potential catches, check if the target evader has already been claimed
            good_indices_to_check = closest_good_idx[batch_indices]
            already_caught_mask = good_agent_caught_this_step[batch_indices, good_indices_to_check]

            # A catch is successful only if the evader has not been claimed yet
            successful_catch_batch_indices = batch_indices[~already_caught_mask]

            if successful_catch_batch_indices.numel() > 0:
                # For successful unique catches, set the flags for both the pursuer and the evader
                final_good_indices = closest_good_idx[successful_catch_batch_indices]
                adv_caught_this_step[successful_catch_batch_indices, i] = True
                good_agent_caught_this_step[successful_catch_batch_indices, final_good_indices] = True


        # Step 3. Apply rewards and update states based on unique captures
        # If no unique captures were made by any pursuer, exit
        if not (adv_caught_this_step.any() or good_agent_caught_this_step.any()):
            return False

        # Apply rewards to pursuers that made a successful catch
        for i, adv in enumerate(adversaries):
            new_collisions = adv_caught_this_step[:, i] & (~adv.heaven_reward_given)
            if new_collisions.any():
                # Trigger Early Termination, move the agent to the heaven location and make it inactive
                if not hasattr(adv, 'rew'):
                    adv.rew = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.float32)
                time_bonus = self.time_reward_coef * (self.max_steps - self.current_step)
                adv.rew[new_collisions] += (self.adv_heaven_reward + time_bonus[new_collisions])
                adv.heaven_reward_given[new_collisions] = True

                adv.active[new_collisions] = False
                adv.state.pos[new_collisions] = self.heaven_position
                adv.state.vel[new_collisions] = 0.0

        # Apply penalties to evaders that were caught
        for j, agent in enumerate(good_agents):
            new_collisions = good_agent_caught_this_step[:, j] & (~agent.heaven_reward_given)
            if new_collisions.any():
                if not hasattr(agent, 'rew'):
                    agent.rew = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.float32)
                agent.rew[new_collisions] += self.agent_heaven_penalty
                agent.heaven_reward_given[new_collisions] = True

                agent.active[new_collisions] = False
                agent.state.pos[new_collisions] = self.heaven_position
                agent.state.vel[new_collisions] = 0.0

        return True



    # return all agents that are not adversaries
    def good_agents(self):
        return [agent for agent in self.world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self):
        return [agent for agent in self.world.agents if agent.adversary]

    def reward(self, agent: Agent):
        """
        Calculates the reward for a given agent.

        This function serves as the reward computation function for the entire
        environment. To ensure that world-level calculations are performed only
        once per step, it uses a common idiom where the logic is executed only
        when the first agent in the list (`is_first`) calls the function.

        The reward structure is hierarchical:
        1.  Sparse Event Rewards: A large positive reward for a pursuer making a
            capture and a large negative penalty for an evader being caught. This is
            handled within `process_collisions`.
        2.  Dense Team-Progress Reward: A team-level reward for the pursuers based
            on the reduction in the Sinkhorn optimal transport cost. This encourages
            coordinated closing-in on the evaders. This reward is only given if no
            capture event occurred in the same step.
        3.  Individual Shaping Rewards: Small, per-agent rewards/penalties to
            encourage desirable behaviors (e.g., closing speed) and discourage
            undesirable ones (e.g., hitting walls, oscillating).

        The function returns the final reward for the specific `agent` passed as an
        argument, respecting the reward sharing settings (`agents_share_rew` and
        `adversaries_share_rew`).

        Args:
            agent: The agent for which to calculate the reward.

        Returns:
            The calculated reward for the agent.
        """
        is_first = agent == self.world.agents[0]

        if is_first:
            # This block is executed only once per step by the first agent to perform all world-level reward calculations
            self.current_step += 1

            # Step 1. Initialize reward accumulators
            # Reset the step reward for all agents to zero before recalculating
            for a in self.world.agents:
                if not hasattr(a, 'rew'):
                    a.rew = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.float32)
                a.rew[:] = 0.0

            # Step 2. Process captures and apply sparse event rewards
            # This function handles the logic of unique captures and applies
            # the one-time heaven rewards/penalties directly to the `a.rew` attribute
            collisions = self.detect_collisions()
            capture_just_happened = self.process_collisions(collisions)

            # Step 3. Calculate dense team-level progress reward (Sinkhorn)
            self.current_team_progress_reward = torch.zeros(self.world.batch_dim, device=self.world.device,
                                                            dtype=torch.float32)
            adversaries = self.adversaries()
            good_agents = self.good_agents()
            if good_agents and adversaries:
                adv_pos = torch.stack([a.state.pos for a in adversaries], dim=1)
                good_pos = torch.stack([ga.state.pos for ga in good_agents], dim=1)
                good_active_mask = torch.stack([ga.active for ga in good_agents], dim=1)

                if good_active_mask.any():
                    dist_matrix = torch.cdist(adv_pos, good_pos, p=2)
                    current_assignment_cost = self._calculate_assignment_cost_sinkhorn(
                        dist_matrix, good_active_mask, epsilon=0.1, n_iters=10
                    )

                    # The progress reward is only given if no capture occurred this step
                    # to prevent rewarding both the process and the result simultaneously
                    assignment_progress_reward = torch.zeros_like(self.prev_assignment_cost)

                    # This logic handles both non-vectorized (returns bool) and vectorized (returns tensor) environments
                    if isinstance(capture_just_happened, bool):
                        if not capture_just_happened:
                            # For a single environment, calculate progress reward if no capture
                            assignment_progress_reward = self.prev_assignment_cost - current_assignment_cost
                    else:
                        no_capture_mask = ~capture_just_happened
                        if torch.any(no_capture_mask):
                            # For vectorized envs, calculate reward only for those without a capture
                            assignment_progress_reward[no_capture_mask] = (
                                    self.prev_assignment_cost[no_capture_mask] - current_assignment_cost[
                                no_capture_mask]
                            )

                    self.current_team_progress_reward = self.distance_reward_coef * assignment_progress_reward
                    self.prev_assignment_cost = current_assignment_cost.detach()

            # Step 4. Calculate and add individual shaping rewards
            # Compute the rule-based actions for the evaders for this step
            self.compute_good_agent_velocities()
            # Calculate individual rewards for all active agents
            for a in self.world.agents:
                if a.active.any():
                    active_mask = a.active
                    if a.adversary:
                        normal_rew = self.adversary_reward(a)
                    else:
                        normal_rew = self.agent_reward(a)
                    a.rew += torch.where(active_mask, normal_rew, 0.0)

            # Step 5. Aggregate team rewards for sharing purposes
            self.agents_rew = torch.stack([a.rew for a in self.good_agents()], dim=-1).sum(-1)
            self.adversary_rew = (torch.stack([a.rew for a in self.adversaries()], dim=-1).sum(-1)
                                  + self.current_team_progress_reward)


        if agent.adversary:
            if self.adversaries_share_rew:
                return self.adversary_rew
            else:
                return agent.rew
        else:
            if self.agents_share_rew:
                return self.agents_rew
            else:
                return agent.rew

    def agent_reward(self, agent: Agent):
        rew = torch.zeros(
            self.world.batch_dim, device=self.world.device, dtype=torch.float32
        )
        return rew

    def adversary_reward(self, agent: Agent):
        """
        Calculates the individual shaping reward for a single pursuer agent.

        This reward is designed to guide the learning process by providing dense
        feedback, independent of the sparse capture events.
        It is only calculated for active pursuers and only if there are active
        evaders remaining in the environment.

        The shaping reward consists of several components:
        - Closing Speed Reward: Positive reward for moving towards the nearest evader.
        - Time Pressure Penalty: A small penalty that increases as the episode progresses.
        - Boundary Penalty: A penalty for getting too close to the world boundaries.
        - Collision Penalties: Penalties for colliding with teammates (other pursuers) or obstacles (landmarks).
        - Oscillation Penalty: A penalty for rapid, jerky movements to encourage smoother paths.

        Args:
            agent: The pursuer agent for which to calculate the reward.

        Returns:
            A tensor containing the calculated shaping reward for the agent across the batch of environments.
            The reward is zero for any environment in the batch where this agent is inactive.
        """
        rew = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.float32)

        # Early exit if the agent is inactive in all batch environments
        if (~agent.active).all():
            return rew

        active_good_agents = [ga for ga in self.good_agents() if ga.active.any()]

        # Proceed with reward calculation only if there are any active evaders left in the environment
        if active_good_agents:
            # only calculate reward for active batch agents
            active_mask = agent.active

            distances_to_active_good = torch.stack(
                [torch.linalg.vector_norm(ga.state.pos - agent.state.pos, dim=-1)
                 for ga in active_good_agents],
                dim=-1
            )

            # Find the minimum distance and the position of the corresponding evader
            current_min_distances, min_dist_indices = torch.min(distances_to_active_good, dim=-1)
            min_dist_this_step = current_min_distances

            active_good_pos_stacked = torch.stack([ga.state.pos for ga in active_good_agents], dim=1)
            batch_indices = torch.arange(self.world.batch_dim, device=self.world.device)
            closest_good_agent_pos_this_step = active_good_pos_stacked[
                batch_indices, min_dist_indices]

            # Closing Speed Reward
            # Reward for velocity component in the direction of the closest evader
            direction_to_closest = (closest_good_agent_pos_this_step - agent.state.pos) / (
                    min_dist_this_step.unsqueeze(-1) + 1e-6)
            closing_speed = torch.sum(agent.state.vel * direction_to_closest, dim=-1)
            rew += self.closing_speed_reward_coef * closing_speed

            # Time Pressure Penalty
            # Penalize the agent more as time goes on to encourage faster captures.
            time_ratio = self.current_step.float() / self.max_steps
            rew += -self.time_pressure_coef * (time_ratio)

            # 4. Boundary Penalty
            # Penalize for moving too close to or beyond the world boundaries
            pos = agent.state.pos
            safe_margin_boundary = agent.shape.radius
            penetration_right = torch.clamp(pos[..., 0] - (self.bound - safe_margin_boundary), min=0)
            penetration_left = torch.clamp((-self.bound + safe_margin_boundary) - pos[..., 0], min=0)
            penalty_x_sq = penetration_right ** 2 + penetration_left ** 2
            penetration_up = torch.clamp(pos[..., 1] - (self.bound - safe_margin_boundary), min=0)
            penetration_down = torch.clamp((-self.bound + safe_margin_boundary) - pos[..., 1], min=0)
            penalty_y_sq = penetration_up ** 2 + penetration_down ** 2
            rew += -self.boundary_penalty_coef * (penalty_x_sq + penalty_y_sq)

            # 5. Teammate Collision Penalty
            # Penalize for getting too close to other active pursuers to encourage spreading out
            for other_adv in self.adversaries():
                if other_adv is agent or (~other_adv.active).all():
                    continue
                dist_to_teammate = torch.linalg.vector_norm(agent.state.pos - other_adv.state.pos, dim=-1)
                safe_dist_team = (agent.shape.radius + other_adv.shape.radius) * 1.5
                penetration_team = torch.clamp(safe_dist_team - dist_to_teammate, min=0)
                rew += -self.collision_penalty_coef * (penetration_team ** 2) * other_adv.active

            # 6. Obstacle Penalty
            # Penalize for getting too close to landmarks
            for landmark in self.world.landmarks:
                dist_to_landmark = torch.linalg.vector_norm(agent.state.pos - landmark.state.pos, dim=-1)
                safe_dist_landmark = (agent.shape.radius + landmark.shape.radius) * 1.2
                penetration_landmark = torch.clamp(safe_dist_landmark - dist_to_landmark, min=0)
                rew += -self.obstacle_penalty_coef * (penetration_landmark ** 2)

            # 7. Oscillation Penalty
            # Penalize for rapid changes in velocity to promote smoother movement
            vel_change_norm = torch.linalg.vector_norm(agent.state.vel - agent.prev_vel, dim=-1)
            rew += -self.oscillation_penalty_coef * torch.clamp(vel_change_norm - self.oscillation_vel_threshold, min=0)
            agent.prev_vel = torch.where(active_mask.unsqueeze(-1), agent.state.vel.clone(), agent.prev_vel)

            return rew * active_mask  # Final reward is masked to ensure inactive agents get zero reward
        else:
            return rew


    def compute_good_agent_velocities(self):
        """
        Computes and sets the velocities for the non-learning evader agents.

        This function implements a rule-based hybrid policy for the evaders ("good agents").
        The policy dynamically switches between two modes based on the perceived threat level:

        1.  APF (Artificial Potential Field) Evasion Mode: This is an emergency
            mode triggered by immediate threats, specifically a pursuer being within
            a critical distance or the agent being too close to a world boundary.
            In this mode, the agent calculates a net repulsive force from all
            entities (pursuers, obstacles, teammates, boundaries) and sets its
            target velocity to move at maximum speed in the direction of this
            force to escape danger.

        2.  Guided Levy Flight Exploration Mode: This is the default mode when
            not under immediate threat. Agents follow a randomized long-term
            velocity target generated by a Levy flight process. This provides
            unpredictable, long-range movement. This base random movement is
            then perturbed by gentle guidance forces from non-threatening
            entities (obstacles and teammates) to encourage local collision
            avoidance and team dispersion without overriding the primary
            exploration behavior.

        After determining the target velocity based on the current mode, the function
        applies kinematic constraints, limiting the acceleration to ensure smooth
        and physically plausible movement, and finally clips the velocity to the
        agent's maximum speed.
        """

        good_agents = self.good_agents()
        if not good_agents:
            return

        device = self.world.device
        batch_dim = self.world.batch_dim
        n_good = len(good_agents)

        # Initialize tensors for the evader's Levy flight policy if they don't exist
        if not hasattr(self, 'target_velocity'):
            self.target_velocity = torch.zeros_like(torch.stack([agent.state.pos for agent in good_agents], dim=1))
            self.direction_timer = torch.zeros(batch_dim, len(good_agents), device=device, dtype=torch.int32)

        # Gather current states in a vectorized manner
        good_pos = torch.stack([agent.state.pos for agent in good_agents], dim=1)
        good_vel = torch.stack([agent.state.vel for agent in good_agents], dim=1)
        good_active = torch.stack([agent.active for agent in good_agents], dim=1)

        # Early exit if no evaders are active in any batch environment
        if not good_active.any():
            return

        adv_pos = torch.stack([agent.state.pos for agent in self.adversaries()], dim=1)
        adv_active = torch.stack([agent.active for agent in self.adversaries()], dim=1)

        # Initialize potential forces from all sources
        # APF_force
        force_from_adv = torch.zeros_like(good_pos)
        force_from_boundary = torch.zeros_like(good_pos)
        # gentle guide force
        force_from_landmark = torch.zeros_like(good_pos)
        force_from_teammate = torch.zeros_like(good_pos)


        # Repulsive force from pursuers
        delta_pos_adv = good_pos.unsqueeze(2) - adv_pos.unsqueeze(1)
        dist_adv = torch.linalg.vector_norm(delta_pos_adv, dim=-1)
        safe_dist_adv = good_agents[0].shape.radius + self.adversaries()[0].shape.radius + self.adv_safe_dist_extra
        force_magnitude_adv = torch.where(dist_adv < safe_dist_adv,
                                          self.adv_repulsion_coef_close / (dist_adv + 1e-6),
                                          self.adv_repulsion_coef_far / (dist_adv + 1e-6))
        force_dir_adv = delta_pos_adv / (dist_adv.unsqueeze(-1) + 1e-6)
        adv_active_mask = adv_active.unsqueeze(1)
        force_from_adv = (force_magnitude_adv.unsqueeze(-1) * force_dir_adv * adv_active_mask.unsqueeze(-1)).sum(dim=2)

        # Repulsive force from landmarks (obstacles)
        if self.world.landmarks:
            landmark_pos = torch.stack([landmark.state.pos for landmark in self.world.landmarks], dim=1)
            delta_pos_land = good_pos.unsqueeze(2) - landmark_pos.unsqueeze(1)
            dist_land = torch.linalg.vector_norm(delta_pos_land, dim=-1)

            # The force from landmarks has a limited range to act as a local guidance field
            influence_radius_land = (good_agents[0].shape.radius + self.world.landmarks[
                0].shape.radius + self.landmark_safe_dist_extra) * 1.0
            base_force_magnitude = self.landmark_repulsion_coef / (dist_land + 1e-6)

            # Force is only applied if the landmark is within the influence radius
            force_magnitude_land = torch.where(
                dist_land < influence_radius_land,
                base_force_magnitude,
                torch.zeros_like(dist_land)
            )
            force_dir_land = delta_pos_land / (dist_land.unsqueeze(-1) + 1e-6)
            force_from_landmark = (force_magnitude_land.unsqueeze(-1) * force_dir_land).sum(dim=2)

        # Repulsive force from boundaries
        pos_x, pos_y = good_pos[..., 0], good_pos[..., 1]
        safe_dist_bound = good_agents[0].shape.radius * self.boundary_safe_dist_factor

        dist_to_x_boundary = self.bound - torch.abs(pos_x)
        force_mag_x = torch.where(
            dist_to_x_boundary < safe_dist_bound,
            self.boundary_repulsion_coef * (safe_dist_bound - dist_to_x_boundary) / (dist_to_x_boundary + 1e-6),
            torch.zeros_like(dist_to_x_boundary)
        )
        boundary_force_x = -force_mag_x * torch.sign(pos_x)

        dist_to_y_boundary = self.bound - torch.abs(pos_y)
        force_mag_y = torch.where(
            dist_to_y_boundary < safe_dist_bound,
            self.boundary_repulsion_coef * (safe_dist_bound - dist_to_y_boundary) / (dist_to_y_boundary + 1e-6),
            torch.zeros_like(dist_to_y_boundary)
        )
        boundary_force_y = -force_mag_y * torch.sign(pos_y)

        force_from_boundary = torch.stack([boundary_force_x, boundary_force_y], dim=-1)


        # Repulsive force from teammates (other evaders) to encourage dispersion
        delta_pos_team = good_pos.unsqueeze(2) - good_pos.unsqueeze(1)
        dist_team = torch.linalg.vector_norm(delta_pos_team, dim=-1)
        force_magnitude_team = torch.where(dist_team < self.teammate_safe_dist,
                                           self.teammate_repulsion_coef / (dist_team + 1e-6),
                                           torch.zeros_like(dist_team))
        force_dir_team = delta_pos_team / (dist_team.unsqueeze(-1) + 1e-6)
        eye_mask = torch.eye(len(good_agents), device=device, dtype=torch.bool).unsqueeze(0)
        force_magnitude_team.masked_fill_(eye_mask, 0.0)
        teammate_active_mask = good_active.unsqueeze(1)
        force_from_teammate = (force_magnitude_team.unsqueeze(-1) * force_dir_team * teammate_active_mask.unsqueeze(-1)).sum(dim=2)

        # Ensure that forces are only applied to and by active agents
        good_active_3d = good_active.unsqueeze(-1)
        force_from_adv *= good_active_3d
        force_from_landmark *= good_active_3d
        force_from_boundary *= good_active_3d
        force_from_teammate *= good_active_3d

        # Determine APF evasion mode trigger
        # The trigger is based on distance to immediate threats (pursuers and boundaries)

        # Calculate distance to the nearest active pursuer.
        dist_to_active_advs = dist_adv.clone()
        dist_to_active_advs.masked_fill_(~adv_active.unsqueeze(1), float('inf'))
        min_dist_to_adv, _ = torch.min(dist_to_active_advs, dim=-1)  # shape: (batch, n_good)

        # Check for threats from pursuers or boundaries
        threatened_by_adv = (min_dist_to_adv < self.apf_trigger_distance_adv)
        threatened_by_boundary = (dist_to_x_boundary < safe_dist_bound) | (dist_to_y_boundary < safe_dist_bound)

        # An evader enters APF mode if it is threatened and is currently active
        mask_apf = (threatened_by_adv | threatened_by_boundary) & good_active


        # Determine target velocities based on mode

        # Default target is the long-term Levy flight velocity
        target_velocities = self.target_velocity.clone()

        # For agents in APF Evasion Mode
        if mask_apf.any():
            # The target velocity is to escape at max speed in the direction of the net repulsive force
            # All forces are summed to find the safest escape vector
            total_force_apf = force_from_adv + force_from_boundary + force_from_landmark + force_from_teammate
            force_magnitude = torch.linalg.vector_norm(total_force_apf[mask_apf], dim=-1, keepdim=True)
            apf_force_direction = total_force_apf[mask_apf] / (force_magnitude + 1e-6)
            target_velocities[mask_apf] = apf_force_direction * good_agents[0].max_speed

        # For agents in Guided Exploration Mode
        mask_explore = (~mask_apf) & good_active
        if mask_explore.any():
            # Update the Levy flight timer
            self.direction_timer[mask_explore] += 1
            change_direction = (self.direction_timer >= self.direction_change_interval) & mask_explore

            # If the timer is up, generate a new random long-term velocity target
            if change_direction.any():
                num_to_change = change_direction.sum().item()
                step_lengths = self._sample_levy_steps(num_to_change, device)
                theta = torch.rand(num_to_change, generator=self.rng_levy, device=device) * 2 * torch.pi

                self.target_velocity[..., 0][change_direction] = step_lengths * torch.cos(theta)
                self.target_velocity[..., 1][change_direction] = step_lengths * torch.sin(theta)
                self.direction_timer[change_direction] = 0

            # The final exploration velocity is the base random target plus gentle guidance
            base_random_vel = self.target_velocity[mask_explore]
            guidance_force = force_from_landmark[mask_explore] + force_from_teammate[mask_explore]
            final_explore_vel = base_random_vel + guidance_force * self.guidance_force_coef
            target_velocities[mask_explore] = final_explore_vel



        # Apply kinematic constraints on good (policy/non-learning) agents

        # Calculate the desired change in velocity to steer towards the target.
        delta_v = target_velocities - good_vel

        # Limit the change to the maximum allowed acceleration
        max_accel_val = self.max_acceleration * good_agents[0].max_speed
        clamped_acceleration = torch.clamp(delta_v, -max_accel_val, max_accel_val)
        new_vel = good_vel + clamped_acceleration

        # Ensure the final velocity does not exceed the agent's maximum speed
        new_vel_norm = torch.linalg.vector_norm(new_vel, dim=-1, keepdim=True)
        max_speed_val = good_agents[0].max_speed
        final_vel = torch.where(
            new_vel_norm > max_speed_val,
            new_vel * (max_speed_val / (new_vel_norm + 1e-6)),
            new_vel
        )


        # Apply final velocities to active agents
        for i, agent in enumerate(good_agents):
            agent.state.vel = torch.where(
                good_active[:, i].unsqueeze(-1),
                final_vel[:, i],
                agent.state.vel
            )

    def _sample_levy_steps(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Generates step lengths from a Levy-like distribution (simplified using power law).
        u = (0, 1]
        step_length = scale * u ^ (-1/alpha)
        """

        u = torch.rand(num_samples, generator=self.rng_levy, device=device)
        # Ensure u is not zero to avoid division by zero or inf if levy_alpha is small.
        u = torch.clamp(u, min=1e-9)

        return self.levy_scale * (u ** (-1.0 / self.levy_alpha))

    def observation(self, agent: Agent):
        # if inactive (in heaven), return all zeros observation
        obs_dim = self._get_obs_dim(agent)
        final_obs = torch.zeros(
            (self.world.batch_dim, obs_dim),
            device=self.world.device,
            dtype=torch.float32,
        )
        if (~agent.active).any():
            agent.state.pos[~agent.active] = self.heaven_position
            agent.state.vel[~agent.active] = 0.0

        if (~agent.active).all():
            return final_obs

        if agent.active.any():

            # get positions of all entities in this agent's reference frame
            entity_pos = []
            for entity in self.world.landmarks:
                entity_pos.append(entity.state.pos - agent.state.pos)

            other_pos = []
            other_vel = []
            for other in self.world.agents:
                if other is agent:
                    continue

                # only add actual pos and vel to other agents when they active
                # otherwise, set them to 0
                zero_pos = torch.zeros_like(other.state.pos)
                zero_vel = torch.zeros_like(other.state.vel)

                if agent.adversary and not other.adversary:
                    # Check whether "other" is active for each environment instance separately
                    # Create a active mask tensor with a shape consistent with batch_dim
                    active_mask = other.active.unsqueeze(-1).expand_as(other.state.pos)
                    # only add actual pos and vel to other agents when they active
                    # otherwise, set them to 0
                    pos = torch.where(active_mask, other.state.pos - agent.state.pos, zero_pos)
                    vel = torch.where(active_mask, other.state.vel, zero_vel)
                    other_pos.append(pos)
                    other_vel.append(vel)
                elif not agent.adversary and not other.adversary and self.observe_same_team:
                    active_mask = other.active.unsqueeze(-1).expand_as(other.state.pos)
                    pos = torch.where(active_mask, other.state.pos - agent.state.pos, zero_pos)
                    vel = torch.where(active_mask, other.state.vel, zero_vel)
                    other_pos.append(pos)
                    other_vel.append(vel)
                elif not agent.adversary and other.adversary:
                    active_mask = other.active.unsqueeze(-1).expand_as(other.state.pos)
                    pos = torch.where(active_mask, other.state.pos - agent.state.pos, zero_pos)
                    other_pos.append(pos)
                elif agent.adversary and other.adversary and self.observe_same_team:
                    active_mask = other.active.unsqueeze(-1).expand_as(other.state.pos)
                    pos = torch.where(active_mask, other.state.pos - agent.state.pos, zero_pos)
                    other_pos.append(pos)

            normal_obs = torch.cat(
                [
                    *([agent.state.vel] if self.observe_vel else []),
                    *([agent.state.pos] if self.observe_pos else []),
                    *entity_pos,
                    *other_pos,
                    *other_vel,
                ],
                dim=-1,
            )
            final_obs[agent.active] = normal_obs[agent.active]

        assert not torch.isnan(final_obs).any(), f"NaN in observation for agent {agent.name}"

        return final_obs

    def _get_obs_dim(self, agent):
        """get the dimension of observation"""
        # get obs dimension
        dim = 0
        # self vel and pos
        if self.observe_vel:
            dim += self.world.dim_p
        if self.observe_pos:
            dim += self.world.dim_p
        # Landmarks
        dim += len(self.world.landmarks) * self.world.dim_p

        # other agents
        for other in self.world.agents:
            if other is agent:
                continue
            if agent.adversary and not other.adversary:
                dim += self.world.dim_p
                dim += self.world.dim_p
            elif not agent.adversary and not other.adversary and self.observe_same_team:
                dim += self.world.dim_p
                dim += self.world.dim_p
            elif not agent.adversary and other.adversary:
                dim += self.world.dim_p
            elif agent.adversary and other.adversary and self.observe_same_team:
                dim += self.world.dim_p

        return dim

    def done(self):
        # Get the active status of all good agents and adversaries
        good_active = torch.stack([agent.active for agent in self.good_agents()], dim=0)
        adv_active = torch.stack([agent.active for agent in self.adversaries()], dim=0)

        # In each environment, all good agents or all adversaries are inactive
        all_good_inactive = ~good_active.any(dim=0)
        all_adv_inactive = ~adv_active.any(dim=0)

        # done status in each vectorize environment
        return all_good_inactive | all_adv_inactive

    def extra_render(self, env_index: int = 0):
        from vmas.simulator import rendering

        geoms = []

        # Draw Heaven at self.heaven_position
        heaven_geom = rendering.make_circle(self.heaven_size)
        xform = rendering.Transform()
        heaven_geom.add_attr(xform)
        xform.set_translation(self.heaven_position[0].item(), self.heaven_position[1].item())
        heaven_geom.set_color(0.9, 0.9, 0.2)  # Golden Heaven
        geoms.append(heaven_geom)

        # Perimeter
        for i in range(4):
            geom = Line(
                length=2
                       * ((self.bound - self.adversary_radius) + self.adversary_radius * 2)
            ).get_geometry()
            xform = rendering.Transform()
            geom.add_attr(xform)

            xform.set_translation(
                (
                    0.0
                    if i % 2
                    else (
                        self.bound + self.adversary_radius
                        if i == 0
                        else -self.bound - self.adversary_radius
                    )
                ),
                (
                    0.0
                    if not i % 2
                    else (
                        self.bound + self.adversary_radius
                        if i == 1
                        else -self.bound - self.adversary_radius
                    )
                ),
            )
            xform.set_rotation(torch.pi / 2 if not i % 2 else 0.0)
            color = Color.BLACK.value
            if isinstance(color, torch.Tensor) and len(color.shape) > 1:
                color = color[env_index]
            geom.set_color(*color)
            geoms.append(geom)
        return geoms

    def _calculate_assignment_cost_sinkhorn(
            self,
            dist_matrix: torch.Tensor,
            good_active_mask: torch.Tensor,
            epsilon: float = 0.1,
            n_iters: int = 10
    ) -> torch.Tensor:
        """
        Calculates the team-level optimal assignment cost using the Sinkhorn algorithm.

        This function models the problem of assigning pursuers to evaders as a
        regularized optimal transport problem. The returned "cost" represents the
        minimum possible total distance if each pursuer were optimally and softly
        assigned to a unique active evader.

        This cost serves as a potential function for a dense team-level progress
        reward. A decrease in this cost from one step to the next (`prev_cost -
        current_cost`) indicates that the pursuer team is effectively coordinating
        and closing in on the evader team, thus providing a strong learning signal
        for cooperation.

        Args:
            dist_matrix: Tensor of shape (batch, n_pursuers, n_evaders) with
                         pair-wise distances.
            good_active_mask: A boolean tensor of shape (batch, n_evaders)
                              indicating which evaders are active.
            epsilon: Sinkhorn regularization strength. Smaller values approach a
                     hard (non-differentiable) assignment.
            n_iters: Number of Sinkhorn-Knopp iterations to perform.

        Returns:
            A tensor of shape (batch,) representing the total optimal assignment
            cost for each environment in the batch.
        """

        batch_dim, n_adv, n_good = dist_matrix.shape

        # Early exit if either team is empty
        if n_good == 0 or n_adv == 0:
            return torch.zeros(batch_dim, device=dist_matrix.device)

        # Set the cost of assigning to an inactive evader to zero. This ensures
        # that the total cost decreases correctly when an evader is captured
        active_mask_3d = good_active_mask.unsqueeze(1).expand_as(dist_matrix)
        cost_matrix = torch.where(active_mask_3d, dist_matrix, 0.0)


        # The Sinkhorn algorithm requires a square matrix. We pad the cost matrix
        # with zeros to handle cases where n_adv != n_good
        max_dim = max(n_adv, n_good)
        padded_cost = torch.zeros((batch_dim, max_dim, max_dim), device=dist_matrix.device)
        padded_cost[:, :n_adv, :n_good] = cost_matrix

        # The Gibbs kernel (or transport kernel) K is derived from the cost matrix
        K = torch.exp(-padded_cost / epsilon)

        # Initialize scaling vectors, v is iteratively updated
        v = torch.ones(batch_dim, max_dim, 1, device=dist_matrix.device)

        # Perform the Sinkhorn-Knopp iterations to approximate the optimal transport plan
        for _ in range(n_iters):
            u = 1.0 / (torch.bmm(K, v) + 1e-8)  # u = 1 / (K @ v)
            v = 1.0 / (torch.bmm(K.transpose(1, 2), u) + 1e-8)  # v = 1 / (K.T @ u)

        # Reconstruct the optimal transport plan matrix P from the scaling vectors.
        # P = diag(u) @ K @ diag(v)
        u, v = u.squeeze(-1), v.squeeze(-1)
        P = u.unsqueeze(-1) * K * v.unsqueeze(1)

        # The total transport cost is the Frobenius inner product of the transport
        # plan P and the original padded cost matrix
        total_cost = torch.sum(P * padded_cost, dim=[1, 2])

        return total_cost

if __name__ == "__main__":
    from vmas.interactive_rendering import render_interactively
    render_interactively(__file__, control_two_agents=True)