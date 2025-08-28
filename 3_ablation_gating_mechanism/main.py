import argparse

import torch

from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.experiment.callback import Callback
from benchmarl.algorithms import MappoConfig
from benchmarl.models.mlp import MlpConfig
from benchmarl.models.pimore import PIMoREConfig
from benchmarl.models.maac import MAACConfig
from benchmarl.models.mapoca import MAPOCAConfig
from benchmarl.models.deepsets import DeepsetsConfig

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class TagAgentScript(Callback):
    def __init__(self):
        super().__init__()
    def on_setup(self):
        """
        frozen good agents' policy network
        """
        print("TagAgentScript: Freezing agent group training")
        print(f"Before: {self.experiment.train_group_map}")
        if "agent" in self.experiment.train_group_map:
            del self.experiment.train_group_map["agent"]
        print(f"After: {self.experiment.train_group_map}")


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--project_name', type=str, default='TNNLS_DS_QG_MoE')
    parser.add_argument('--algorithm', type=str, default='mappo', choices=['pimore', 'pimappo', 'maac', 'mapoca', 'mappo'])
    parser.add_argument('--save_folder', type=str, default=None)
    parser.add_argument('--load_folder', type=str, default=None)
    
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_adversaries', type=int, default=3)
    parser.add_argument('--num_good_agents', type=int, default=2)

    args_temp = parser.parse_known_args()[0]

    if (args_temp.num_adversaries == 3) and (args_temp.num_good_agents == 2):
        parser.add_argument('--max_episode_len', type=int, default=100)
    elif (args_temp.num_adversaries == 4) and (args_temp.num_good_agents == 3):
        parser.add_argument('--max_episode_len', type=int, default=150)
    elif (args_temp.num_adversaries == 5) and (args_temp.num_good_agents == 4):
        parser.add_argument('--max_episode_len', type=int, default=200)

    parser.add_argument('--policy_num_cells', nargs='+', type=int, default=[256, 128])
    parser.add_argument('--policy_share', action='store_false', help='default is True, if input is False')

    if args_temp.algorithm == 'pimore':
        parser.add_argument('--critic_share_param', action='store_false')
        parser.add_argument('--critic_aggr', type=str, default="sum")
        parser.add_argument('--critic_num_experts', type=int, default=4)
        parser.add_argument('--critic_top_k', type=int, default=2)
        parser.add_argument('--critic_out_features_local_nn', type=int, default=64)
        parser.add_argument('--critic_local_nn_num_cells', nargs='+', type=int,
                            default=[96,64, 56,48, 56,48, 56,48, 56,48])
        parser.add_argument('--critic_global_nn_num_cells', nargs='+', type=int, default=[64, 64])
        parser.add_argument('--critic_moe_type', type=str, default='residual',
                            help='type of moe structure. Options: "vanilla", "vanilla_sparse", "residual"')
        parser.add_argument('--critic_gate_type', type=str, default="quadratic",
                            help='gate type for moe critic. Options: "quadratic", "linear", "mlp", "one_hot", "unbalance_expert"')
    elif args_temp.algorithm == 'maac':
        parser.add_argument('--critic_share_param', action='store_true')
        parser.add_argument('--critic_num_cells_e', nargs='+', type=int, default=[64, 64])
        parser.add_argument('--critic_num_cells_q', nargs='+', type=int, default=[64, 64])
        parser.add_argument('--critic_num_heads', type=int, default=4)
    elif args_temp.algorithm == 'mapoca':
        parser.add_argument('--critic_share_param', action='store_false')
        parser.add_argument('--critic_num_cells_e', nargs='+', type=int, default=[64, 64])
        parser.add_argument('--critic_num_cells_v', nargs='+', type=int, default=[64, 64])
        parser.add_argument('--critic_num_heads', type=int, default=4)
        parser.add_argument('--critic_embedding_dim', type=int, default=64)
    elif args_temp.algorithm == 'mappo':
        parser.add_argument('--critic_share_param', action='store_false')
        parser.add_argument('--critic_num_cells', nargs='+', type=int, default=[128, 128])
    elif args_temp.algorithm == 'pimappo':
        parser.add_argument('--critic_share_param', action='store_false')
        parser.add_argument('--critic_aggr', type=str, default="sum")
        parser.add_argument('--critic_out_features_local_nn', type=int, default=64)
        parser.add_argument('--critic_local_nn_num_cells', nargs='+', type=int, default=[128, 112])
        parser.add_argument('--critic_global_nn_num_cells', nargs='+', type=int, default=[64, 64])

    parser.add_argument('--render', action='store_true', help='Render the environment, default is False,if input is True')
    parser.add_argument('--eval', action='store_false', help='Evaluate the model, default is True, if input is False')

    parser.add_argument('--n_env', type=int, default=800)


    args = parser.parse_args()

    # Loads from "benchmarl/conf/experiment/base_experiment.yaml"
    experiment_config = ExperimentConfig.get_from_yaml()

    # Override parameters
    experiment_config.project_name = args.project_name
    experiment_config.save_folder = args.save_folder
    experiment_config.load_folder = args.load_folder

    vmas_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    train_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    experiment_config.sampling_device = vmas_device
    experiment_config.train_device = train_device


    experiment_config.max_n_frames= 50_000_000
    experiment_config.on_policy_n_envs_per_worker= args.n_env
    experiment_config.on_policy_collected_frames_per_batch = experiment_config.on_policy_n_envs_per_worker * \
        args.max_episode_len
    experiment_config.on_policy_n_minibatch_iters= 15
    experiment_config.on_policy_minibatch_size= 4096
    experiment_config.render = args.render
    experiment_config.share_policy_params = True
    experiment_config.evaluation = args.eval
    experiment_config.evaluation_interval = experiment_config.on_policy_collected_frames_per_batch
    experiment_config.evaluation_episodes = 100
    # experiment_config.checkpoint_interval = 90_000
    experiment_config.loggers = ["wandb"]

    experiment_config.gamma = 0.99
    experiment_config.lr = 0.00005
    experiment_config.clip_grad_norm=True
    experiment_config.clip_grad_val=5
    experiment_config.soft_target_update=True
    experiment_config.polyak_tau=0.005
    experiment_config.hard_target_update_frequency= 5
    experiment_config.exploration_eps_init= 0.8
    experiment_config.exploration_eps_end= 0.01
    experiment_config.exploration_anneal_frames= 1_000_000



    task = VmasTask.UAV_PERSUIT_EVASION_EARLY_TERMINATION.get_from_yaml()

    task.config = {
        "max_steps": args.max_episode_len,
        "num_good_agents": args.num_good_agents,
        "num_adversaries": args.num_adversaries,
        "num_landmarks": 2,
        "adversaries_share_rew": True,
        "bound": 1.3,
        # reward
        ## sparse event reward
        "adv_heaven_reward": 34,
        "time_reward_coef": 0.5,
        ## dense progress reward
        "distance_reward_coef": 0.3,
        ## behavior shaping reward
        "closing_speed_reward_coef": 0.15,
        "boundary_penalty_coef": 0.15,
        "collision_penalty_coef": 0.5,
        "obstacle_penalty_coef": 0.2,
        "oscillation_penalty_coef": 0.01,
        "time_pressure_coef": 0.05,

        "smoothing_distance_coef": 0.7,
        "min_dist_clamp": 1.0,
        "not_catch_penalty": -0.0,

        # target evasion policy
        "max_speed_adv": 0.4,
        "max_speed_good": 0.38,
        "adv_repulsion_coef_close": 0.5,
        "adv_repulsion_coef_far": 0.00,
        "adv_safe_dist_extra": 0.5,  # danger distance to adversaries
        "landmark_repulsion_coef": 0.01,
        "landmark_safe_dist_extra": 0.25,
        "boundary_repulsion_coef": 0.01,
        "boundary_safe_dist_factor": 2.0,
        "teammate_repulsion_coef": 0.01,
        "teammate_safe_dist": 0.25,

        "apf_trigger_distance_adv":0.5,
        "guidance_force_coef":0.3,
        "max_acceleration": 0.2,  # max_acceleration = "max_acceleration"*"max_speed_adv"
        "direction_change_interval": 30,
    }

    # create algo from scratch
    algorithm_config = MappoConfig(
            share_param_critic=args.critic_share_param,
            clip_epsilon=0.2,
            entropy_coef=0.01,
            critic_coef=1,
            loss_critic_type="l2",
            lmbda=0.98,
            scale_mapping="biased_softplus_1.0",
            use_tanh_normal=True,
            minibatch_advantage=False,
        )

    policy_model_config = MlpConfig(
            num_cells=args.policy_num_cells,
            layer_class=torch.nn.Linear,
            activation_class=torch.nn.ReLU,
        )

    if args.algorithm == 'pimore':
        critic_model_config = PIMoREConfig(
            aggr=args.critic_aggr,
            num_experts=args.critic_num_experts,
            top_k=args.critic_top_k,
            gate_type=args.critic_gate_type,
            moe_type=args.critic_moe_type,
            out_features_local_nn=args.critic_out_features_local_nn,
            local_nn_num_cells=args.critic_local_nn_num_cells,
            local_nn_activation_class=torch.nn.ReLU,
            global_nn_num_cells=args.critic_global_nn_num_cells,
            global_nn_activation_class=torch.nn.ReLU,
        )
    elif args.algorithm == 'maac':
        critic_model_config = MAACConfig(
            num_cells_e=args.critic_num_cells_e,  # encoder MLP units
            num_cells_q=args.critic_num_cells_q,  # per-agent Q value MLP units
            num_heads=args.critic_num_heads,  # heads numbers of multi-head attention
            activation_class=torch.nn.ReLU,
        )
    elif args.algorithm == 'mapoca':
        critic_model_config = MAPOCAConfig(
            num_cells_e=args.critic_num_cells_e,  # encoder MLP
            num_cells_v=args.critic_num_cells_v,  # embedding MLP
            num_heads=args.critic_num_heads,
            embedding_dim=args.critic_embedding_dim,
            activation_class=torch.nn.ReLU,
        )
    elif args.algorithm == 'mappo':
        critic_model_config = MlpConfig(
            num_cells=args.critic_num_cells,
            layer_class=torch.nn.Linear,
            activation_class=torch.nn.ReLU,
        )
    elif args.algorithm == 'pimappo':
        critic_model_config = DeepsetsConfig(
            aggr=args.critic_aggr,
            out_features_local_nn=args.critic_out_features_local_nn,
            local_nn_num_cells=args.critic_local_nn_num_cells,
            local_nn_activation_class=torch.nn.ReLU,
            global_nn_num_cells=args.critic_global_nn_num_cells,
            global_nn_activation_class=torch.nn.ReLU,
        )

    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=policy_model_config,
        critic_model_config=critic_model_config,
        seed=args.seed,
        config=experiment_config,
        callbacks=[TagAgentScript()],
    )
    print(f"Callbacks registered: {experiment.callbacks}")
    print("critic_model_config", critic_model_config)
    print("Starting experiment...")
    experiment.run()
    print("Experiment finished.")






