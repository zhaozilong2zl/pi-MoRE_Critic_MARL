Anonymous repository for the manuscript "Tackling Early Termination in Multi-Agent Reinforcement Learning with Permutation Invariant Mixture of Residual Experts Critics" submitted for double-blind review

# Algorithms
1. Permutation-Invariant Mixture-of-Residual-Experts (pi-MoRE), our method
2. Permutation-Invariant MAPPO (pi-MAPPO), our method without the MoE mechanism
3. Multi-Agent POsthumous Credit Assignment (MA-POCA, For more details, please refer to: https://arxiv.org/abs/2111.05992)
4. Multi-Actor-Attention-Critic (MAAC, For more details, please refer to: https://arxiv.org/abs/1810.02912)
5. MAPPO (For more details, please refer to: https://arxiv.org/abs/2103.01955)

Our implementations of the above algorithms are stored in repository directory ```./dependencies/BenchMARL-DS_MoE/benchmarl/models/moe.py```, ```./dependencies/benchmarl/models/mapoca.py``` and ```./dependencies/benchmarl/models/maac.py```.    

key parameters for our pi-MoRE algorithms:

1. ```moe_type```: moe structure of algorithms, can be set to 'residual', 'vanilla', 'vanilla_sparse'
2. ```gate_type```: gate mechanism of moe, can be set to 'quadratic', 'linear', 'mlp', 'one_hot'
3. ```num_experts```: hyperparameter of all moe structure.
4. ```num_cells```: list specifying the hidden layer sizes of expert networks (each expert is a two-hidden-layer MLP). if moe_type is set to 'residual', the first two numbers correspond to the shared expert, while the remaining ones correspond to the sparse experts.
5. ```top_k```: hyperparameter of 'residual' and 'vanilla_sparse' structure

# Environments
The multi-UAV multi-target interception scenario used in our experiments is stored in repository directory ```./dependencies/VectorizedMultiAgentSimulator-inactive_logic/vmas/scenarios/mpe/uav_pursuit_evasion_early_termination.py```.


# Code organization
The code is organized as follows:
```
1_main_results
|--main.py
|--run_main.sh
2_ablation_residual_structure
|--main_ablation_stru.py
|--run_ablation_stru.sh
3_ablation_gating_mechanism
|--main_ablation_gate.py
|--run_ablation_gate.sh
4_ablation_top_k
|--main_ablation_topk.py
|--run_ablation_topk.sh
dependencies
|--BenchMARL-DS_MoE
    |--benchmarl
        |--models
            |--pimore.py
            |--pimappo.py
            |--maac.py
            |--mapoca.py
            |--mappo.py
|--VectorizedMultiAgentSimulator-inactive_logic
    |--vmas
        |--scenarios
            |--mpe
                |--uav_pursuit_evasion_early_termination.py
diagnostic_experiment
|--main_diagnostic_experiment.py
|--partial_results
```



# Recommend requirements
We tested the algorithms on the following setup:
## OS Requirements   
Linux: Ubuntu 20.04.1 (5.15.0-139-generic)   
Driver Version: 535.216.01   
CUDA Version: 12.2   
GPU Device: NVIDIA GeForce RTX 4060
## Python Dependencies
Python: 3.9.18   
For other dependent packages, please refer to the following environment setup section.

# Environment setup
## 1. Create and activate conda environment
We highly recommend using Anaconda or Miniconda to manage the environment.

First, create a new Conda environment. We name it `pimore` as below.

```bash
conda create -n pimore python=3.9.18
conda activate pimore
```

## 2. Install all dependencies
After activating the environment, run our unified installation script. This script will install all required packages. All our modifications to third-party libraries are included locally in the `dependencies/` folder to ensure full anonymity and reproducibility.

Execute the following command from the project's root directory:

```bash
bash install.sh
```

The script will sequentially install:
1.  Our modified version of `VectorizedMultiAgentSimulator`.
2.  Our modified version of `BenchMARL`.
3.  `torchrl`, `wandb`, `tensordict`, `numpy`, `matplotlib`, `seaborn` and `pandas` with specific versions.


## 3. Logging with Weights & Biases (W&B, wandb)

We use `wandb` for experiment logging and visualization by default.

Before running any training scripts, please set up your W&B account. You can either log in to an existing account or create a new one for free.

# Running Experiments

This section provides instructions on how to train the models presented in our paper.

## Training

To train an agent with a specific algorithm in default parameters, you can use our main script as below:

```bash
python main.py --algorithm [ALGO_NAME] --seed [SEED] --num_adversaries [NUM_ADV] --num_good_agents [NUM_GOOD]
```

**Arguments**:
*   `--algo [ALGO_NAME]`: Specifies the algorithm to use. (`pi-MoRE`, `pi-MAPPO`, `MAAC`, `MA-POCA`, `MAPPO`)
*   `--seed [SEED]`: Specifies the training seed.
*   `--num_adversaries [NUM_ADV]`: Specifies the number of pursuers.
*   `--num_good_agents [NUM_GOOD]`: Specifies the number of evaders.(Note: in our manuscripts, we used [NUM_ADV, NUM_GOOD] $ \in $  {[3,2], [4,3], [5,4]})

**Example**:
If you want to test your own parameters, you can change the default parameters in main script and run or directly run following script in terminal:

```bash
python main_tnnls_dsmoe.py --algorithm "pi-MoRE" --seed 0 --num_adversaries 4 --num_good_agents 3 --critic_gate_type "quadratic" --critic_moe_type "residual" --critic_local_nn_num_cells 96 64 56 48 56 48 56 48 56 48  --critic_num_experts 4 --critic_top_k 2
```

Training logs, checkpoints, and results will be saved under the wandb directory inside the root path.

# Directly run manuscripts results
## diagnostic experiment
If you want to directly run the main experiments in Section Problem Formulation-C, first run following cmd in root:
```bash
cd diagnostic_experiment
```
Then run the main script (if the terminal return warning of missing packages, you can follow the README.md inside the diagnostic_experiment to check the missing python packages.):
```bash
python main_diagnostic_experiment.py
```
This will sequentially run the 4 network architectures with seed = 0~30.

## main results
If you want to directly run the main experiments in Section Experimental Results-C, first run following cmd in root:
```bash
cd 1_main_results
```
Then run the provided bash:
```bash
bash run_main.sh
```
This will sequentially run the 5 algorithms with seed = 0,1,2 and environment setting with [NUM_ADV, NUM_GOOD] = [3,2],[4,3],[5,4].

## ablation 1: MoE Structure
If you want to directly run the main experiments in Section Experimental Results - D - 1), first run following cmd in root:
```bash
cd 2_ablation_residual_structure
```
Then run the provided bash:
```bash
bash run_ablation_stru.sh
```
This will sequentially run the 3 moe structures with seed = 0,1,2 and environment setting with [NUM_ADV, NUM_GOOD] = [3,2],[4,3],[5,4].

## ablation 2: Gating Mechanism
If you want to directly run the main experiments in Section Experimental Results - D - 2), first run following cmd in root:
```bash
cd 3_ablation_gating_mechanism
```
Then run the provided bash:
```bash
bash run_ablation_gate.sh
```
This will sequentially run the 4 gating mechanisms with seed = 0,1,2 and environment setting with [NUM_ADV, NUM_GOOD] = [3,2],[4,3],[5,4].

## ablation 3: Top-K Routing
If you want to directly run the main experiments in Section Experimental Results - D - 3), first run following cmd in root:
```bash
cd 4_ablation_top_k
```
Then run the provided bash:
```bash
bash run_ablation_topk.sh
```
This will sequentially run the 4 gating mechanism with seed = 0,1,2 and environment setting with [NUM_ADV, NUM_GOOD] = [3,2],[4,3],[5,4].

