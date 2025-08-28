# code for Fig. 2, the Revised Average Task in Section 2-C
# heavily inspired by https://arxiv.org/abs/2111.05992 and https://github.com/Unity-Technologies/paper-ml-agents/tree/main/ma-poca

import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
import math
import os
from tqdm import tqdm

if not hasattr(np, 'float'):
    np.float = float

# MLP
class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_size=32):
        super(MLP, self).__init__()
        hidden_size = hidden_size
        self.dense1 = torch.nn.Linear(input_dim, hidden_size)
        self.dense2 = torch.nn.Linear(hidden_size, hidden_size)
        self.dense3 = torch.nn.Linear(hidden_size, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, data):
        h = self.dense1(data)
        h = self.relu(h)
        h = self.dense2(h)
        h = self.relu(h)
        return self.dense3(h)

# Attention
class AttentionNetwork(torch.nn.Module):
    """
    our implementation of the Residual Self-Attention model follows the description in https://arxiv.org/abs/2111.05992
    """
    # input_dim=1
    def __init__(self, input_dim=1, embed_dim=32, num_heads=4):
        super(AttentionNetwork, self).__init__()
        self.embed = torch.nn.Linear(input_dim, embed_dim)
        self.attention = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.fc = torch.nn.Linear(embed_dim, 1)
        self.layer_norm1 = torch.nn.LayerNorm(embed_dim)
        self.layer_norm2 = torch.nn.LayerNorm(embed_dim)
        self.abs_state_val = 0.0  # assume that absorbing state is 0.0

    def forward(self, data):

        batch_size, n_input = data.shape  # data shape: (batch_size, n_input)

        key_padding_mask = (data == self.abs_state_val)  # mask shape: (batch_size, n_input)

        embedded_data = self.embed(data.unsqueeze(-1))  # data shape becomes: (batch_size, n_input, 1) for embedding
        embedded_data = self.layer_norm1(embedded_data)  # LayerNorm after embedding

        # masked self-attention
        attn_output, _ = self.attention(embedded_data, embedded_data, embedded_data,
                                        key_padding_mask=key_padding_mask)

        # Residual connection and layerNorm
        attn_output = self.layer_norm2(attn_output + embedded_data)

        aggregated_embedding = attn_output.mean(dim=1)  # Shape: (batch_size, embed_dim)

        # valid_output_mask = ~key_padding_mask.unsqueeze(-1)  # Shape: (batch_size, n_input, 1)
        # attn_output_masked = attn_output * valid_output_mask
        # # count how many valid outputs
        # num_valid = valid_output_mask.sum(dim=1)  # Shape: (batch_size, 1)
        # num_valid = num_valid.clamp(min=1)
        #
        # # obtain the aggregated embedding by summing the masked attention output
        # aggregated_embedding = attn_output_masked.sum(dim=1) / num_valid # Shape: (batch_size, embed_dim)



        return self.fc(aggregated_embedding)

# permutation invariant networks (Deepsets)
class PiNetwork(torch.nn.Module):
    """
    https://arxiv.org/abs/1703.06114
    """
    def __init__(self, input_dim=1, phi_hdim=32, emb_dim=32, rho_hdim=32):
        super(PiNetwork, self).__init__()
        self.phi = torch.nn.Sequential(
            torch.nn.Linear(input_dim, phi_hdim),
            torch.nn.ReLU(),
            torch.nn.Linear(phi_hdim, emb_dim),
            torch.nn.ReLU()
        )
        self.rho = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, rho_hdim),
            torch.nn.ReLU(),
            torch.nn.Linear(rho_hdim, 1)
        )

    def forward(self, data):
        batch_size, n_input = data.shape
        x = data.unsqueeze(-1)
        x = self.phi(x)
        x = x.sum(dim=1)
        return self.rho(x)

# permutation invariant Mixture of Experts (pi-MoE), with top-k / one-hot gate
class PiMoE(torch.nn.Module):
    """
    simplified version of our methods: deepsets + MoE, where MoE replace the phi function in the deepsets
    """
    def __init__(self, input_dim=1, exp_hdim=32, emb_hdim=32, rho_hdim=32, num_experts=2, top_k=2, one_hot_gate=False):
        super(PiMoE, self).__init__()
        self.num_experts = num_experts
        self.experts = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(input_dim, exp_hdim),
                torch.nn.ReLU(),
                torch.nn.Linear(exp_hdim, emb_hdim),
                torch.nn.ReLU()
            ) for _ in range(num_experts)
        ])
        self.gate = torch.nn.Linear(input_dim, num_experts)
        self.rho = torch.nn.Sequential(
            torch.nn.Linear(emb_hdim, rho_hdim),
            torch.nn.ReLU(),
            torch.nn.Linear(rho_hdim, 1)
        )
        self.top_k = top_k
        self.one_hot_gate = one_hot_gate

    def forward(self, data):
        batch_size, n_input = data.shape
        x = data.unsqueeze(-1)
        gate_logits = self.gate(x)

        if self.one_hot_gate:
            # one-hot logits
            if self.num_experts==2:
                gate_weights = torch.zeros_like(gate_logits)
                is_zero = (x.squeeze(-1) == 0)
                gate_weights[..., 0] = is_zero.float()
                gate_weights[..., 1] = (~is_zero).float()
            else:
                print(f"one-hot should have num_experts == 2 but now got {self.num_experts}")
        else:
            ## top_k logits
            #top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
            #top_k_softmax = torch.softmax(top_k_logits, dim=-1)
            #sparse_weights = torch.zeros_like(gate_logits)
            #sparse_weights.scatter_(-1, top_k_indices, top_k_softmax)
            #gate_weights = sparse_weights

            gate_weights = torch.softmax(gate_logits, dim=-1)

        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(x)
            expert_outputs.append(expert_out)
        expert_outputs = torch.stack(expert_outputs, dim=2)  # shape: (batch_size, n_input, num_experts, emb_hdim)
        gate_weights = gate_weights.unsqueeze(-1)  # shape: (batch_size, n_input, num_experts, 1)
        weighted_outputs = expert_outputs * gate_weights
        weighted_outputs = weighted_outputs.sum(dim=2)
        x = weighted_outputs.sum(dim=1)
        return self.rho(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def generate_batch(batch, max_num_absorb, absorb_state, LOW, HIGH, sample_random, N, v0):
    """
    sample batch data, data shape: (batch, N), and the element are constrainted in [LOW, HIGH]
    """
    inputs = np.random.uniform(LOW, HIGH, (batch, N))
    inputs = np.float32(inputs)

    target_revised_avg = np.zeros((batch, 1), dtype=np.float32)

    for b in range(batch):
        current_inputs = inputs[b]

        # if sample_random=True, there are less than num_to_absorb_state elements set to the absorbing state
        if sample_random:
            num_to_absorb_state = random.choice(range(max_num_absorb+1))
        else:
            num_to_absorb_state = max_num_absorb

        # randomly select indices to set to the absorbing state
        indices_to_zero = np.random.choice(N, num_to_absorb_state, replace=False)
        current_inputs[indices_to_zero] = absorb_state


        np.random.shuffle(current_inputs)  # shuffling the order
        inputs[b] = current_inputs

        # calculate the revised average

        zeros_mask = (current_inputs == absorb_state)
        non_zeros_mask = ~zeros_mask

        sum_non_zeros = np.sum(current_inputs[non_zeros_mask])

        num_zeros = np.sum(zeros_mask)
        revised_sum = sum_non_zeros + num_zeros * v0
        current_revised_avg = revised_sum / float(N)
        target_revised_avg[b] = current_revised_avg

    return inputs, target_revised_avg.astype(np.float32)


if __name__ == '__main__':

    n_input = 10  # length of inputs
    num_absorb_state = 4  # maximum number of absorbing states
    sample_random = True
    absorbing_state_value = 0.0
    LOW = 0.25
    HIGH = 0.75
    v_0 = 0.1  # fixed value for absorbing state, used in the revised average task

    batch_size = 500
    num_seed = 30
    epochs = 200*5
    LOG = True

    force_retrain = False  # if is false, will load the existing data file if it exists
    partial_data_dirname = "partial_results"
    os.makedirs(partial_data_dirname, exist_ok=True)  # create directory for partial results if it does not exist

    network_types = ["MLP", "Attention", "PiNetwork", "PiMoE_top_k"]

    sns.set_style("whitegrid", {'font.family': 'serif', 'font.serif': 'Times New Roman'})

    # global random seed
    np.random.seed(1336)
    torch.manual_seed(1336)
    # ignore warning
    torch.set_default_dtype(torch.float32)
    torch.set_default_device('cpu')


    for net_type in network_types:

        model_file_name = os.path.join(partial_data_dirname, f"data_{net_type}_num_seed_{num_seed}.csv")

        if not force_retrain and os.path.exists(model_file_name):
            print(f"Skipping {net_type}, Results already exist.")
            continue

        net_dfs = []

        pbar = tqdm(range(num_seed), desc=f"training progress of {net_type}", unit="seed")
        for seed in pbar:
            torch.manual_seed(seed)
            if net_type == "MLP":
                encoder = MLP(input_dim=n_input, hidden_size=64)
            elif net_type == "Attention":
                encoder = AttentionNetwork(input_dim=1, embed_dim=32, num_heads=4)
            elif net_type == "PiNetwork":
                encoder = PiNetwork(input_dim=1, phi_hdim=32, emb_dim=64, rho_hdim=32)
            elif net_type == "PiMoE_one_hot":
                encoder = PiMoE(input_dim=1, exp_hdim=24, emb_hdim=64, rho_hdim=16, num_experts=2, one_hot_gate=True)
            elif net_type == "PiMoE_top_k":
                encoder = PiMoE(input_dim=1, exp_hdim=24, emb_hdim=64, rho_hdim=16, num_experts=2, top_k=2)
            else:
                raise NotImplementedError(f"no this kind if net_type: {net_type}")

            if seed==0:
                print(f"{net_type} has {count_parameters(encoder):,} trainable parameters")

            optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)
            loss_values = []

            for e in range(epochs):
                data, targ = generate_batch(batch=batch_size,
                                            max_num_absorb=num_absorb_state,
                                            absorb_state=absorbing_state_value,
                                            LOW=LOW,
                                            HIGH=HIGH,
                                            sample_random=sample_random,
                                            N= n_input,
                                            v0=v_0)
                data = torch.from_numpy(data)
                targ = torch.from_numpy(targ)
                pred = encoder(data)
                loss = torch.mean((targ - pred) ** 2)
                if LOG:
                    loss_values.append(loss.log().item())
                else:
                    loss_values.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (e + 1) % 100 == 0:
                    pbar.set_postfix(current_loss=f"{loss.item():.4e}")

            pbar.set_postfix(last_seed_loss=f"{loss.item():.4e}", status="Done")

            df = pd.DataFrame({
                "Epochs": range(len(loss_values)),
                "Log Mean Squared Error": loss_values,
                "seed": seed
            })
            net_dfs.append(df)

        pbar.close()
        combined_df = pd.concat(net_dfs, ignore_index=True)
        combined_df['Network'] = net_type

        combined_df.to_csv(model_file_name, index=False)
        print(f"\nSaved results for {net_type} to {model_file_name}")

    print("all training finished, loading data for plotting.")
    all_dfs = []
    for net_type in network_types:
        model_file_name = os.path.join(partial_data_dirname, f"data_{net_type}_num_seed_{num_seed}.csv")
        df = pd.read_csv(model_file_name)
        all_dfs.append(df)
    final_df = pd.concat(all_dfs, ignore_index=True)

    rename_map = {
        "MLP": "MLP",
        "Attention": "Attention",
        "PiNetwork": "PiNet",
        "PiMoE_top_k": "PiMoE",
        # "PiMoE_one_hot": "PiMoE(one-hot)"
    }
    final_df['Network'] = final_df['Network'].map(rename_map)


    # plotting

    fig, ax = plt.subplots(figsize=(8,6), dpi=300)
    font_size = 16

    sns.lineplot(
        data=final_df,
        x="Epochs",
        y="Log Mean Squared Error",
        hue="Network",
        style="Network",
        palette="colorblind",
        linewidth=2.5,
        ci=95,  # 95% confidence interval
        ax=ax,
    )

    legend = ax.get_legend()
    plt.setp(legend.get_title(), fontsize=font_size)
    for text in legend.get_texts():
        text.set_fontsize(font_size - 1)


    ax.set_xlabel("Epochs", fontsize=font_size)
    ax.set_ylabel("Log Mean Squared Error", fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size - 2)

    ax.set_xlim(0, epochs)
    ax.set_ylim(-16, -1)

    plt.tight_layout()
    plt.savefig(f"model_comparison_{num_absorb_state}_states.pdf", dpi=300, bbox_inches='tight')

