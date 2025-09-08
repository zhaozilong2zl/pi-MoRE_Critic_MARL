"""
Implementation of the pi-MoRE critic model

"""

from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import Optional, Sequence, Type

import torch
from tensordict import TensorDictBase
from torch import nn, Tensor
from torchrl.modules import MLP
from torchrl.modules import MultiAgentMLP

from benchmarl.models.common import Model, ModelConfig

import os
import csv
from datetime import datetime

class MoREnet(nn.Module):
    """
    Implements the Mixture of Residual Experts (MoRE) network.

    This module replaces the standard MLP (phi network) in the DeepSets architecture.
    It is specifically designed to handle heterogeneous inputs, such as the mix of
    information-rich observations from active agents and zero-padded observations
    from inactive agents.

    The primary architecture is the 'residual' MoE, which consists of:
    1.  A Shared Expert: A single MLP that processes every input to learn a robust
        baseline representation and ensure stable gradient flow.
    2.  Sparse Experts: A set of smaller, specialized MLPs.
    3.  A Gating Network: A trainable network that dynamically selects a small
        subset of sparse experts (Top-K) for each input. This allows for
        specialized processing of different input patterns.
    4.  A Residual Connection: The final output is the sum of the shared expert's
        output and the weighted output of the selected sparse experts.
    Args:
            in_features (int): Dimensionality of the input features.
            out_features (int): Dimensionality of the output features.
            n_agents (int): Number of agents in the environment.
            moe_type (str): The type of MoE architecture. 'residual' is the main model.
                            'vanilla' and 'vanilla_sparse' are for ablation studies.
            gate_type (str): The type of gating network ('linear', 'mlp', 'quadratic', 'one_hot').
            top_k (int): The number of experts to route each input to.
            num_experts (int): The number of sparse experts.
            num_cells (list): A list of integers defining the hidden layer sizes for all MLPs.
                              For 'residual' moe_type, it's consumed as: [shared_1, shared_2,
                              expert1_1, expert1_2, expert2_1, expert2_2, ...].
            activation_class: The activation function to use in the MLPs.
            device: The device to place the tensors on.
    """
    def __init__(self,in_features: int,
                 out_features: int,
                 n_agents: int,
                 moe_type: str = "residual",  # 'residual' or 'shared_expert'
                 gate_type: str = "linear",
                 top_k: int = 2,
                 num_experts: int = 4,
                 num_cells: list = [128, 64, 64, 32, 64, 32, 64, 32, 64, 32],
                 activation_class=nn.ReLU,
                 device="cuda"):
        super().__init__()

        self.num_experts = num_experts
        self.device = device

        # MoE structure
        cells_iter = iter(num_cells)

        self.moe_type = moe_type
        self.top_k = top_k

        if self.moe_type == "residual":
            print("using residual moe structure")

            shared_expert_cells = [next(cells_iter), next(cells_iter)]  # 2 cells for shared expert in default
            self.shared_expert = MLP(
                in_features=in_features,
                out_features=out_features,
                num_cells=shared_expert_cells,
                activation_class=activation_class,
                device=device
            )

            # num_experts sparse experts, each with 2 cells in default
            self.experts = nn.ModuleList()
            for i in range(self.num_experts):
                experts_cells = [next(cells_iter), next(cells_iter)]
                self.experts.append(
                    MLP(
                        in_features=in_features,
                        out_features=out_features,
                        num_cells=experts_cells,
                        activation_class=activation_class,
                        device=device
                    )
                )

        elif self.moe_type == "vanilla":
            print("using vanilla moe structure")
            self.experts = nn.ModuleList()
            for i in range(self.num_experts):
                experts_cells = [next(cells_iter), next(cells_iter)]
                self.experts.append(
                    MLP(
                        in_features=in_features,
                        out_features=out_features,
                        num_cells=experts_cells,
                        activation_class=activation_class,
                        device=device
                    )
                )

        elif self.moe_type == "vanilla_sparse":
            print("using vanilla_sparse moe structure")
            self.experts = nn.ModuleList()
            for i in range(self.num_experts):
                experts_cells = [next(cells_iter), next(cells_iter)]
                self.experts.append(
                    MLP(
                        in_features=in_features,
                        out_features=out_features,
                        num_cells=experts_cells,
                        activation_class=activation_class,
                        device=device
                    )
                )

        # Gate network
        self.gate_type = gate_type
        self.in_features = in_features
        if self.gate_type in ["quadratic", "unbalance_expert"]:
            # implement refer to https://arxiv.org/abs/2410.11222
            # each expert_i correspond to A_i ([in_features, in_features]) and c_i (scalar)
            self.quadratic_A_params = nn.ParameterList()
            for _ in range(num_experts):
                param = nn.Parameter(torch.randn(self.in_features, self.in_features, device=self.device))
                nn.init.xavier_uniform_(param)
                self.quadratic_A_params.append(param)

            self.quadratic_c_params = nn.ParameterList()
            for _ in range(num_experts):
                param = nn.Parameter(torch.randn(1, device=self.device))
                nn.init.zeros_(param)
                self.quadratic_c_params.append(param)

        elif self.gate_type=="linear":
            self.gate = nn.Linear(self.in_features, num_experts).to(self.device)

        elif self.gate_type=="mlp":
            self.gate = nn.Sequential(
                nn.Linear(self.in_features, 64),
                activation_class(),
                nn.Linear(64, num_experts),
            ).to(self.device)

        # one_hot gate type is handled inside the forward pass as it's data-dependent

    def _get_gate_logits(self, x_gate: Tensor) -> Tensor:
        """Computes the routing logits for the experts based on the gate type."""
        if self.gate_type in ["quadratic", "unbalance_expert"]:
            expert_scores = []
            for i in range(self.num_experts):
                A_i = self.quadratic_A_params[i]  # [F, F]
                A_i_sym = 0.5 * (A_i + A_i.T)
                c_i = self.quadratic_c_params[i]  # [1]

                # compute x^T A_i x
                quadratic_term = torch.sum((x_gate @ A_i_sym) * x_gate, dim=-1)
                score = quadratic_term + c_i.squeeze()
                expert_scores.append(score)

            stacked_scores = torch.stack(expert_scores, dim=-1)  # [..., num_experts]
            gates = torch.softmax(stacked_scores, dim=-1)  # [..., num_experts]

        elif self.gate_type == "linear" or self.gate_type == "mlp":
            gates = torch.softmax(self.gate(x_gate), dim=-1)  # [..., num_experts]

        elif self.gate_type == "one_hot":  # expert_num should be 2
            # detect all zero input
            is_all_zero = torch.all(x_gate == 0, dim=-1)  # [batch_size, n_agents] or [batch_size]

            gates = torch.stack([
                (~is_all_zero).float(),  # first expert：active(no all zero) -> 1
                is_all_zero.float()  # second expert：inactive(all zero) -> 1
            ], dim=-1)
        
        return gates

    def forward(self, x):
        # x:  [batch_size, max_steps, agents, feature] or [batch_size * max_steps, agents, feature]
        if x.dim() > 4:  # if have set(group) dim, [batch_size * max_steps, agents, set, feature]
            x_gate = x.mean(dim=-2)  # using mean pooling to aggregate set features
        else:
            x_gate = x

        if self.moe_type == "residual":
            # residual MoE with Top-K Routing
            all_sparse_experts_outputs = torch.stack([e(x) for e in self.experts], dim=-1)  # [..., out_features, num_experts]

            shared_experts_outputs = self.shared_expert(x)  # [..., out_features]
            
            gate_logits = self._get_gate_logits(x_gate)  # [..., num_experts]
            
            # Top-K gating
            top_k_logits, top_k_indices = torch.topk(input=gate_logits, k=self.top_k, dim=-1)
            top_k_weights = torch.softmax(top_k_logits, dim=-1)  # [..., top_k]
            
            # gather outputs for top-k experts
            # expand indices from [...,top_k] -> [..., 1, top_k] ->[..., out_features, top_k]
            expand_indices = top_k_indices.unsqueeze(dim=-2).expand(-1, *x.shape[1:-1],all_sparse_experts_outputs.shape[-2],self.top_k) 
            # gather outputs from all_sparse_experts_outputs according to expand_indices
            top_k_expert_outputs = torch.gather(all_sparse_experts_outputs, dim=-1, index=expand_indices)
            
            # weighted sum of top-k expert outputs
            gated_expert_outputs = torch.sum(top_k_expert_outputs * top_k_weights.unsqueeze(-2), dim=-1)  # [..., out_features]
            
            return shared_experts_outputs + gated_expert_outputs
        
        elif self.moe_type == "vanilla":
            # vanilla MoE with all experts
            gates = self._get_gate_logits(x_gate)  # [..., num_experts]
            
            expert_outputs = torch.stack([e(x) for e in self.experts], dim=-1)  # [..., out_features, num_experts]

            return torch.sum(gates.unsqueeze(-2) * expert_outputs, dim=-1)

        elif self.moe_type == "vanilla_sparse":
            # vanilla MoE with sparse experts
            # residual MoE with Top-K Routing
            all_sparse_experts_outputs = torch.stack([e(x) for e in self.experts],
                                                     dim=-1)  # [..., out_features, num_experts]

            gate_logits = self._get_gate_logits(x_gate)  # [..., num_experts]

            # Top-K gating
            top_k_logits, top_k_indices = torch.topk(input=gate_logits, k=self.top_k, dim=-1)
            top_k_weights = torch.softmax(top_k_logits, dim=-1)  # [..., top_k]

            # gather outputs for top-k experts
            # expand indices from [...,top_k] -> [..., 1, top_k] ->[..., out_features, top_k]
            expand_indices = top_k_indices.unsqueeze(dim=-2).expand(-1, *x.shape[1:-1],
                                                                    all_sparse_experts_outputs.shape[-2], self.top_k)
            # gather outputs from all_sparse_experts_outputs according to expand_indices
            top_k_expert_outputs = torch.gather(all_sparse_experts_outputs, dim=-1, index=expand_indices)

            # weighted sum of top-k expert outputs
            gated_expert_outputs = torch.sum(top_k_expert_outputs * top_k_weights.unsqueeze(-2),
                                             dim=-1)  # [..., out_features]

            return gated_expert_outputs



class PIMoRE(Model):
    """
    A wrapper class that integrates the pi-MoRE critic into the BenchMARL framework.

    This class constructs a DeepSets network (`_DeepsetsNet`) where the local
    processing network (phi) is replaced by our custom `MoREnet`. It handles
    the BenchMARL-specific logic of parsing input specifications, managing
    parameter sharing, and constructing the appropriate local and global
    network structures for centralized training.

    Args:
        aggr (str): The aggregation strategy for the DeepSets backbone.
        local_nn_num_cells (Sequence[int]): Hidden layer sizes for the MoREnet experts.
        out_features_local_nn (int): Output dimensionality of the MoREnet (phi network).
        global_nn_num_cells (Sequence[int]): Hidden layer sizes for the global MLP (rho network).
        gate_type (str): The gate type to be used in the MoREnet.
        moe_type (str): The MoE architecture type for the MoREnet.
        top_k (int): The number of experts to select in the MoREnet.
        num_experts (int): The number of sparse experts in the MoREnet.
    """

    def __init__(
        self,
        aggr: str,
        local_nn_num_cells: Sequence[int],
        local_nn_activation_class: Type[nn.Module],
        out_features_local_nn: int,
        global_nn_num_cells: Sequence[int],
        global_nn_activation_class: Type[nn.Module],
        gate_type: str,
        moe_type: str,
        top_k: int = 2,
        num_experts: int = 2,
        **kwargs,
    ):

        super().__init__(**kwargs)
        self.aggr = aggr
        self.local_nn_num_cells = local_nn_num_cells
        self.local_nn_activation_class = local_nn_activation_class
        self.global_nn_num_cells = global_nn_num_cells
        self.global_nn_activation_class = global_nn_activation_class
        self.out_features_local_nn = out_features_local_nn
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate_type = gate_type
        self.moe_type = moe_type

        self.input_local_set_features = sum(
            [self.input_spec[key].shape[-1] for key in self.set_in_keys_local]
        )
        self.input_local_tensor_features = sum(
            [self.input_spec[key].shape[-1] for key in self.tensor_in_keys_local]
        )
        self.input_global_set_features = sum(
            [self.input_spec[key].shape[-1] for key in self.set_in_keys_global]
        )
        self.input_global_tensor_features = sum(
            [self.input_spec[key].shape[-1] for key in self.tensor_in_keys_global]
        )

        self.output_features = self.output_leaf_spec.shape[-1]

        if self.input_local_set_features > 0:  # Need local deepsets
            self.local_deepsets = nn.ModuleList(
                [
                    self._make_deepsets_net(
                        in_features=self.input_local_set_features,
                        out_features_local_nn=self.out_features_local_nn,
                        in_fetures_global_nn=self.out_features_local_nn
                        + self.input_local_tensor_features,
                        out_features=(
                            self.output_features
                            if not self.centralised
                            else self.out_features_local_nn
                        ),
                        aggr=self.aggr,
                        local_nn_activation_class=self.local_nn_activation_class,
                        global_nn_activation_class=self.global_nn_activation_class,
                        local_nn_num_cells=self.local_nn_num_cells,
                        global_nn_num_cells=self.global_nn_num_cells,
                    )
                    for _ in range(self.n_agents if not self.share_params else 1)
                ]
            )
        if self.centralised:  # Need global deepsets
            self.global_deepsets = nn.ModuleList(
                [
                    self._make_deepsets_net(
                        in_features=(
                            self.input_global_set_features
                            if self.input_local_set_features == 0
                            else self.out_features_local_nn
                        ),
                        out_features_local_nn=self.out_features_local_nn,
                        in_fetures_global_nn=self.out_features_local_nn
                        + self.input_global_tensor_features,
                        out_features=self.output_features,
                        aggr=self.aggr,
                        local_nn_activation_class=self.local_nn_activation_class,
                        global_nn_activation_class=self.global_nn_activation_class,
                        local_nn_num_cells=self.local_nn_num_cells,
                        global_nn_num_cells=self.global_nn_num_cells,
                    )
                    for _ in range(self.n_agents if not self.share_params else 1)
                ]
            )
        self._params_counted = False  # flag: if get counting params yet
        self._init_params = -1  # flag: avoid counting params before instantiating

    def _make_deepsets_net(
        self,
        in_features: int,
        out_features: int,
        aggr: str,
        local_nn_num_cells: Sequence[int],
        local_nn_activation_class: Type[nn.Module],
        global_nn_num_cells: Sequence[int],
        global_nn_activation_class: Type[nn.Module],
        out_features_local_nn: int,
        in_fetures_global_nn: int,
    ) -> _DeepsetsNet:
        local_nn = MoREnet(
            in_features=in_features,
            out_features=out_features_local_nn,
            n_agents=self.n_agents,
            num_experts=self.num_experts,
            top_k=self.top_k,
            gate_type=self.gate_type,
            moe_type=self.moe_type,
            num_cells=local_nn_num_cells,
            activation_class=local_nn_activation_class,
            device=self.device,
        )
        global_nn = MLP(
            in_features=in_fetures_global_nn,
            out_features=out_features,
            num_cells=global_nn_num_cells,
            activation_class=global_nn_activation_class,
            device=self.device,
        )
        return _DeepsetsNet(local_nn, global_nn, aggr=aggr)

    def _perform_checks(self):
        super()._perform_checks()

        input_shape_tensor_local = None
        self.tensor_in_keys_local = []
        input_shape_set_local = None
        self.set_in_keys_local = []

        input_shape_tensor_global = None
        self.tensor_in_keys_global = []
        input_shape_set_global = None
        self.set_in_keys_global = []

        error_invalid_input = ValueError(
            f"DeepSet set inputs should all have the same shape up to the last dimension, got {self.input_spec}"
        )

        for input_key, input_spec in self.input_spec.items(True, True):
            if self.input_has_agent_dim and len(input_spec.shape) == 3:
                self.set_in_keys_local.append(input_key)
                if input_shape_set_local is None:
                    input_shape_set_local = input_spec.shape[:-1]
                elif input_spec.shape[:-1] != input_shape_set_local:
                    raise error_invalid_input
            elif self.input_has_agent_dim and len(input_spec.shape) == 2:
                self.tensor_in_keys_local.append(input_key)
                if input_shape_tensor_local is None:
                    input_shape_tensor_local = input_spec.shape[:-1]
                elif input_spec.shape[:-1] != input_shape_tensor_local:
                    raise error_invalid_input
            elif not self.input_has_agent_dim and len(input_spec.shape) == 2:
                self.set_in_keys_global.append(input_key)
                if input_shape_set_global is None:
                    input_shape_set_global = input_spec.shape[:-1]
                elif input_spec.shape[:-1] != input_shape_set_global:
                    raise error_invalid_input
            elif not self.input_has_agent_dim and len(input_spec.shape) == 1:
                self.tensor_in_keys_global.append(input_key)
                if input_shape_tensor_global is None:
                    input_shape_tensor_global = input_spec.shape[:-1]
                elif input_spec.shape[:-1] != input_shape_tensor_global:
                    raise error_invalid_input
            else:
                raise ValueError(
                    f"DeepSets input value {input_key} from {self.input_spec} has an invalid shape"
                )

        # Centralized model not needing any local deepsets
        if (
            self.centralised
            and not len(self.set_in_keys_local)
            and self.input_has_agent_dim
        ):
            self.set_in_keys_global = self.tensor_in_keys_local
            input_shape_set_global = input_shape_tensor_local
            self.tensor_in_keys_local = []

        if (not self.centralised and not len(self.set_in_keys_local)) or (
            self.centralised
            and not self.input_has_agent_dim
            and not len(self.set_in_keys_global)
        ):
            raise ValueError("DeepSets found no set inputs, maybe use an MLP?")

        if len(self.set_in_keys_local) and input_shape_set_local[-2] != self.n_agents:
            raise ValueError()
        if (
            len(self.tensor_in_keys_local)
            and input_shape_tensor_local[-1] != self.n_agents
        ):
            raise ValueError()
        if (
            len(self.set_in_keys_global)
            and self.input_has_agent_dim
            and input_shape_set_global[-1] != self.n_agents
        ):
            raise ValueError()

        if (
            self.output_has_agent_dim
            and (
                self.output_leaf_spec.shape[-2] != self.n_agents
                or len(self.output_leaf_spec.shape) != 2
            )
        ) or (not self.output_has_agent_dim and len(self.output_leaf_spec.shape) != 1):
            raise ValueError()

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        
        # parameter count
        if not self._params_counted:
            total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            if self._init_params == total_params:
                print(f"\n--- Model Parameter Count (from {self.__class__.__name__} forward) ---")
                # add more information about the model
                model_info = f"Agent Group: {self.agent_group}, Model Index: {self.model_index}, Is Critic: {self.is_critic}"
                print(model_info)
                print(f"{self.__class__.__name__} Model Parameters: {total_params}")
                # print device
                try:
                    if any(p.requires_grad for p in self.parameters()):
                        print(
                            f"{self.__class__.__name__} Model Device: {next(p for p in self.parameters() if p.requires_grad).device}")
                    else:
                        print(f"{self.__class__.__name__} Model Device: N/A (No trainable parameters)")
                except StopIteration:
                    print(f"{self.__class__.__name__} Model Device: Device access failed (possibly no parameters?)")
                print("---------------------------------------------------------------\n")
                self._params_counted = True
            self._init_params = total_params
        
        if len(self.set_in_keys_local):
            # Local deep sets
            input_local_sets = torch.cat(
                [tensordict.get(in_key) for in_key in self.set_in_keys_local], dim=-1
            )
            input_local_tensors = None
            if len(self.tensor_in_keys_local):
                input_local_tensors = torch.cat(
                    [tensordict.get(in_key) for in_key in self.tensor_in_keys_local],
                    dim=-1,
                )
            if self.share_params:
                local_output = self.local_deepsets[0](
                    input_local_sets, input_local_tensors
                )
            else:
                local_output = torch.stack(
                    [
                        net(input_local_sets, input_local_tensors)[..., i, :]
                        for i, net in enumerate(self.local_deepsets)
                    ],
                    dim=-2,
                )
        else:
            local_output = None

        if self.centralised:
            if local_output is None:
                # gather local output
                local_output = torch.cat(
                    [tensordict.get(in_key) for in_key in self.set_in_keys_global],
                    dim=-1,
                )

            # Global deepsets
            input_global_tensors = None
            if len(self.tensor_in_keys_global):
                input_global_tensors = torch.cat(
                    [tensordict.get(in_key) for in_key in self.tensor_in_keys_global],
                    dim=-1,
                )
            if self.share_params:
                global_output = self.global_deepsets[0](
                    local_output, input_global_tensors
                )
            else:
                global_output = torch.stack(
                    [
                        net(local_output, input_global_tensors)
                        for i, net in enumerate(self.global_deepsets)
                    ],
                    dim=-2,
                )
            tensordict.set(self.out_key, global_output)
        else:
            tensordict.set(self.out_key, local_output)

        return tensordict


class _DeepsetsNet(nn.Module):
    """
    We use the DeepSets network as our backbone architecture.
    https://arxiv.org/abs/1703.06114
    """

    def __init__(
        self,
        local_nn: torch.nn.Module,
        global_nn: torch.nn.Module,
        set_dim: int = -2,
        aggr: str = "sum",
    ):
        super().__init__()
        self.aggr = aggr
        self.set_dim = set_dim
        self.local_nn = local_nn
        self.global_nn = global_nn

    def forward(self, x: Tensor, extra_global_input: Optional[Tensor]) -> Tensor:
        x = self.local_nn(x)
        x = self.reduce(x, dim=self.set_dim, aggr=self.aggr)
        if extra_global_input is not None:
            x = torch.cat([x, extra_global_input], dim=-1)
        x = self.global_nn(x)
        return x

    @staticmethod
    def reduce(x: Tensor, dim: int, aggr: str) -> Tensor:
        if aggr == "sum" or aggr == "add":
            return torch.sum(x, dim=dim)
        elif aggr == "mean":
            return torch.mean(x, dim=dim)
        elif aggr == "max":
            return torch.max(x, dim=dim)[0]
        elif aggr == "min":
            return torch.min(x, dim=dim)[0]
        elif aggr == "mul":
            return torch.prod(x, dim=dim)


@dataclass
class PIMoREConfig(ModelConfig):
    aggr: str = MISSING
    out_features_local_nn: int = MISSING
    local_nn_num_cells: Sequence[int] = MISSING
    local_nn_activation_class: Type[nn.Module] = MISSING
    global_nn_num_cells: Sequence[int] = MISSING
    global_nn_activation_class: Type[nn.Module] = MISSING
    num_experts: int = MISSING
    top_k: int = MISSING
    gate_type: str = MISSING
    moe_type: str = MISSING

    @staticmethod
    def associated_class():
        return PIMoRE
