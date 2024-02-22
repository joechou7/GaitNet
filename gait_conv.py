from typing import Any
from math import ceil
import torch
from torch import Tensor
from torch.nn import Sequential, Linear, BatchNorm1d
from torch.nn import ELU, Conv1d
from torch.nn import Embedding
from torch_geometric.nn import Reshape
from torch_geometric.nn import HeteroLinear



class GaitConv(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 num_node_types: int = None, num_edge_types: int = None, edge_feature_dim: int = None,
                 edge_feature_emb_dim: int = None, edge_type_emb_dim: int = None,
                 bias: bool = True):

        super(GaitConv, self).__init__()

        self.in_channels = in_channels
        if edge_feature_emb_dim is None:
            hidden_channels = in_channels // 4
        else:
            hidden_channels = edge_feature_emb_dim
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        C_in, C_delta, C_out = in_channels, hidden_channels, out_channels
        D, K = edge_feature_dim, kernel_size


        if num_node_types is not None:
            self.lin = HeteroLinear(C_in, C_delta, num_node_types, bias=bias)
        else:
            self.lin = Linear(C_in, C_delta, bias=bias)

        self.act = Sequential(
            ELU(),
            BatchNorm1d(C_delta),
        )

        self.reset_parameters(self.lin)
        self.reset_parameters(self.act)
        C_in = C_delta


        if edge_type_emb_dim is not None:
            self.edge_type_emb = Sequential(
                Embedding(num_edge_types, edge_type_emb_dim),
                ELU(),
                BatchNorm1d(edge_type_emb_dim),
                Reshape(-1, K, edge_type_emb_dim),
            )

            self.reset_parameters(self.edge_type_emb)
            C_in += edge_type_emb_dim


        if edge_feature_emb_dim is not None:
            self.mlp1 = Sequential(
                Linear(D, C_delta),
                ELU(),
                BatchNorm1d(C_delta),
                Linear(C_delta, C_delta),
                ELU(),
                BatchNorm1d(C_delta),
                Reshape(-1, K, C_delta),
            )

            self.mlp2 = Sequential(
                Linear(D * K, K**2),
                ELU(),
                BatchNorm1d(K**2),
                Reshape(-1, K, K),
                Conv1d(K, K**2, K, groups=K),
                ELU(),
                BatchNorm1d(K**2),
                Reshape(-1, K, K),
                Conv1d(K, K**2, K, groups=K),
                BatchNorm1d(K**2),
                Reshape(-1, K, K),
            )

            self.reset_parameters(self.mlp1)
            self.reset_parameters(self.mlp2)
            C_in += edge_feature_emb_dim


        depth_multiplier = int(ceil(C_out / C_in))

        self.conv = Sequential(
            Conv1d(C_in, C_in * depth_multiplier, K, groups=C_in),
            Reshape(-1, C_in * depth_multiplier),
            Linear(C_in * depth_multiplier, C_out, bias=bias),
        )

        self.reset_parameters(self.conv)



    def reset_parameters(self, value: Any):
        if hasattr(value, 'reset_parameters'):
            value.reset_parameters()
        else:
            for child in value.children() if hasattr(value, 'children') else []:
                self.reset_parameters(child)



    def forward(self, x: Tensor, edge_index: Tensor, node_type: Tensor = None,
                edge_feature: Tensor = None, edge_type: Tensor = None):

        x = x.unsqueeze(-1) if x.dim() == 1 else x
        N = x.size(0)
        K = self.kernel_size

        x_star = []

        if node_type is not None:
            x = self.act(self.lin(x, node_type))
        else:
            x = self.act(self.lin(x))
        s = edge_index[1]
        x = x[s].view(N, K, self.hidden_channels)
        x_star.append(x)

        if edge_type is not None:
            x_star.append(self.edge_type_emb(edge_type))

        if edge_feature is not None:
            edge_feature = edge_feature.unsqueeze(-1) if edge_feature.dim() == 1 else edge_feature
            D = edge_feature.size(-1)
            x_star.append(self.mlp1(edge_feature))

        x_star = torch.cat(x_star, dim=-1)
        x_star = x_star.transpose(1, 2).contiguous()

        if edge_feature is not None:
            transform_matrix = self.mlp2(edge_feature.view(N, K * D))
            x_star = torch.matmul(x_star, transform_matrix)

        out = self.conv(x_star)

        return out


