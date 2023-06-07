import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys
sys.path.append("models/")
from models.mlp import MLP
import models.structure_model.CIRCA as CIRCA


class TreeIsomorphismNetwork(nn.Module):
    def __init__(self,
                 num_nodes,
                 in_matrix,
                 out_matrix,
                 in_dim,
                 hidden_dim,
                 out_dim=None,
                 dropout=0.0,
                 device=torch.device('cpu'),
                 depth=2,
                 num_mlp_layers=2,
                 tree_pooling_type='sum'):
        super(TreeIsomorphismNetwork, self).__init__()
        self.origin_adj = np.where(in_matrix <= 0, in_matrix, 1.0)
        # self.adj = self.origin_adj * in_matrix + self.origin_adj.T * out_matrix
        self.adj = self.origin_adj + self.origin_adj.T
        import time
        begin_time = time.time()
        self.tree = CIRCA.CodingTree(self.adj)
        if self.adj.shape[0] <= 2:
            m = 'v1'
        else:
            m = 'v2'
        self.tree.build_coding_tree(mode=m, k=depth)
        self.tree = CIRCA.get_child_h(self.tree, k=depth)
        self.tree = CIRCA.map_id(self.tree, k=depth)
        # self.tree = CIRCA.build_random_tree(self.tree, k=depth)  # random tree
        self.tree_node = CIRCA.update_node(self.tree.tree_node)
        self.tree = {
            'node_size': [0] * (depth + 1),
            'leaf_size': num_nodes,
            'edges': [[] for _ in range(depth + 1)],
        }
        layer_idx = [0]
        for layer in range(depth + 1):
            layer_nodes = [i for i, n in self.tree_node.items() if n.child_h == layer]
            layer_idx.append(layer_nodes[0] + len(layer_nodes))
            self.tree['node_size'][layer] = len(layer_nodes)

        for _, n in self.tree_node.items():
            if n.child_h > 0:
                n_idx = n.ID - layer_idx[n.child_h]  # n_idx is the offset of parent node
                c_base = layer_idx[n.child_h - 1]
                # c - c_base is the offset of children node
                self.tree['edges'][n.child_h].extend([(n_idx, c - c_base) for c in n.children])
        build_tree_time = time.time() - begin_time
        print("Coding tree generation costs time of {}".format(build_tree_time))
        sys.exit()
        self.model = nn.ModuleList()
        self.model.append(
            TINConv(tree=self.tree,
                    depth=depth,
                    num_mlp_layers=num_mlp_layers,
                    input_dim=in_dim,
                    hidden_dim=hidden_dim,
                    output_dim=out_dim,
                    final_dropout=dropout,
                    tree_pooling_type=tree_pooling_type,
                    device=device
                    )
        )

    def forward(self, inputs):
        return self.model[0](inputs)


class TINConv(nn.Module):
    def __init__(self, tree,
                 depth,
                 num_mlp_layers,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 final_dropout,
                 tree_pooling_type,
                 device):
        '''
        depth: the depth of coding trees (EXCLUDING the leaf layer)
        num_mlp_layers: number of layers in mlps (EXCLUDING the leaf nodes)
        input_dim: dimensionality of input features
        hidden_dim: dimensionality of hidden units at ALL layers
        output_dim: number of classes for prediction
        final_dropout: dropout ratio on the final linear layer
        tree_pooling_type: how to aggregate entire nodes in a tree (root, sum, mean)
        device: which device to use
        '''

        super(TINConv, self).__init__()

        self.tree = tree
        self.final_dropout = final_dropout
        self.device = device
        self.depth = depth
        self.tree_pooling_type = tree_pooling_type

        # List of MLPs (for hashing)
        self.mlps = torch.nn.ModuleList([None])
        # List of batch_norms applied to the output of MLP (input of the final prediction linear layer)
        self.batch_norms = torch.nn.ModuleList([None])

        for layer in range(1, self.depth+1):  # layer 1 starts from the bottom of coding tree
            if layer == 1:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Linear function that maps the hidden representation at different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()  # note that there is depth+1 layers of hidden state
        self.embedding_dim = []
        for layer in range(self.depth+1):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(input_dim, output_dim))
                self.embedding_dim.append(input_dim)
            else:
                self.linears_prediction.append(nn.Linear(hidden_dim, output_dim))
                self.embedding_dim.append(hidden_dim)

        self.neighbor_pools = self.__preprocess_neighbor_sum_pool()
        self.repeated = False
        self.tree_pool = {
            "avg": torch.mean,
            "sum": torch.sum,
            "max": torch.max
        }

    def __preprocess_neighbor_sum_pool(self):
        # aggregate neighbour information via sum operation
        neighbor_pools = [None]
        for layer in range(1, self.depth+1):
            # 用(vi, vj) pair表示edge, 分层存储, vi和vj的索引是其layer-wise offset
            edge_mat_list = [torch.LongTensor(self.tree['edges'][layer])]
            adj_block_idx = torch.cat(edge_mat_list, 0).transpose(0, 1)
            adj_block_elem = torch.ones(adj_block_idx.shape[1])
            adj_block = torch.sparse.FloatTensor(adj_block_idx, adj_block_elem, torch.Size([self.tree['node_size'][layer], self.tree['node_size'][layer-1]]))
            neighbor_pools.append(adj_block.to(self.device))

        return neighbor_pools

    def next_layer(self, h, layer, adj_block=None):
        # MLP(neighbor_pool(h))
        # [num of parents, num of children] * [num of children, hidden size]
        pooled = torch.spmm(adj_block, h)
        # representation of neighboring and center nodes
        hn = self.mlps[layer](pooled)
        hn = self.batch_norms[layer](hn)
        # non-linearity
        hn = F.relu(hn)

        return hn

    def forward(self, text_feature):
        # [batch, node_size, node_embedding]->[batch * node_size, node_embedding]
        batch_size = text_feature.shape[0]
        h = text_feature.reshape(-1, text_feature.shape[-1])
        h_rep = [h]  # node representation in each layer

        if not self.repeated:
            for layer in range(1, self.depth+1):
                self.neighbor_pools[layer] = self.neighbor_pools[layer].to_dense()
                self.neighbor_pools[layer] = torch.block_diag(*[self.neighbor_pools[layer] for _ in range(text_feature.shape[0])])  # 分块对角矩阵
            self.repeated = True

        # propagate from leaf to root
        for layer in range(1, self.depth+1):
            h = self.next_layer(h, layer, adj_block=self.neighbor_pools[layer])
            h_rep.append(h)

        logits = 0

        if self.tree_pooling_type != 'root':
            # sum the linear prediction of each layer
            for layer in range(self.depth):
                # [batch * node, hidden] -> [batch, node, hidden] -> [batch, hidden]
                assert self.tree_pooling_type in self.tree_pool.keys()
                pooled_h = self.tree_pool[self.tree_pooling_type](
                    h_rep[layer].view(batch_size, -1, self.embedding_dim[layer]),
                    dim=1)
                if self.tree_pooling_type == 'max':
                    pooled_h = pooled_h[0]
                dropped_h = F.dropout(pooled_h, self.final_dropout, training=self.training)
                logits += self.linears_prediction[layer](dropped_h)

        # root pool
        dropped_h = F.dropout(h.squeeze(1), self.final_dropout, training=self.training)
        logits += self.linears_prediction[self.depth](dropped_h)

        return logits  # [batch, output_size]
