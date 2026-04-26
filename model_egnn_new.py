import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool as gep


def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


class EGNN(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=3, act_fn=nn.SiLU(), residual=None, attention=False,
                 normalize=False, coords_agg='mean', tanh=False):
        super(EGNN, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)

        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord = coord + agg
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff ** 2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)

        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        return h


class EGNNBlock(nn.Module):
    def __init__(self, gcn_layers_dim, dropout_rate=0., relu_layers_index=[], dropout_layers_index=[]):
        super(EGNNBlock, self).__init__()

        self.conv_layers = nn.ModuleList()
        for i in range(len(gcn_layers_dim) - 1):
            conv_layer = EGNN(input_nf=gcn_layers_dim[i], output_nf=gcn_layers_dim[i + 1],
                              hidden_nf=gcn_layers_dim[i + 1])
            self.conv_layers.append(conv_layer)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x, edge_index, edge_weight, coords, batch):
        output = x
        embeddings = []
        for conv_layer_index in range(len(self.conv_layers)):
            output = self.conv_layers[conv_layer_index](output, edge_index, coords, edge_weight)
            if conv_layer_index in self.relu_layers_index:
                output = self.relu(output)
            if conv_layer_index in self.dropout_layers_index:
                output = self.dropout(output)
            embeddings.append(gep(output, batch))
        return embeddings


class EGNNModel(nn.Module):
    def __init__(self, layers_dim):
        super(EGNNModel, self).__init__()
        self.num_layers = len(layers_dim) - 1
        self.graph_conv = EGNNBlock(layers_dim, relu_layers_index=list(range(self.num_layers)))

    def forward(self, graph_batchs):
        embedding_batchs = list(
            map(lambda graph: self.graph_conv(graph.x, graph.edge_index, graph.edge_weight, graph.coords, graph.batch),
                graph_batchs))
        embeddings = []
        for i in range(self.num_layers):
            embeddings.append(torch.cat(list(map(lambda embedding_batch: embedding_batch[i], embedding_batchs)), 0))
        return embeddings


class LinearBlock(nn.Module):
    def __init__(self, linear_layers_dim, dropout_rate=0., relu_layers_index=[], dropout_layers_index=[]):
        super(LinearBlock, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(len(linear_layers_dim) - 1):
            layer = nn.Linear(linear_layers_dim[i], linear_layers_dim[i + 1])
            self.layers.append(layer)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x):
        output = x
        embeddings = []
        for layer_index in range(len(self.layers)):
            output = self.layers[layer_index](output)
            if layer_index in self.relu_layers_index:
                output = self.relu(output)
            if layer_index in self.dropout_layers_index:
                output = self.dropout(output)
            embeddings.append(output)
        return embeddings


# 彻底切除全局图与对比模块后的干净架构
class MLC_DTA(nn.Module):
    def __init__(self, d_ms_dims, t_ms_dims, embedding_dim=128):
        super(MLC_DTA, self).__init__()

        self.output_dim = embedding_dim * 2

        self.drug_graph_conv = EGNNModel(d_ms_dims)
        self.target_graph_conv = EGNNModel(t_ms_dims)

        # ==========================================
        # 🚨 【抢救回来：非线性投影头 (Projection Head)】
        # 替代原 Contrast 类中被误删的 self.proj
        # 这是特征从 3D 几何空间过渡到亲和力预测空间的极其关键的缓冲层！
        # ==========================================
        self.drug_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ELU(),
            nn.Linear(embedding_dim, embedding_dim))

        self.target_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ELU(),
            nn.Linear(embedding_dim, embedding_dim))

        # 严格还原原作者的 Xavier 初始化，保证收敛动态一致
        for model in self.drug_proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)
        for model in self.target_proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def forward(self, drug_graph_batchs, target_graph_batchs):
        # 1. 提取 EGNN 原始 3D 特征
        drug_embedding = self.drug_graph_conv(drug_graph_batchs)[-1]
        target_embedding = self.target_graph_conv(target_graph_batchs)[-1]

        # 2. 通过投影头进行特征映射
        drug_proj_emb = self.drug_proj(drug_embedding)
        target_proj_emb = self.target_proj(target_embedding)

        # 3. 返回给 PredictModule
        return drug_proj_emb, target_proj_emb

class PredictModule(nn.Module):
    def __init__(self, embedding_dim=128, output_dim=1):
        super(PredictModule, self).__init__()
        # 固化为 2 * dim (128+128)
        self.prediction_func, prediction_dim_func = (lambda x, y: torch.cat((x, y), -1), lambda dim: 2 * dim)
        mlp_layers_dim = [prediction_dim_func(embedding_dim), 1024, 512, output_dim]
        self.mlp = LinearBlock(mlp_layers_dim, 0.1, relu_layers_index=[0, 1], dropout_layers_index=[0, 1])

    def forward(self, data, drug_embedding, target_embedding):
        # 彻底修复原来的 cpu().numpy() 瓶颈，直接用原生 tensor 切片
        drug_id, target_id = data.drug_id.long(), data.target_id.long()

        drug_feature = drug_embedding[drug_id]
        target_feature = target_embedding[target_id]

        concat_feature = self.prediction_func(drug_feature, target_feature)
        mlp_embeddings = self.mlp(concat_feature)
        link_embeddings = mlp_embeddings[-2]
        out = mlp_embeddings[-1]

        return out, link_embeddings