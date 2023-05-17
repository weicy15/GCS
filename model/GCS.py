import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_add_pool

from typing import Callable, Union
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import OptPairTensor, Adj, Size
from torch_scatter import scatter, scatter_max, scatter_min, scatter_add

from cam_util.aug import permute_edges
from copy import deepcopy
from torch_geometric.utils import subgraph

def normalize(cam, batch, eps=1e-20):
    cam = cam.clone()
    # batch_num
    batch_max, _ = scatter_max(cam.squeeze(), batch)
    batch_min, _ = scatter_min(cam.squeeze(), batch)
    batch_max_expand = []
    batch_min_expand = []
    for i in batch:
        batch_max_expand.append(batch_max[i])
        batch_min_expand.append(batch_min[i])

    batch_max_expand = torch.tensor(batch_max_expand).unsqueeze(1).to(cam.device)
    batch_min_expand = torch.tensor(batch_min_expand).unsqueeze(1).to(cam.device)
    normalized_cam = (cam - batch_min_expand) / (batch_max_expand + eps)
    normalized_cam = normalized_cam.clamp_min(0)
    normalized_cam = normalized_cam.clamp_max(1)

    return normalized_cam

def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)


class WGINConv(MessagePassing):
    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(WGINConv, self).__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_weight = None,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_weight) -> Tensor:
        return x_j if edge_weight is None else x_j * edge_weight.view(-1, 1)


    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class TUEncoder(torch.nn.Module):
    def __init__(self, num_dataset_features, emb_dim=300, num_gc_layers=5, drop_ratio=0.0, pooling_type="layerwise", is_infograph=False):
        super(TUEncoder, self).__init__()

        self.pooling_type = pooling_type
        self.emb_dim = emb_dim
        self.num_gc_layers = num_gc_layers
        self.drop_ratio = drop_ratio
        self.is_infograph = is_infograph

        self.out_node_dim = self.emb_dim
        if self.pooling_type == "standard":
            self.out_graph_dim = self.emb_dim
        elif self.pooling_type == "layerwise":
            self.out_graph_dim = self.emb_dim * self.num_gc_layers
        else:
            raise NotImplementedError

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_gc_layers):

            if i:
                nn = Sequential(Linear(emb_dim, emb_dim), ReLU(), Linear(emb_dim, emb_dim))
            else:
                nn = Sequential(Linear(num_dataset_features, emb_dim), ReLU(), Linear(emb_dim, emb_dim))
            conv = WGINConv(nn)
            bn = torch.nn.BatchNorm1d(emb_dim)

            self.convs.append(conv)
            self.bns.append(bn)


    def forward(self, batch, x, edge_index, edge_attr=None, edge_weight=None):
        xs = []
        for i in range(self.num_gc_layers):
            x = self.convs[i](x, edge_index, edge_weight)
            x = self.bns[i](x)
            if i == self.num_gc_layers - 1:
                # remove relu for the last layer
                x = F.dropout(x, self.drop_ratio, training=self.training)
            else:
                x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)
            xs.append(x)
        # compute graph embedding using pooling
        if self.pooling_type == "standard":
            xpool = global_add_pool(x, batch)
            return xpool, x

        elif self.pooling_type == "layerwise":
            xpool = [global_add_pool(x, batch) for x in xs]
            xpool = torch.cat(xpool, 1)
            if self.is_infograph:
                return xpool, torch.cat(xs, 1)
            else:
                return xpool, x
        else:
            raise NotImplementedError

    def get_embeddings(self, loader, device, is_rand_label=False):
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                if isinstance(data, list):
                    data = data[0].to(device)
                data = data.to(device)
                batch, x, edge_index = data.batch, data.x, data.edge_index
                edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None

                if x is None:
                    x = torch.ones((batch.shape[0], 1)).to(device)
                x, _ = self.forward(batch, x, edge_index, edge_weight)

                ret.append(x.cpu().numpy())
                if is_rand_label:
                    y.append(data.rand_label.cpu().numpy())
                else:
                    y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y



class Model(torch.nn.Module):
    def __init__(self, meta_info, opt, device):
        super(Model, self).__init__()
        self.name = 'GCSå'
        self.device = device
        self.warm_ratio = opt.warm_ratio
        self.inner_iter = opt.inner_iter
        self.thres = opt.thres

        self.encoder = TUEncoder(num_dataset_features=1, emb_dim=opt.emb_dim, num_gc_layers=opt.num_gc_layers, drop_ratio=opt.drop_ratio, pooling_type=opt.pooling_type)
        self.proj_head = Sequential(Linear(self.encoder.out_graph_dim, opt.emb_dim), ReLU(inplace=True), Linear(opt.emb_dim, opt.emb_dim))

        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def _contrastive_score(self, query, key, queues):
        eye = torch.eye(query.size(0)).type_as(query)
        pos = torch.einsum('nc,nc->n', [query, key]).unsqueeze(0)
        neg = torch.cat([torch.einsum('nc,kc->nk', [query, queue]) * (1 - eye) for queue in queues], dim=1)
        score = (pos.exp().sum(dim=1) / neg.exp().sum(dim=1)).log()
        return score
    

    def _compute_cam(self, feature, score, batch, clamp_negative_weights=True):


        # feature (batch_nodes, embedding_dim)
        grad = torch.autograd.grad(score.sum(), feature)[0]

        # (batch_nodes, 1)
        weight = torch.mean(grad, dim=-1, keepdim=True)

        if clamp_negative_weights:  # positive weights only
            weight = weight.clamp_min(0)

        # (batch_nodes, 1)
        cam = torch.sum(weight * feature, dim=1, keepdim=True).detach()

        normalized_cam = normalize(cam, batch).squeeze().detach()
        return normalized_cam


    def get_features(self, batch, x, edge_index, edge_attr, edge_weight, keep_node):
        if keep_node is not None:
            edge_index, edge_attr, edge_mask = subgraph(keep_node, edge_index, edge_attr, return_edge_mask=True)
            edge_weight = torch.masked_select(edge_weight, edge_mask.to(edge_weight.device))
            new_x = torch.zeros(x.shape).to(x.device)
            new_x[keep_node] = x[keep_node]
            x = new_x

        z, node_emb = self.encoder(batch, x, edge_index, edge_attr, edge_weight)
        return z, node_emb
    
    def get_projection(self, z):
        return self.proj_head(z)

    def get_contrastive_cam(self, batch, n_iters=1, return_intermediate=False):
        key, queues = None, []
        _masks, _masked_images = [], []

        mask_edge = torch.zeros(batch.edge_index.shape[1]) + 1e-10
        keep_indice = torch.arange(batch.x.shape[0])
        mask_edge_list = []
        keep_node_list = []

        for it in range(n_iters):
            z, node_emb = self.get_features(batch.batch, batch.x, batch.edge_index, None, (1 - mask_edge).to(self.device), keep_indice)
            output = self.get_projection(z)

            if it == 0:
               key = output  # original graph
            # queues.append(output.detach())  # masked images

            # score = self._contrastive_score(output, key, queues)
            score = self.calc_loss(key, output)
            
            # (batch_nodes, 1)
            node_cam = self._compute_cam(node_emb, score, batch.batch, clamp_negative_weights=True)
            mask_node = torch.max(mask_node, node_cam) if it > 0 else node_cam
            mask_node = mask_node.detach()
            indicater = torch.where(mask_node > self.thres, 1, 0)
            keep_node_list.append(indicater)

            src, dst = batch.edge_index[0], batch.edge_index[1]
            # batch_edge, 1
            edge_cam = (node_cam[src] + node_cam[dst]) /2 
            mask_edge = torch.max(mask_edge, edge_cam) if it > 0 else edge_cam
            mask_edge = mask_edge.detach()
            edge_indicater = torch.where(mask_edge > self.thres, 1, 0)
            mask_edge_list.append(edge_indicater)

        return keep_node_list, mask_edge_list





    def contrast_train(self, batch, keep_node=None, mask_edge=None):
        if keep_node is None and mask_edge is None:
            aug = permute_edges(deepcopy(batch).cpu()).to(self.device)
            z_aug, _ = self.get_features(aug.batch, aug.x, aug.edge_index, None, None, None)
            x_aug = self.get_projection(z_aug)
        else:
            z_aug, _ = self.get_features(batch.batch, batch.x, batch.edge_index, None, mask_edge, keep_node)
            x_aug = self.get_projection(z_aug)

        z, _ = self.get_features(batch.batch, batch.x, batch.edge_index, None, None, None)
        x = self.get_projection(z)

        contrast_loss = self.calc_loss(x, x_aug)

        return contrast_loss



    def positive(self, batch, indicater, mask_edge):
        # view1
        env_indicator = indicater.new_ones(indicater.shape) * 0.5
        env_indicator = torch.bernoulli(env_indicator)
        keep_node = torch.nonzero(indicater + env_indicator, as_tuple=False).view(-1,)

        edge_env_indicator = mask_edge.new_ones(mask_edge.shape) * 0.5
        edge_env_indicator = torch.bernoulli(edge_env_indicator)
        new_mask_edge = mask_edge + edge_env_indicator
        new_mask_edge = new_mask_edge.clamp_max(1)

        new_mask_edge = new_mask_edge.to(self.device)
        keep_node = keep_node.to(self.device)

        z_aug, _ = self.get_features(batch.batch, batch.x, batch.edge_index, None, new_mask_edge, keep_node)
        x_aug_1 = self.get_projection(z_aug)


        # view2
        env_indicator = indicater.new_ones(indicater.shape) * 0.5
        env_indicator = torch.bernoulli(env_indicator)
        keep_node = torch.nonzero(indicater + env_indicator, as_tuple=False).view(-1,)

        edge_env_indicator = mask_edge.new_ones(mask_edge.shape) * 0.5
        edge_env_indicator = torch.bernoulli(edge_env_indicator)
        new_mask_edge = mask_edge + edge_env_indicator
        new_mask_edge = new_mask_edge.clamp_max(1)

        new_mask_edge = new_mask_edge.to(self.device)
        keep_node = keep_node.to(self.device)

        z_aug, _ = self.get_features(batch.batch, batch.x, batch.edge_index, None, new_mask_edge, keep_node)
        x_aug_2 = self.get_projection(z_aug)

        contrast_loss = self.calc_loss(x_aug_1, x_aug_2)

        return contrast_loss


    def negative(self, batch, indicater, mask_edge):
        # view1
        indicater = 1 - indicater
        env_indicator = indicater.new_ones(indicater.shape) * 0.1
        env_indicator = torch.bernoulli(env_indicator)
        keep_node = torch.nonzero(indicater + env_indicator, as_tuple=False).view(-1,)

        mask_edge = 1 - mask_edge
        edge_env_indicator = mask_edge.new_ones(mask_edge.shape) * 0.1
        edge_env_indicator = torch.bernoulli(edge_env_indicator)
        new_mask_edge = mask_edge + edge_env_indicator
        new_mask_edge = new_mask_edge.clamp_max(1)

        new_mask_edge = new_mask_edge.to(self.device)
        keep_node = keep_node.to(self.device)

        z_aug, _ = self.get_features(batch.batch, batch.x, batch.edge_index, None, new_mask_edge, keep_node)
        x_aug = self.get_projection(z_aug)

        z, _ = self.get_features(batch.batch, batch.x, batch.edge_index, None, None, None)
        x = self.get_projection(z)

        contrast_loss = self.calc_loss(x, x_aug)

        return contrast_loss

    @staticmethod
    def calc_loss(x, x_aug, temperature=0.2, sym=True):
        # x and x_aug shape -> Batch x proj_hidden_dim

        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        if sym:

            loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
            loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)

            loss_0 = - torch.log(loss_0).mean()
            loss_1 = - torch.log(loss_1).mean()
            loss = (loss_0 + loss_1)/2.0
        else:
            loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
            loss_1 = - torch.log(loss_1).mean()
            return loss_1

        return loss

    def reg_mask(self, mask, batch, size):

        key_num = scatter_add(mask, batch, dim=0, dim_size=size)
        env_num = scatter_add((1 - mask), batch, dim=0, dim_size=size)
        non_zero_mask = scatter_add((mask > 0).to(torch.float32), batch, dim=0, dim_size=size) 
        all_mask = scatter_add(torch.ones_like(mask).to(torch.float32), batch, dim=0, dim_size=size)
        non_zero_ratio = non_zero_mask / (all_mask + 1e-8)
        return key_num + 1e-8, env_num + 1e-8, non_zero_ratio

    def forward(self, batch, progress):
        if progress < self.warm_ratio:
            return self.contrast_train(batch)
        else:
            keep_node_list, mask_edge_list = self.get_contrastive_cam(batch, self.inner_iter)
            indicater = keep_node_list[-1]
            mask_edge = mask_edge_list[-1]

            pos_score = self.positive(batch, indicater, mask_edge)
            neg_score = self.negative(batch, indicater, mask_edge)

            return pos_score - 0.1 * neg_score
