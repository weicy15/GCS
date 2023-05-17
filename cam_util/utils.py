import copy

import numpy as np
import torch
from torch_geometric.data import Data


def initialize_edge_weight(data):
	data.edge_weight = torch.ones(data.edge_index.shape[1], dtype=torch.float)
	return data

def initialize_node_features(data):
	num_nodes = int(data.edge_index.max()) + 1
	data.x = torch.ones((num_nodes, 1))
	return data

def set_tu_dataset_y_shape(data):
	num_tasks = 1
	data.y = data.y.unsqueeze(num_tasks)
	return data

