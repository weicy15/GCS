import random

import numpy as np
import torch
import argparse
from datasets.transfer_mol_dataset import MoleculeDataset
from munch import Munch

from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

from early_stop import EarlyStopping
from datetime import datetime
import os
import shutil
from tqdm import tqdm
import json

import torch
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR
from torch import optim

from model.GCS_transfer import Model
from cam_util.utils import initialize_edge_weight
from util.utils import scaffold_split


def arg_parse():
    str2bool = lambda x: x.lower() == "true"
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')

    parser.add_argument('--dataset_name', type=str, default='chembl_filtered', help='dataset name')
    parser.add_argument('--dataset_root', type=str, default='storage/datasets', help='dataset dir')

    parser.add_argument('--cuda_device', type=str, default='3')
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)

    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')

    parser.add_argument('--lr_decay', type=int, default=30)
    parser.add_argument('--lr_gamma', type=float, default=0.1)
    parser.add_argument('--lr_scheduler', type=str, default='none', help='cos, step')
    parser.add_argument('--milestones', nargs='+', type=int, default=[40,60,80])

    parser.add_argument('--warm_ratio', type=float, default=0.1, help='Number epochs to start cam contrast')
    parser.add_argument('--inner_iter', type=int, default=1, help='Number epochs to start cam contrast')
    parser.add_argument('--num_gc_layers', type=int, default=5, help='Number of GNN layers before pooling')
    parser.add_argument('--emb_dim', type=int, default=300)
    parser.add_argument('--drop_ratio', type=float, default=0.0, help='Dropout Ratio / Probability')
    parser.add_argument('--thres', type=float, default=0.1, help='0 to 1 for controlling the node to drop')
    parser.add_argument('--JK', type=str, default="last", help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    # parser.add_argument('--graph_pooling', type=str, default="mean", help='graph level pooling (sum, mean, max, set2set, attention)')

    parser.add_argument('--note',  type=str, default='pretrain', help='note to record')

    parser.add_argument('--trails', type=int, default=1, help='number of runs (default: 0)')   
    parser.add_argument('--seed', type=int, default=618)

    parser.add_argument("--loadFilename", type=str, default=None)


    args = parser.parse_args()
    return args


def set_seed(seed):
    # Fix Random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.CEX = False


def directory_name_generate(model, note):
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    directory = "data/{}".format(model.name)
    directory = os.path.join(directory, current_time)
    directory = directory + '__' + note
    return directory



def load_data(opt):
    my_transforms = Compose([initialize_edge_weight])
    dataset = MoleculeDataset(opt.dataset_root + "/transfer_dataset/"+opt.dataset_name, dataset=opt.dataset_name,
                              transform=my_transforms)
    print(dataset)
    train_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    meta_info = Munch()
    meta_info.dataset_type = 'molecule'
    meta_info.model_level = 'graph'


    return train_loader, dataset, meta_info


def train(opt):
    train_loader, dataset, meta_info = load_data(opt)

    device = torch.device("cuda:{0}".format(opt.cuda_device))
    model = Model(meta_info, opt, device)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    if opt.lr_scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=opt.lr_decay, gamma=opt.lr_gamma)
    elif opt.lr_scheduler == 'multi':
        scheduler = MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.lr_gamma)
    elif opt.lr_scheduler == 'cos':
        scheduler = CosineAnnealingLR(optimizer, T_max=opt.epochs)
    else:
        scheduler = MultiStepLR(optimizer, milestones=[99999999999999], gamma=opt.lr_gamma)

    start_epoch = 0

    if opt.loadFilename != None:
        checkpoint = torch.load(opt.loadFilename)
        sd = checkpoint['sd']

        opt_sd = checkpoint['opt']

        start_epoch = checkpoint['epoch'] + 1
        scheduler_sd = checkpoint['sche']

        model.load_state_dict(sd)
        optimizer.load_state_dict(opt_sd)

        scheduler.load_state_dict(scheduler_sd)

    directory = directory_name_generate(model, opt.note)

    stop_manager = EarlyStopping(directory, patience=100)
    for epoch in range(start_epoch, opt.epochs):
        model.train()
        
        show = int(float(len(train_loader)) / 2.0)
        with tqdm(total=len(train_loader), desc="epoch"+str(epoch)) as pbar:
            for index, batch in enumerate(train_loader):

                batch = batch.to(device)
                model.train()

                loss = model(batch, epoch/opt.epochs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(1)
                if index % show == 0:
                    print("Train Iter:[{:<3}/{}], Model_Loss:[{:.4f}]".format(index, len(train_loader), loss))

        scheduler.step()

        save_dic = {
                'epoch': epoch,
                'sd': model.state_dict(),
                'opt': optimizer.state_dict(),
                'sche': scheduler.state_dict(),
            }

        stop_manager(epoch, save_dic)



    torch.save({
        'sd': model.state_dict(),
    }, os.path.join(directory, 'latest.tar'))
    
    with open(os.path.join(directory, 'model_arg.json'), 'wt') as f:
        json.dump(vars(opt), f, indent=4)

    shutil.rmtree(directory + '/buffer')


if __name__ == "__main__":
    opt = arg_parse()
    set_seed(opt.seed)
    total = []
    for _ in range(opt.trails):
        train(opt)
    
    print(total)
    print(opt)