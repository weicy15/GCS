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
from torch import optim, nn

from model.contra_cam_v7_for_transfer import Model
from cam_util.chem_gnn import GNN_graphpred
from cam_util.utils import initialize_edge_weight
from util.utils import scaffold_split
import pandas as pd
from sklearn.metrics import roc_auc_score


def arg_parse():
    str2bool = lambda x: x.lower() == "true"
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')

    parser.add_argument('--dataset_name', type=str, default='bbbp', help='dataset name')
    parser.add_argument('--dataset_root', type=str, default='storage/datasets', help='dataset dir')

    parser.add_argument('--cuda_device', type=str, default='4')
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)

    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--lr_scale', type=float, default=1, help='relative learning rate for the feature extraction layer (default: 1)')

    parser.add_argument('--lr_decay', type=int, default=30)
    parser.add_argument('--lr_gamma', type=float, default=0.1)
    parser.add_argument('--lr_scheduler', type=str, default='none', help='cos, step')
    parser.add_argument('--milestones', nargs='+', type=int, default=[40,60,80])

    parser.add_argument('--warm_ratio', type=float, default=0.1, help='Number epochs to start cam contrast')
    parser.add_argument('--inner_iter', type=int, default=3, help='Number epochs to start cam contrast')
    parser.add_argument('--num_gc_layers', type=int, default=5, help='Number of GNN layers before pooling')
    parser.add_argument('--emb_dim', type=int, default=300)
    parser.add_argument('--drop_ratio', type=float, default=0.5, help='Dropout Ratio / Probability')
    parser.add_argument('--thres', type=float, default=0.5, help='0 to 1 for controlling the node to drop')
    parser.add_argument('--JK', type=str, default="last", help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--graph_pooling', type=str, default="mean", help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--split', type=str, default="scaffold", help="random or scaffold or random_scaffold")

    parser.add_argument('--note',  type=str, default='finetune', help='note to record')

    parser.add_argument('--trails', type=int, default=1, help='number of runs (default: 0)')   
    parser.add_argument('--seed', type=int, default=618)

    parser.add_argument("--loadFilename", type=str, default=None)
    parser.add_argument('--input_model_file', type=str, default='data/contra_cam_v7_for_transfer/Dec11_21-30-24__pretrain/latest.tar', help='filename to read the pretrain model (if there is any)')

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
    dataset = MoleculeDataset(opt.dataset_root + "/transfer_dataset/"+opt.dataset_name, dataset=opt.dataset_name)

    if opt.split == "scaffold":
        smiles_list = pd.read_csv(opt.dataset_root + '/transfer_dataset/' + opt.dataset_name + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
    else:
        raise ValueError("Invalid split option.")

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    meta_info = Munch()
    meta_info.dataset_type = 'molecule'
    meta_info.model_level = 'graph'


    return train_loader, val_loader, test_loader, dataset, meta_info



def model_eval(model, loader, device):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return sum(roc_list)/len(roc_list) #y_true.shape[1]


def train(opt):

    if opt.dataset_name == "tox21":
        num_tasks = 12
    elif opt.dataset_name == "hiv":
        num_tasks = 1
    elif opt.dataset_name == "pcba":
        num_tasks = 128
    elif opt.dataset_name == "muv":
        num_tasks = 17
    elif opt.dataset_name == "bace":
        num_tasks = 1
    elif opt.dataset_name == "bbbp":
        num_tasks = 1
    elif opt.dataset_name == "toxcast":
        num_tasks = 617
    elif opt.dataset_name == "sider":
        num_tasks = 27
    elif opt.dataset_name == "clintox":
        num_tasks = 2
    else:
        raise ValueError("Invalid dataset name.")

############

    train_loader, val_loader, test_loader, dataset, meta_info = load_data(opt)

    device = torch.device("cuda:{0}".format(opt.cuda_device))
    model = GNN_graphpred(opt.num_gc_layers, opt.emb_dim, num_tasks, JK=opt.JK, drop_ratio=opt.drop_ratio, graph_pooling=opt.graph_pooling, gnn_type=opt.gnn_type)
    model = model.to(device)
    

    if not opt.input_model_file == "":
        checkpoint = torch.load(opt.input_model_file)
        sd = checkpoint['sd']
        full_model = Model(meta_info, opt, device)
        full_model.load_state_dict(sd)
        # model.from_pretrained(full_model.encoder.state_dict())
        model.gnn.load_state_dict(full_model.encoder.state_dict())
        model.name = full_model.name + '_finetune'

    model_param_group = []
    model_param_group.append({"params": model.gnn.parameters()})
    if opt.graph_pooling == "attention":
        model_param_group.append({"params": model.pool.parameters(), "lr": opt.lr * opt.lr_scale})
    model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr": opt.lr * opt.lr_scale})
    optimizer = optim.Adam(model_param_group, lr=opt.lr, weight_decay=opt.weight_decay)

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

    criterion = nn.BCEWithLogitsLoss(reduction="none")##########

    stop_manager = EarlyStopping(directory, patience=100)
    for epoch in range(start_epoch, opt.epochs):
        model.train()
        
        show = int(float(len(train_loader)) / 2.0)
        with tqdm(total=len(train_loader), desc="epoch"+str(epoch)) as pbar:
            for index, batch in enumerate(train_loader):

                batch = batch.to(device)
                model.train()

                pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                y = batch.y.view(pred.shape).to(torch.float64)
                is_valid = y ** 2 > 0
                # Loss matrix
                loss_mat = criterion(pred.double(), (y + 1) / 2)
                # loss matrix after removing null target
                loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
                loss = torch.sum(loss_mat) / torch.sum(is_valid)


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(1)
                if index % show == 0:
                    print("Train Iter:[{:<3}/{}], Model_Loss:[{:.4f}]".format(index, len(train_loader), loss))

        scheduler.step()


        val_score = model_eval(model, val_loader, device)
        test_score = model_eval(model, test_loader, device)
        print("Epoch:[{}/{}], valid:[{:.8f}]".format(epoch, opt.epochs, test_score))

        save_dic = {
                'epoch': epoch,
                'sd': model.state_dict(),
                'opt': optimizer.state_dict(),
                'sche': scheduler.state_dict(),
            }
        stop_manager(val_score, save_dic)
        if stop_manager.early_stop:
            print("Early stopping")
            break


    ####### final test ###########
    val_score = model_eval(model, val_loader, device)
    test_score = model_eval(model, test_loader, device)
    print("Final_test {}".format(test_score))
    
    best_checkpoint = torch.load(os.path.join(
        directory + '/buffer', '{}_{}.tar'.format(stop_manager.best_score, 'checkpoint')))
    torch.save({
        'sd': best_checkpoint['sd'],
    }, os.path.join(directory, 'best_{}.tar'.format(stop_manager.best_score)))
    torch.save({
        'sd': model.state_dict(),
    }, os.path.join(directory, 'latest.tar'))
    
    with open(os.path.join(directory, 'model_arg.json'), 'wt') as f:
        json.dump(vars(opt), f, indent=4)

    shutil.rmtree(directory + '/buffer')


    ######## best test ############
    model.load_state_dict(best_checkpoint['sd'])
    val_score = model_eval(model, val_loader, device)
    test_score = model_eval(model, test_loader, device)

    print("Best_test {}".format(test_score))
    return stop_manager.best_score

if __name__ == "__main__":
    opt = arg_parse()
    set_seed(opt.seed)
    # opt.input_model_file
    total = []
    for _ in range(opt.trails):
        test_score = train(opt)
        total.append(test_score)

    print(total)
    print(opt)