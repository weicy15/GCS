import random

import numpy as np
import torch
import argparse
from cam_util.dataset import TUDataset, TUEvaluator
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

from model.GCS import Model
from cam_util.embedding_evaluation import EmbeddingEvaluation
from sklearn.svm import LinearSVC, SVC
from cam_util.utils import initialize_edge_weight, initialize_node_features, set_tu_dataset_y_shape



def arg_parse():
    str2bool = lambda x: x.lower() == "true"
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')

    parser.add_argument('--dataset_name', type=str, default='PROTEINS', help='dataset name')
    parser.add_argument('--dataset_root', type=str, default='storage/datasets', help='dataset dir')

    parser.add_argument('--cuda_device', type=str, default='0')

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)

    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')

    parser.add_argument('--lr_decay', type=int, default=30)
    parser.add_argument('--lr_gamma', type=float, default=0.1)
    parser.add_argument('--lr_scheduler', type=str, default='none', help='cos, step')
    parser.add_argument('--milestones', nargs='+', type=int, default=[40,60,80])

    parser.add_argument('--warm_ratio', type=float, default=0, help='Number epochs to start cam contrast')
    parser.add_argument('--inner_iter', type=int, default=3, help='Number epochs to start cam contrast')
    parser.add_argument('--num_gc_layers', type=int, default=5, help='Number of GNN layers before pooling')
    parser.add_argument('--pooling_type', type=str, default='layerwise', help='GNN Pooling Type Standard/Layerwise')
    parser.add_argument('--emb_dim', type=int, default=32)
    parser.add_argument('--drop_ratio', type=float, default=0.5, help='Dropout Ratio / Probability')
    parser.add_argument('--thres', type=float, default=0.3, help='0 to 1 for controlling the node to drop')
    parser.add_argument('--downstream_classifier', type=str, default="non-linear", help="Downstream classifier is linear or non-linear")

    parser.add_argument('--note',  type=str, default='', help='note to record')

    parser.add_argument('--trails', type=int, default=10, help='number of runs (default: 0)')   
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
    my_transforms = Compose([initialize_node_features, initialize_edge_weight, set_tu_dataset_y_shape])
    dataset = TUDataset("storage/datasets", opt.dataset_name, transform=my_transforms)

    train_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
    test_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)

    meta_info = Munch()
    meta_info.dataset_type = 'real'
    meta_info.model_level = 'graph'

    meta_info.dim_node = dataset.num_node_attributes
    meta_info.dim_edge = dataset.num_edge_attributes

    return train_loader, dataset, meta_info


def model_eval(model, dataset, device, eval_type):
    model.eval()
    evaluator = TUEvaluator()
    if eval_type == "linear":
        ee = EmbeddingEvaluation(LinearSVC(dual=False, fit_intercept=True), evaluator, dataset.task_type, dataset.num_tasks,
                             device, param_search=True)
    else:
        ee = EmbeddingEvaluation(SVC(), evaluator, dataset.task_type,
                                 dataset.num_tasks,
                                 device, param_search=True)

    train_score, val_score, test_score = ee.kf_embedding_evaluation(model.encoder, dataset)
    return train_score, val_score, test_score


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

        train_score, val_score, test_score = model_eval(model, dataset, device, opt.downstream_classifier)
        print("Epoch:[{}/{}], valid:[{:.8f}]".format(epoch, opt.epochs, val_score))

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
    train_score, val_score, test_score = model_eval(model, dataset, device, opt.downstream_classifier)
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
    train_score, val_score, test_score = model_eval(model, dataset, device, opt.downstream_classifier)
    print("Best_test {}".format(test_score))
    print(dataset.name)
    print(model.name)
    return test_score

if __name__ == "__main__":
    opt = arg_parse()
    set_seed(opt.seed)
    total = []
    for _ in range(opt.trails):
        test_score = train(opt)
        total.append(test_score)
    
    print(total)
    print(opt)