import numpy as np
import torch
import os


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, directory, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
      
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.directory = directory
    def __call__(self, score, save_dic):

        if self.best_score is None:
            self.best_score = score
            if not os.path.exists(self.directory + '/buffer'):
                os.makedirs(self.directory + '/buffer')
            torch.save(save_dic, os.path.join(self.directory + '/buffer', '{}_{}.tar'.format(score, 'checkpoint')))

        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            if not os.path.exists(self.directory + '/buffer'):
                os.makedirs(self.directory + '/buffer')
            torch.save(save_dic, os.path.join(self.directory + '/buffer', '{}_{}.tar'.format(score, 'checkpoint')))

