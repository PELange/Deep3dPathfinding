import os
import numpy as np
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets_3d import ImageDataset
from model import define_G

def evaluate_generator(data_loader, net_g, result_folder):
    criterionMSE = nn.MSELoss().to(device)
    
    all_path_mse = {}
    for i, batch in enumerate(data_loader):
        maze, maze_with_path, only_path = batch[0].to(device), batch[1].to(device), batch[2].to(device)

        output = net_g(maze)
        prediction = torch.max(output, 1, keepdim=True)[1].float()
        prediction_path = torch.where((prediction == 0), torch.zeros_like(prediction),
                            torch.ones_like(prediction))
        
        path_mse = criterionMSE(prediction_path, only_path)
        all_path_mse[i] = path_mse.item()
    
    return all_path_mse

if __name__ == '__main__':
    torch.set_printoptions(threshold=5000)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # folder with input data
    dataset_dir = './training_data/10x3x10_1path/'
    model_folder = './results/'
    
    # filder with output data
    result_folder = './results/eval/'
    os.makedirs(result_folder, exist_ok=True)

    # create Generator and read in curretly trained weights
    generator = define_G(1, 5, 20, 3, norm='instance', init_type='orthogonal', init_gain=0.02, 
                        n_blocks=10, activation='LeakyReLU').to(device)
    generator.load_state_dict(torch.load(model_folder + 'generator.pt'))
    
    # Dataloader with evaluation data
    val_data_loader = DataLoader(ImageDataset(dataset_dir, mode='eval'),
                             batch_size=1, shuffle=True, num_workers=1)

    # evaluate generator
    all_path_mse = evaluate_generator(val_data_loader, generator, result_folder)
    
    print(all_path_mse)
            