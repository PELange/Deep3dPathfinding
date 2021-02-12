import argparse
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch import autograd

from datasets_3d import ImageDataset
from model import define_D, define_G, get_scheduler, GANLoss, update_learning_rate

cudnn.benchmark = True

def calculate_gradient_penalty(model, input, real_images, fake_images):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    eta = torch.FloatTensor(real_images.size(0),1,1,1,1).uniform_(0,1)
    eta = eta.expand(real_images.size(0), real_images.size(1), real_images.size(2), real_images.size(3), real_images.size(4))
    eta = eta.to(device)

    interpolated = (eta * real_images + ((1 - eta) * fake_images)).to(device)
   
    # define it to calculate gradient
    interpolated = Variable(interpolated, requires_grad=True)

    # calculate probability of interpolated examples
    prob_interpolated = model(interpolated)

    # calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                              grad_outputs=torch.ones(
                                   prob_interpolated.size()).to(device),
                              create_graph=True, retain_graph=True)[0]

    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
    return grad_penalty

def train(num_filter=32, channels=1, num_classes=5, batch_size=32, num_dim=3,
          dataset_dir='./training_data/', result_folder='./results/',
          epoch_count=1, niter=100, niter_decay=100, lr_decay_iters=50):

    os.makedirs(result_folder, exist_ok=True)

    # Dataset loader
    training_data_loader = DataLoader(ImageDataset(dataset_dir),
                                      batch_size=batch_size, shuffle=True)
    testing_data_loader = DataLoader(ImageDataset(dataset_dir, mode='val'),
                                     batch_size=1, shuffle=True, num_workers=1)

    # build generator and discriminator
    print('===> Building models')
    net_g = define_G(channels, num_classes, num_filter, num_dim, norm='instance', init_type='orthogonal', init_gain=0.02, 
                        n_blocks=10, activation='LeakyReLU').to(device)
    net_d = define_D(channels, num_filter, norm='instance', init_type='orthogonal').to(device)
    
    # cross entropy weight -> 1D tensor with weight for each num_classes
    weight = torch.FloatTensor([1, 1, 1, 1, 1]).to(device)

    # loss for Discriminator and Generator
    criterionGAN = GANLoss().to(device)

    # Generator loss based on Discriminator response
    criterionL1 = nn.L1Loss().to(device)
    # loss used for validation of testing dataset
    criterionMSE = nn.MSELoss().to(device)
    # loss for closeness to ground truth?
    criterionCE = nn.CrossEntropyLoss(weight=weight).to(device)

    # setup optimizer
    lr = 0.0002
    beta1 = 0.4
    lr_policy = 'lambda'

    optimizer_g = optim.Adam(net_g.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_d = optim.Adam(net_d.parameters(), lr=lr, betas=(beta1, 0.999))
    net_g_scheduler = get_scheduler(optimizer_g, lr_policy, niter, niter_decay, lr_decay_iters)
    net_d_scheduler = get_scheduler(optimizer_d, lr_policy, niter, niter_decay, lr_decay_iters)

    for epoch in range(epoch_count, niter + niter_decay + 1):

        ######################
        # train
        ######################
        
        for iteration, batch in enumerate(training_data_loader, 1):
            # real_maze: with obstacles, without path
            # real_maze_with_path: with obstacles, with path
            # real_path: without obstacles, with path
            real_maze, real_maze_with_path, real_path = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            
            # give generator map with obstacles and start/end but without path
            # generator returns generated image/map with its idea of a path
            maze_pred_path = net_g(real_maze).to(device)
            fake_b = torch.max(maze_pred_path, 1, keepdim=True)[1].float()

            # generate mask that contains the fake path (1), other fields are 0 
            fake_path = torch.where((fake_b == 0), torch.zeros_like(fake_b),
                                    torch.ones_like(fake_b))

            ######################
            # (1) Update D network
            ######################

            # Set gradients back to zero before backpropagation
            optimizer_d.zero_grad()

            # train with fake data
            # predict whether fake path is fake or not and calculate loss
            pred_fake = net_d.forward(fake_path)
            loss_d_fake = criterionGAN(pred_fake, False)

            # train with real data
            # predict whether real path is fake or not and calculate loss
            pred_real = net_d.forward(real_path)
            loss_d_real = criterionGAN(pred_real, True)

            # calculate mean of fake and real loss for D and do backpropagation
            loss_d = (loss_d_fake + loss_d_real) * 0.5
            loss_d.backward()

            # gradient_penalty = calculate_gradient_penalty(net_d, real_maze.data, real_maze_with_path.data, fake_b.data)
            gradient_penalty = calculate_gradient_penalty(net_d, real_maze.data, real_path.data, fake_path.data)
            gradient_penalty.backward()

            optimizer_d.step()

            ######################
            # (2) Update G network
            ######################

            # Set gradients back to zero before backpropagation
            optimizer_g.zero_grad()

            # train with fake data
            # predict whether fake path is fake or not and calculate loss
            pred_fake = net_d.forward(fake_path)
            loss_g_gan = criterionGAN(pred_fake, True)

            loss_g_ce = criterionCE(maze_pred_path, real_maze_with_path[:, 0, ...].long()) * 10 

            loss_g = loss_g_gan + loss_g_ce
            loss_g.backward()

            optimizer_g.step()
            

        update_learning_rate(net_g_scheduler, optimizer_g)
        update_learning_rate(net_d_scheduler, optimizer_d)

        # test
        avg_path_mse = 0.0
        for i, batch in enumerate(testing_data_loader):
            maze, maze_with_path, only_path = batch[0].to(device), batch[1].to(device), batch[2].to(device)

            output = net_g(maze)
            prediction = torch.max(output, 1, keepdim=True)[1].float()
            path_only_prediction = torch.where((prediction == 0), torch.zeros_like(prediction),
                                        torch.ones_like(prediction))
            
            path_mse = criterionMSE(path_only_prediction, only_path)

            avg_path_mse += path_mse.item()

        print("===> Avg. Path MSE: {:.4f}".format(avg_path_mse / len(testing_data_loader)))

        #checkpoint
        torch.save(net_g.state_dict(), result_folder + 'generator.pt')
        torch.save(net_d.state_dict(), result_folder + 'discriminator.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_filter', type=int, default=20, help='Size of the input/output grid.')
    parser.add_argument('--channels', type=int, default=1, help='Number of channels in the input image.')
    parser.add_argument('--num_classes', type=int, default=5, help='Output number of channels/classes.')
    parser.add_argument('--dataset_dir', type=str, default='./training_data/10x3x10_1path/', help='Path to the dataset with images.')
    parser.add_argument('--results_dir', type=str, default='./results/',
                        help='Where all the results/weights will be saved.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training.')
    parser.add_argument('--epoch_count', type=int, default=1,
                        help='From which epoch to start.')
    parser.add_argument('--number_of_epochs', type=int, default=200,
                        help='Number of epochs to train.')

    parsed_args = parser.parse_args()

    # define cuda device on which tensors and modules are run
    num_dim = 3
    torch.set_printoptions(threshold=5000)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train(parsed_args.num_filter, parsed_args.channels, parsed_args.num_classes,
          parsed_args.batch_size, num_dim, parsed_args.dataset_dir, parsed_args.results_dir,
          parsed_args.epoch_count, parsed_args.number_of_epochs)