import optuna
import os
import time
import json
import logging
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from training import calculate_gradient_penalty
from datasets import ImageDataset
from model import define_D, define_G, get_scheduler, GANLoss, update_learning_rate

def objective(trial):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataset_dir = './training_data/10x3x10_1path/'
    index = trial.number
    
    # Generator values
    num_filter = trial.suggest_categorical('num_filter', [10, 15, 20, 25, 30])
    weight_init_type = trial.suggest_categorical('weight_init_type', ['normal', 'xavier', 'orthogonal'])
    norm_type = trial.suggest_categorical('norm_type', ['batch', 'instance'])
    activation = trial.suggest_categorical('activation', ['ReLU', 'LeakyReLU'])
    n_blocks = trial.suggest_categorical('n_blocks', [5, 7, 10, 12, 15])
    
    # Optimizer values
    lr = trial.suggest_categorical('lr_g', [0.0001, 0.0002, 0.0003, 0.0004])
    beta1 = trial.suggest_categorical('beta1_g', [0.4, 0.5, 0.6, 0.7])

    # learning rate policy
    lr_policy = trial.suggest_categorical('lr_policy', ['lambda', 'step', 'cosine'])
    
    # training values
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    training_epochs = trial.suggest_categorical('training_epochs', [100, 150, 200])
    
    # define Generator und Discriminator
    net_g = define_G(1, 5, num_filter, 3, norm=norm_type, init_type=weight_init_type, init_gain=0.02, 
                            n_blocks=n_blocks, activation=activation).to(device)
    net_d = define_D(1, num_filter, norm=norm_type, init_type=weight_init_type).to(device)

    # define optimizer
    optimizer_g = optim.Adam(net_g.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_d = optim.Adam(net_d.parameters(), lr=lr, betas=(beta1, 0.999))

    # training
    trained_generator, trained_discriminator = train(batch_size, training_epochs, 
                                net_g, net_d, optimizer_g, optimizer_d, lr_policy, device, dataset_dir)
    
    # validation and calculate accuracy
    avg_mse_eval = evaluate(dataset_dir, trained_generator)
    
    # save checkpoints
    index_directory = './results/' + str(index) + '/'
    os.makedirs(index_directory, exist_ok=True)
    
    torch.save(trained_generator.state_dict(), index_directory + 'generator.pt')
    torch.save(trained_discriminator.state_dict(), index_directory + 'discriminator.pt')
    
    json_path_params = index_directory + 'params.json'
    
    params_dict = {}
    params_dict['accuracy'] = avg_mse_eval
    for key, value in trial.params.items():
         params_dict[key] = value
            
    with open(json_path_params, 'w') as outfile:
        json.dump(params_dict, outfile, indent=4, separators=(',', ': '))
    
    return avg_mse_eval



def train(batch_size, epochs, net_g, net_d, optimizer_g, optimizer_d, lr_policy, device, dataset_dir):

    channels = 1
    num_classes = 5
    result_folder = './results/'
    epoch_count = 1
    niter = epochs
    niter_decay = 100
    lr_decay_iters = 50


    os.makedirs(result_folder, exist_ok=True)

    # Dataset loader
    training_data_loader = DataLoader(ImageDataset(dataset_dir),
                                      batch_size=batch_size, shuffle=True)
    testing_data_loader = DataLoader(ImageDataset(dataset_dir, mode='val'),
                                     batch_size=1, shuffle=True, num_workers=1)
    
    # loss functions
    criterionGAN = GANLoss().to(device)
    criterionMSE = nn.MSELoss().to(device)
    weight = torch.FloatTensor([1, 1, 1, 1, 1]).to(device)
    criterionCE = nn.CrossEntropyLoss(weight=weight).to(device)

    net_g_scheduler = get_scheduler(optimizer_g, lr_policy, niter, niter_decay, lr_decay_iters)
    net_d_scheduler = get_scheduler(optimizer_d, lr_policy, niter, niter_decay, lr_decay_iters)

    start_time = time.time()
    for epoch in range(epoch_count, niter + niter_decay + 1):

        ######################
        # train
        ######################

        for iteration, batch in enumerate(training_data_loader, 1):
            real_maze, real_maze_with_path, real_path = batch[0].to(device), batch[1].to(device), batch[2].to(device)

            maze_pred_path = net_g(real_maze).to(device)
            fake_b = torch.max(maze_pred_path, 1, keepdim=True)[1].float()

            fake_path = torch.where((fake_b == 0), torch.zeros_like(fake_b),
                                    torch.ones_like(fake_b))

            ######################
            # (1) Update D network
            ######################

            optimizer_d.zero_grad()

            pred_fake = net_d.forward(fake_path)
            loss_d_fake = criterionGAN(pred_fake, False)

            pred_real = net_d.forward(real_path)
            loss_d_real = criterionGAN(pred_real, True)

            loss_d = (loss_d_fake + loss_d_real) * 0.5
            loss_d.backward()

            gradient_penalty = calculate_gradient_penalty(net_d, real_maze.data, real_path.data, fake_path.data)
            gradient_penalty.backward()

            optimizer_d.step()

            ######################
            # (2) Update G network
            ######################

            optimizer_g.zero_grad()

            pred_fake = net_d.forward(fake_path)
            loss_g_gan = criterionGAN(pred_fake, True)

            loss_g_ce = criterionCE(maze_pred_path, real_maze_with_path[:, 0, ...].long()) * 10
            loss_g = loss_g_gan + loss_g_ce
            
            loss_g.backward()

            optimizer_g.step()
        
        update_learning_rate(net_g_scheduler, optimizer_g)
        update_learning_rate(net_d_scheduler, optimizer_d)

        # test
        avg_test_mse = 0.0
        for i, batch in enumerate(testing_data_loader):
            maze, maze_with_path, only_path = batch[0].to(device), batch[1].to(device), batch[2].to(device)

            output = net_g(maze)
            prediction = torch.max(output, 1, keepdim=True)[1].float()
            predicted_path = torch.where((prediction == 0), torch.zeros_like(prediction),
                                    torch.ones_like(prediction))
            
            path_mse = criterionMSE(predicted_path, only_path)
            avg_test_mse += path_mse.item()

        print('===> Avg. MSE: {:.4f}'.format(avg_test_mse / len(testing_data_loader)))
    
    end_time = time.time()
    training_duration = end_time - start_time
    print('===> Total time: %dmin' % (training_duration / 60))
    
    return net_g, net_d
    
def evaluate(dataset_dir, trained_generator):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    criterionMSE = nn.MSELoss().to(device)
    val_data_loader = DataLoader(ImageDataset(dataset_dir, mode='eval'),
                             batch_size=1, shuffle=True, num_workers=1)

    avg_path_mse = 0.0
    for i, batch in enumerate(val_data_loader):
        maze, maze_with_path, only_path = batch[0].to(device), batch[1].to(device), batch[2].to(device)

        output = trained_generator(maze)
        prediction = torch.max(output, 1, keepdim=True)[1].float()
        prediction_path = torch.where((prediction == 0), torch.zeros_like(prediction),
                            torch.ones_like(prediction))

        path_mse = criterionMSE(prediction_path, only_path)
        avg_path_mse += path_mse.item()

    return avg_path_mse / len(val_data_loader)




if __name__ == "__main__":
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "imi-gan-pathfinding"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, direction='minimize')
    study.optimize(objective, n_trials=10)
    
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))