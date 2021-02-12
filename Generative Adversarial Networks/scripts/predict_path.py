import numpy as np
import torch
import json
import argparse
import time

from model import define_D, define_G
from datasets_3d import importMaze, map_values_back

def get_path(path, maze, start_cell, target_cell):
    height, width, depth = list(maze.shape)
    start_y, start_z, start_x = start_cell
    goal_y, goal_z, goal_x = target_cell

    success = True
    visited_cells = np.full_like(maze, False)
    final_path = []
    neighbors = [(0, 0, 1),(0, 0, -1),
                (0, 1, 0),(0, -1, 0),
                (1, 0, 0),(-1, 0, 0)]

    # loop as long as we are not yet at the goal
    while not (start_x == goal_x and start_y == goal_y and start_z == goal_z):
        # add current voxel to path
        final_path += [start_x, start_y, start_z]
        visited_cells[start_y][start_z][start_x] = True
        nextPathVoxel = None

        # look at all the voxel around you
        for dx, dy, dz in neighbors:
            x, y, z = start_x + dx, start_y + dy, start_z + dz

            # skip if voxel out of bounds
            if x < 0 or y < 0 or z < 0 or x >= width or y >= height or z >= depth:
                continue

            # check if voxel belongs to the path
            if visited_cells[y, z, x] == False and path[y, z, x] == 9 and maze[y, z, x] != 11:
                nextPathVoxel = (y, z, x)

        # another voxel for the path was found
        if nextPathVoxel:
            # mark this voxel as visited and set coordinates as current voxel
            start_y, start_z, start_x = nextPathVoxel

        # no new path voxel was found and no complete path was found
        else:

            # no path after the start voxel -> quit
            if final_path == [start_x, start_y, start_z]:
                success = False
                break

            # take one step back and look around in case of small loop
            path[start_y, start_z, start_x] = 12
            final_path = final_path[:-3]
            start_x, start_y, start_z = final_path[-3:]

            for dx, dy, dz in neighbors:
                x, y, z = start_x + dx, start_y + dy, start_z + dz

                # skip if voxel out of bounds
                if x < 0 or y < 0 or z < 0 or x >= width or y >= height or z >= depth:
                    continue

                # check if voxel belongs to the path
                if visited_cells[y, z, x] == False and path[y, z, x] == 9 and maze[y, z, x] != 11:
                    nextPathVoxel = (y, z, x)
                    start_y, start_z, start_x = nextPathVoxel
            
            if not nextPathVoxel:
                success = False
                break
    
    if (success):
        final_path += [goal_x, goal_y, goal_z]

    print(final_path)
    return success, final_path, path

def get_path_voxel(maze_with_partial_path):
    path_voxel_indices = torch.where(maze_with_partial_path == 9)
    y, z, x = path_voxel_indices
    indices = []
    for index in range(0, len(x)):
        indices.append(x[index].item())
        indices.append(y[index].item())
        indices.append(z[index].item())
    return indices

def get_prediction(json_path, num_dim, generator_path):
    torch.set_printoptions(threshold=5000)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load generator
    generator = define_G(1, 5, 20, 3, norm='instance', init_type='orthogonal', init_gain=0.02, 
                        n_blocks=10, activation='LeakyReLU').to(device)
    generator.load_state_dict(torch.load(generator_path))

    # read in maze
    maze, path, start, end = importMaze(json_path)

    # get image in correct torch array form
    maze_tensor = (torch.from_numpy(maze).type(torch.FloatTensor))
    maze_tensor = maze_tensor.unsqueeze(0).unsqueeze(0)

    # get array with predicted path
    output = generator(maze_tensor)
    _, prediction = torch.max(output, 1, keepdim=True)

    # map values back from 0-5 to 9-14
    map_values_back(prediction)

    # prepare path output format
    map_values_back(maze)
    prediction = prediction.squeeze(0).squeeze(0)

    success, path, predicition = get_path(prediction, maze, start, end)

    path_voxel = []
    if not success:
        print('not sucessful')
        path_voxel = get_path_voxel(prediction)
    return path, path_voxel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_dim', type=int, default=3, help='Height dimension.')
    parser.add_argument('--level_path', type=str, default='./results/', help='Path to level.')
    parser.add_argument('--generator_path', type=str, default='./results/generator.pt', help='Path to the correct generator.')

    parsed_args = parser.parse_args()

    # define cuda device on which tensors and modules are run
    torch.set_printoptions(threshold=5000)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train(parsed_args.num_filter, parsed_args.channels, parsed_args.num_classes,
          parsed_args.batch_size, num_dim, parsed_args.dataset_dir, parsed_args.results_dir,
          parsed_args.epoch_count, parsed_args.number_of_epochs)

    start_time = time.time() * 1000

    path, path_voxel = get_prediction(parsed_args.level_path, parsed_args.num_dim, parsed_args.generator_path)

    end_time = time.time() * 1000
    duration = end_time - start_time

    with open(json_path, 'r') as json_file:
        json_data = json.load(json_file)
        json_data['resultPath'] = path
        json_data['resultTime'] = duration
        json_data['resultVoxel'] = path_voxel

    with open(json_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=2)

