import glob
import json
import torch
import numpy as np

from torch.utils.data import Dataset

# Import given json-file and convert it into a maze
def importMaze(maze_file):
    with open(maze_file) as json_file:
        data = json.load(json_file)

        # Get agent and target pos and convert them to correct dimension
        agent_pos = data['agentStart']
        target_pos = data['agentGoal']
        start_cell = (agent_pos[1], agent_pos[2], agent_pos[0])
        target_cell = (target_pos[1], target_pos[2], target_pos[0])

        # Get dimension of maze
        dimension = data['xdimensions']
        colDim = dimension[0]
        rowDim = dimension[2]
        depthDim = dimension[1]

        # Get level
        levelRaw = np.array(data['exportlevel']).astype(np.float32)
        # Reshape level to correct dimension
        maze = levelRaw.reshape((depthDim, rowDim, colDim))
        maze[start_cell] = 9
        maze[target_cell] = 9

        # get path
        pathRaw = np.array(data['djikstraPath']).astype(np.float32)
        # Reshape path coordinates
        amountOfSteps = pathRaw.size / 3
        pathCoordinates = (pathRaw.reshape(int(amountOfSteps), 3)).astype(int)

        path = np.copy(maze)
        path[pathCoordinates[:,1], pathCoordinates[:,2], pathCoordinates[:,0]] = 9

        map_values(maze)
        map_values(path)

    return maze, path, start_cell, target_cell

# map values to 0-4 for models
def map_values(tensor):
    tensor[tensor==9] = 0
    tensor[tensor==10] = 1
    tensor[tensor==11] = 2
    tensor[tensor==12] = 3
    tensor[tensor==13] = 4

# map values back to 9-14 for the output
def map_values_back(tensor):
    tensor[tensor==0] = 9
    tensor[tensor==1] = 10
    tensor[tensor==2] = 11
    tensor[tensor==3] = 12
    tensor[tensor==4] = 13

class ImageDataset(Dataset):
    def __init__(self, root, mode='train'):
        mode_folder = ''
        if mode == 'train':
            mode_folder = 'training'
        elif mode != 'eval':
            mode_folder = 'validation'
        else:
            mode_folder = 'evaluation'

        #print('%s/%s/*.json' % (root, mode_folder))
        json_maze_data = sorted(glob.glob('%s/%s/*.json' % (root, mode_folder)), key=lambda fname: fname)

        self.inp_files = []
        self.out_files = []

        for file_name in json_maze_data:
            maze, path, _, _ = importMaze(file_name)

            self.inp_files.append(maze)
            self.out_files.append(path)

    def __getitem__(self, index):
        inp_img = self.inp_files[index % len(self.inp_files)]
        out_img = self.out_files[index % len(self.out_files)]

        inp_img = (torch.from_numpy(inp_img).type(torch.FloatTensor))
        out_img = (torch.from_numpy(out_img).type(torch.FloatTensor))

        mask = torch.where((out_img == 0), torch.zeros_like(out_img), torch.ones_like(out_img))

        # inp_img: maze 3d tensor with values 0-5
        # out_img: maze 3d tensor with path and values 0-5
        # mask: shows only path (0) otherwise 1
        return inp_img.unsqueeze(0), out_img.unsqueeze(0), mask.unsqueeze(0)

    def __len__(self):
        return len(self.inp_files)