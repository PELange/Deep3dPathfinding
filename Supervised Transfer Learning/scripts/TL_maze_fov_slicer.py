from __future__ import print_function
import os, sys, time, datetime, json, random, math

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import json
from numpy.core.fromnumeric import reshape
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
import jsbeautifier
tf.get_logger().setLevel('ERROR')
from os import walk
from tensorflow import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD , Adam, RMSprop
from collections import deque
import matplotlib.pyplot as plt
import h5py
import copy

##### Section: Defining GLOBALS
tf.compat.v1.disable_eager_execution()
opts = jsbeautifier.default_options()
opts.indent_size = 2

# Possible cell marks
barrier_mark = 11.0 # Cell where the agent cannot stand
empty_mark = 12.0 # Passable cell for the agent

INPUT_FOLDER_PATH = os.path.realpath(__file__) + '/../../inputs/mazeToRead/'
OUTPUT_FOLDER_PATH = os.path.realpath(__file__) + "/../../output/TL/mazeFOVs/"
MAZE_HASHS_FILE_PATH = os.path.realpath(__file__) + "/../../mazeHashs.h5"
TRAINING_STATS_FILE_PATH = OUTPUT_FOLDER_PATH + "fov_slicer_statistics" + ".txt"

PARTIAL_MAZE_KERNEL_WIDTH = 5 # Defines the size of the partial maze (for 3 -> partial maze is 3x3)
MAZE_HASHS = set()

##### Section: Utility functions

# Import given json-file and convert it into a maze
def importMaze(maze_file):
    with open(maze_file) as json_file:
        data = json.load(json_file)

        # Get dimension of maze
        dimension = data['xdimensions']
        colDim = dimension[0]
        rowDim = dimension[2]

        # Get seed
        seed = data['levelseed']
        
        # Get level
        levelRaw = np.array(data['exportlevel']).astype(np.float32)
        # Reshape level to correct dimension
        maze = levelRaw.reshape((rowDim, colDim))
    return maze, seed

# Create given directory if not exists
def makeDir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Get all mazes out of input folder
def getMazeFilePath(input_folder_path):
    _, _, files = next(walk(input_folder_path))
    file_paths = list()
    if len(files) <= 0:
        raise Exception("no maze file to read found")
    else:
        for file in files:
            file_paths.append(input_folder_path + file)
    return np.array(file_paths)

# Small utility for printing readable time strings:
def format_time(seconds):
    if seconds < 60:
        s = float(seconds)
        return "%.1f seconds" % (s,)
    elif seconds < 4000:
        m = seconds / 60.0
        return "%.2f minutes" % (m,)
    else:
        h = seconds / 3600.0
        return "%.2f hours" % (h,)

# Normalize prev_envstate and envstate for the neural network, so the calculations are faster
def getFovHash(fov):
    fov_copy = copy.deepcopy(fov)
    fov_copy = fov_copy.reshape((1, -1))
    mazeMin = barrier_mark
    mazeMax = empty_mark
    for i in range(len(fov_copy[0])):
        fov_copy[0][i] = int((fov_copy[0][i] - mazeMin) / (mazeMax - mazeMin))
    fov_copy = fov_copy.flatten()
    fov_str = ''
    for num in fov_copy:
        fov_str += str(int(num))
    return fov_str

# Checks if a cell not a barrier
def check_maze_cell_valid(maze, row, col):
    if maze[row][col] == barrier_mark:
        return False
    else:
        return True

# Get partial maze out of full maze
def getPartialMaze(maze, fov_center):
    fov_row, fov_col = fov_center
    partialMaze = list()
    kernel_radius = int((PARTIAL_MAZE_KERNEL_WIDTH - 1) / 2)

    # Iterate over each cell of kernel
    for row in range(-kernel_radius, kernel_radius + 1):
        for col in range(-kernel_radius, kernel_radius + 1):
            kernel_row = fov_row + row
            kernel_col = fov_col + col

            # Check if current viewed cell is in bounds
            if isInBounds(kernel_row, kernel_col):
                current_cell = maze[kernel_row, kernel_col]
                partialMaze.append(current_cell)
            # If current viewed cell is out of bounds, add barrier cell
            else:
                partialMaze.append(barrier_mark)
    
    return np.asarray(partialMaze).reshape((PARTIAL_MAZE_KERNEL_WIDTH, PARTIAL_MAZE_KERNEL_WIDTH))

# Check if cell is in bounds of maze
def isInBounds(row, col):
    return (row > -1 and row < nrows and col > -1 and col < ncols)

# Save generated partialMaze
def savePartialMaze(partialMaze, maze_name):
    maze_file_path = OUTPUT_FOLDER_PATH + maze_name + ".json"
    partRow, partCol = np.shape(partialMaze)
    partialMaze_arr = partialMaze.reshape((1, -1))
    with open(maze_file_path, "w") as maze_file:
        data = {}
        data['xdimensions'] = [partCol, 0, partRow]
        data['levelseed'] = MAZE_SEED
        data['exportlevel'] = partialMaze_arr.tolist()
        json.dump(data, maze_file, indent=3)

# Save current maze hashes
def saveMazeHash():
    if os.path.isfile(MAZE_HASHS_FILE_PATH): os.remove(MAZE_HASHS_FILE_PATH)
    with h5py.File(MAZE_HASHS_FILE_PATH, "w") as maze_hash_file:
        maze_hash_arr = np.array(list(MAZE_HASHS))
        maze_hash_file.create_dataset('maze_hashs', shape=maze_hash_arr.shape, data=maze_hash_arr)

# Load current maze hashes
def loadMazeHash():
    maze_hashs = set()
    with h5py.File(MAZE_HASHS_FILE_PATH, "r") as maze_hash_file:
        maze_hashs_data = maze_hash_file['maze_hashs']
        for hash in maze_hashs_data: maze_hashs.add(hash)
    
    return maze_hashs

##### Section: Main Function
start_time = datetime.datetime.now()
timestamp = str(time.time()).split('.')[0] # Get current timeStamp for saving files
# Create input and output folder
makeDir(INPUT_FOLDER_PATH)
makeDir(OUTPUT_FOLDER_PATH)

if not os.path.isfile(TRAINING_STATS_FILE_PATH):
    open(TRAINING_STATS_FILE_PATH, "w")

partial_mazes_invalid_count = 0
partMazes_valid_count = 0
partMazes_count = 0
MAZE_FILE_PATHS = getMazeFilePath(INPUT_FOLDER_PATH)

if os.path.isfile(MAZE_HASHS_FILE_PATH):
    MAZE_HASHS = loadMazeHash()

with open(TRAINING_STATS_FILE_PATH, "a") as outfile:
    outfile.write("----- Starting generation of partial mazes for " + str(len(MAZE_FILE_PATHS)) + " mazes -----\n")
    outfile.write( "-----------------------------------------------------------------------------------------------------------------------\n")

for maze_file_path in MAZE_FILE_PATHS:
    start_time_inner = datetime.datetime.now()
    target_count = 0

    MAZE, MAZE_SEED = importMaze(maze_file_path)
    print("maze seed: " + str(MAZE_SEED))
    print("Current full maze:")
    print(MAZE)
    nrows, ncols = np.shape(MAZE) # Get count of dimension
    partMazes_count += nrows * ncols

    # Iterate over every partialMaze of imported maze
    for col in range(0, ncols):
        for row in range(0, nrows):
            partMaze = getPartialMaze(MAZE, (row, col))
            partNrows, partNcols = np.shape(partMaze)
            maze_name = str(ncols) + "x1x" + str(nrows) + "_" + str(MAZE_SEED) + "_" + "(" + str(col) + "," + str(row) + ")"
            print("Current partial maze: " + maze_name)


            # Skip maze if center cell is a wall
            if not check_maze_cell_valid(partMaze, 2, 2):
                print("----- Maze: " + maze_name + " has a barrier in the middle, continue with next maze -----\n")
                with open(TRAINING_STATS_FILE_PATH, "a") as outfile:
                    outfile.write("----- Maze: " + maze_name + " has a barrier in the middle, continue with next maze -----\n")
                partial_mazes_invalid_count += 1
            else:
                fov_decimal_hash = int(getFovHash(partMaze), 2)
                # If maze is already in hashset, skip it
                if fov_decimal_hash in MAZE_HASHS:
                    print("Hash: " + str(fov_decimal_hash) + " is in dict, skipping this maze")
                    with open(TRAINING_STATS_FILE_PATH, "a") as outfile:
                        outfile.write("----- Partial maze: " + maze_name + " already in dataset, skipping -----\n")
                    partial_mazes_invalid_count += 1
                else:
                    print("Hash of current maze: " + str(fov_decimal_hash))
                    MAZE_HASHS.add(fov_decimal_hash)
                    savePartialMaze(partMaze, maze_name)
                    partMazes_valid_count += 1
                    with open(TRAINING_STATS_FILE_PATH, "a") as outfile:
                        outfile.write("----- Saved partial maze: " + maze_name + " -----\n")

dt = datetime.datetime.now() - start_time
t = format_time(dt.total_seconds())
# Save maze hashs
saveMazeHash()

with open(TRAINING_STATS_FILE_PATH, "a") as outfile:
    outfile.write("----- Generation of partialMazes finished for: " + str(len(MAZE_FILE_PATHS)) + " mazes with " + str(partMazes_valid_count) + " partial mazes generated in: " + t + "-----\n")
    outfile.write("----- " + str(partial_mazes_invalid_count) + " out of " + str(partMazes_count) + " partial mazes had a barrier in the middle or already exist in dataset and were skipped -----\n")
    outfile.write( "-----------------------------------------------------------------------------------------------------------------------\n")