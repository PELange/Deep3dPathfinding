from __future__ import print_function
from math import *
import os, sys, time, datetime, json, random, math
import numpy as np
import tensorflow as tf
import jsbeautifier
tf.get_logger().setLevel('ERROR')
from os import stat, walk
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD , Adam, RMSprop
from keras.layers.advanced_activations import PReLU
from collections import deque, defaultdict
import copy
tf.compat.v1.disable_eager_execution()
opts = jsbeautifier.default_options()
opts.indent_size = 2

###### Section: Defining GLOBALS

# Possible cell marks
target_mark = 9.0
agent_mark = 10.0 # Current agent position
barrier_mark = 11.0 # Cell where the agent cannot stand
empty_mark = 12.0 # Passable cell for the agent

LEFT = 0
FWD = 1 # Forward
RIGHT = 2
BWD = 3 # Backward

# Actions dictionary
actions_dict = {
    LEFT: 'left',
    FWD: 'fwd',
    RIGHT: 'right',
    BWD: 'bwd'
}

num_actions = len(actions_dict)

MODEL_INPUT_FOLDER_PATH = os.path.realpath(__file__) + '/../../model/'
MAZE_INPUT_FOLDER_PATH = os.path.realpath(__file__) + '/../../inputs/mazeToRead/'
OUTPUT_FOLDER_PATH = os.path.realpath(__file__) + "/../../output/TL/paths/"

# Tweaking factors (Hyper parameters)
MIN_REWARD_FACTOR = 0.5
PARTIAL_MAZE_KERNEL_WIDTH = 5 # Defines the size of the partial maze (for 3 -> partial maze is 3x3)
MAZE_SIZE_N_EPISODE_PERCENTILE = 1 # At how many steps should the NN stop the episode, depending on maze.size (1.0 = 100% of maze.size)

# Section: Utility functions

# Import given json-file and convert it into a maze
def importMaze(maze_file):
    with open(maze_file) as json_file:
        data = json.load(json_file)

        # Get agent and target pos and convert them to correct dimension
        agent_pos = data['agentStart']
        target_pos = data['agentGoal']
        start_cell = (agent_pos[2], agent_pos[0])
        target_cell = (target_pos[2], target_pos[0])

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
    return start_cell, target_cell, maze, seed

# Create given directory if not exists
def makeDir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Get all mazes out of input folder
def getMazeFilePaths(input_folder_path):
    _, _, files = next(walk(input_folder_path))
    file_paths = list()
    if len(files) <= 0:
        raise Exception("no maze file to read found")
    else:
        for file in files:
            file_paths.append(input_folder_path + file)
    return np.array(file_paths)

# Measure distance using manhattan distance
def manhattan_distance(p1,p2):
    return sum(abs(a-b) for a,b in zip(p1,p2))

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
        
# Utility function to make the actions more readable in output file
def format_path(action):
    actionSplit = action.split('_')
    if (int(actionSplit[0]) == 0):
        return "LEFT" + "       " + actionSplit[1]
    elif (int(actionSplit[0]) == 1):
        return "FORWARD" + "    " + actionSplit[1]
    elif (int(actionSplit[0]) == 2):
        return "RIGHT" + "      " + actionSplit[1]
    elif (int(actionSplit[0]) == 3):
        return "BACKWARD" + "   " + actionSplit[1]
    elif (int(actionSplit[0]) == 4):
        return "UP" + "         " + actionSplit[1]
    elif (int(actionSplit[0]) == 5):
        return "DOWN" + "       " + actionSplit[1]
    else:
        return "ERROR"

# Normalize prev_envstate and envstate for the neural network, so the calculations are faster
def normalize_envstate(envstate):
    mazeMin = target_mark
    mazeMax = empty_mark
    for i in range(len(envstate[0])):
        envstate[0][i] = (envstate[0][i] - mazeMin) / (mazeMax - mazeMin)
    return envstate

##### Section: Environment Class
# maze is a 2d Numpy array of floats
class Qmaze(object):
    def __init__(self, maze, agent_pos, target):
        self._maze = np.array(maze) # Maze that never changes
        self.target = target # target cell

        self.free_cells = [(r,c) for r in range(nrows) for c in range(ncols) if self._maze[r,c] == empty_mark]
        self.free_cells.remove(self.target)

        if self._maze[self.target] == 0.0:
            raise Exception("Invalid maze: target cell cannot be blocked!")
        if not agent_pos in self.free_cells:
            raise Exception("Invalid agent Location: must sit on a free cell")
        self.reset(agent_pos)

    # Reset maze to factory settings
    def reset(self, agent_pos):
        row, col = agent_pos
        self.agent = (row, col, 'start')
        self.visited = set()

    # Update envstate based on given action, or dont update envstate if the action is invalid (e.g. trying to walk through a wall)
    def update_state(self, action):        
        agent_row, agent_col, nmode = self.agent

        if self._maze[agent_row, agent_col] != barrier_mark:
            self.visited.add((agent_row, agent_col))  # Mark current cell as visited cell

        valid_actions = self.valid_actions() # Get valid actions for current agent pos
                
        # No valid action possible, agent is blocked -> no change in agent position
        if not valid_actions:
            nmode = 'blocked'
        if action in valid_actions:
            nmode = 'valid'

            if action == LEFT:
                agent_col -= 1
            elif action == FWD:
                agent_row -= 1
            if action == RIGHT:
                agent_col += 1
            elif action == BWD:
                agent_row += 1

        # Invalid action -> no change in agent position
        else:             
            nmode = 'invalid'

        # new state
        self.agent = (agent_row, agent_col, nmode)

    # Update envstate and game status according to given action
    def act(self, action, n_steps):
        self.update_state(action)
        status = self.game_status(n_steps)
        envstate = self.getFOV()
        return envstate, status

    # Get current game status
    def game_status(self, n_steps):
        # Return lose if n_episodes exceeds threshold
        if n_steps >= self._maze.size * MAZE_SIZE_N_EPISODE_PERCENTILE:
            return 'lose'
        agent_row, agent_col, _ = self.agent
        target_row, target_col = self.target
        # Return win if agent reached target
        if agent_row == target_row and agent_col == target_col:
            return 'win'

        return 'not_over'

    def getFOV(self):
        partial_maze, target_in_FOV = self.getPartialMaze()
        partial_maze = self.extend_maze_to_7x7(partial_maze)
        if not target_in_FOV:
            partial_maze = self.setTargetDirection(partial_maze)
        print("agent pos:")
        print(self.agent)
        partial_maze_final = partial_maze.reshape((1, -1))
        return partial_maze_final

    # Extend 5x5 Maze with outer lane, for all possible directions, when target is not in current FOV
    # Returns a 7x7 Maze
    def extend_maze_to_7x7(self, maze_orig):
        maze = copy.deepcopy(maze_orig)
        maze = np.vstack([[12.0, 12.0, 12.0, 12.0, 12.0], maze])
        maze = np.vstack([maze, [12.0, 12.0, 12.0, 12.0, 12.0]])
        maze = np.column_stack([[12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0], maze])
        maze = np.column_stack([maze, [12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0]])
        return maze

    # Get manhattan distance and direction vector to target
    def setTargetDirection(self, partial_maze):
        agent_row, agent_col, _ = self.agent
        tar_row, tar_col = self.target
        row_index = -1
        col_index = -1
        row_diff = tar_row - agent_row
        col_diff = tar_col - agent_col

        # Case target is under agent (row_diff is positive)
        if row_diff >= 0:
            if row_diff >= 3:
                row_index = 6
            else:
                row_index = 3 + row_diff
        # Case target is above agent (row_diff is negative)
        elif row_diff < 0:
            if row_diff <= -3:
                row_index = 0
            else:
                row_index = 3 + row_diff # + Because row_diff is negative

        
        # Case target is to the right of agent (col_diff is positive)
        if col_diff >= 0:
            if col_diff >= 3:
                col_index = 6
            else:
                col_index = 3 + col_diff
        # Case target is to the left of agent (col_diff is negative)
        elif col_diff < 0:
            if col_diff <= -3:
                col_index = 0
            else:
                col_index = 3 + col_diff # + Because col_diff is negative

        partial_maze[row_index][col_index] = target_mark

        return partial_maze

    # Get partial maze around current agent position
    def getPartialMaze(self):
        agent_row, agent_col, _ = self.agent
        target_row, target_col = self.target
        partialMaze = list()
        kernel_radius = int((PARTIAL_MAZE_KERNEL_WIDTH - 1) / 2)
        target_in_FOV = False

        # Iterate over each cell of kernel
        for row in range(-kernel_radius, kernel_radius + 1):
            for col in range(-kernel_radius, kernel_radius + 1):
                kernel_row = agent_row + row
                kernel_col = agent_col + col

                # Check if current viewed cell is in bounds
                if self.isInBounds(kernel_row, kernel_col):
                    if kernel_row == target_row and kernel_col == target_col:
                        current_cell = target_mark
                        target_in_FOV = True
                    else:
                        current_cell = self._maze[kernel_row, kernel_col]

                    partialMaze.append(current_cell)
                # If current viewed cell is out of bounds, add barrier cell
                else:
                    partialMaze.append(barrier_mark)
        
        return np.asarray(partialMaze).reshape((PARTIAL_MAZE_KERNEL_WIDTH, PARTIAL_MAZE_KERNEL_WIDTH)), target_in_FOV

    # Check if cell is in bounds
    def isInBounds(self, row, col):
        return (row > -1 and row < nrows and col > -1 and col < ncols)

    # Calculate all valid actions for given envstate
    def valid_actions(self):
        # Get agent position
        row, col, _ = self.agent

        actions = [0, 1, 2, 3]
        # Out of bounds checks

        # Remove Forward
        if row == 0:
            actions.remove(1)
        # Remove Backward
        elif row == nrows-1:
            actions.remove(3)

        # Remove Left
        if col == 0:
            actions.remove(0)
        # Remove Right
        elif col == ncols-1:
            actions.remove(2)

        # Remove Left
        if col > 0 and self._maze[row,col-1] == barrier_mark:
            actions.remove(0)
        # Remove Forward
        if row > 0 and self._maze[row-1,col] == barrier_mark:
            actions.remove(1)
        # Remove Right
        if col < ncols-1 and self._maze[row,col+1] == barrier_mark:
            actions.remove(2)
        # Remove Backward
        if row < nrows-1 and self._maze[row+1,col] == barrier_mark:
            actions.remove(3)             

        return actions

##### Section: Main Class
class Main(object):
    def __init__(self, model, **opt):
        self.weights_file = opt.get('weights_file', "")

        self.actions = list() # List to save the final actions that represent the best found path
        self.coords = []
        self.model = model

        # If you want to continue training from a previous model,
        # just supply the h5 file name to weights_file option
        if self.weights_file:
            print("loading weights from file: %s" % (self.weights_file,))
            self.model.load_weights(self.weights_file)
    
    def predict(self, qmaze):
        start_time = datetime.datetime.now()
        # Clear action and coord list if predicting paths for multiples mazes
        self.actions.clear()
        self.coords.clear()
        # Append start cell
        self.coords.append(START_CELL[1])
        self.coords.append(0)
        self.coords.append(START_CELL[0])
        timestamp = str(time.time()).split('.')[0] # Get current timeStamp for saving files
        path_length = 0

        # Run the path finding algorithm
        path_found = False
        qmaze.reset(START_CELL)
        game_over = False

        # get initial envstate (1d flattened canvas)
        envstate = qmaze.getFOV()
        # Normalize initial envstate
        envstate = normalize_envstate(envstate)

        n_steps = 0
        # Start the game loop
        while not game_over:
            # Get valid actions for current envstate
            valid_actions = qmaze.valid_actions()
            if not valid_actions: break
            prev_envstate = envstate
            # Get next action                
            action = self.predictNextAction(prev_envstate)

            # Apply action, and get new envstate
            envstate, game_status = qmaze.act(action, n_steps)
            # Set break condition if game is over
            if game_status == 'win':
                path_found = True
                game_over = True
            elif game_status == 'lose':
                game_over = True
            else:
                game_over = False

            # Normalize next envstate
            envstate = normalize_envstate(envstate)

            # Only add action to current actions list if it is a valid action
            agent_row, agent_col, _ = qmaze.agent
            agent_coords = "[" + str(agent_col) + ",0," + str(agent_row) + "]"

            if (action in valid_actions):
                self.actions.append(str(action) + "_VALID; " + agent_coords)
                self.coords.append(agent_col)
                self.coords.append(0)
                self.coords.append(agent_row)
                path_length += 1
            else:
                self.actions.append(str(action) + "_INVALID; " + agent_coords)

            n_steps += 1
        
        dt = datetime.datetime.now() - start_time
        predict_time = dt.total_seconds() * 1000.0

        self.save_files(path_found, timestamp, maze_file_path, predict_time)

        if path_found:
            print("Path found with length: " + str(len(self.actions)))
        else:
            print("No path found after " + str(n_steps) + " steps")
        
        return predict_time

        
    # Calculate the next step based on the epsilon greedy policy
    # Either random action (explore) or best action (exploit) depending on epsilon
    def predictNextAction(self, state):
        state = state.reshape((1, 7, 7, 1))
        qValues = self.model.predict(state)
        print(qValues)
        return np.argmax(self.model.predict(state)[0])
    
    # Save trained model weights and architecture, this will be used by the visualization code
    def save_files(self, path_found, timestamp, maze_file_path, predict_time):
        path_file = OUTPUT_FOLDER_PATH + timestamp + "_" + INPUT_FILE_NAME + "_path_info" + ".txt"
        coord_file = OUTPUT_FOLDER_PATH + timestamp + "_" + INPUT_FILE_NAME + "_path" + ".json"

        # Output Data of path into file
        with open(path_file, "w") as outfile:
            outfile.write("Maze Size: " + str(ncols) + "x1x" + str(nrows))
            outfile.write("\nPath Length: " + str(len(self.actions)))
            outfile.write("\nGame Status: " + str(path_found))
            outfile.write("\nTime needed to predict in milliseconds: " + str(predict_time))
            outfile.write("\nStart:" + "[" + str(START_CELL[1]) + ",0," + str(START_CELL[0]) + "]")
            outfile.write("\nTarget:" + "[" + str(TARGET_CELL[1]) + ",0," + str(TARGET_CELL[0]) + "]")
            outfile.write("\nFinal best path:\n")

            for action in self.actions:
                action = format_path(action)
                outfile.write("%s\n" % action)

        # Output coords of path in seperate file for unity
        if maze_file_path is not None:
            with open(maze_file_path, "r") as import_file, open(coord_file, "w") as export_file:
                import_data = json.load(import_file)
                export_data = dict(import_data)
                export_data['resultPath'] = self.coords
                export_data['resultTime'] = predict_time
                json.dump(export_data, export_file, indent=3)

##### Section: Model Definition
def build_model():
    # 49 = MAZE_SIZE because they have all the dimension of 7x7
    # Define input layer
    maze_input = tf.keras.Input(shape=(7, 7, 1))
    # Add hidden layers
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same')(maze_input)
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same')(x)
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same')(x)
    x = layers.PReLU()(x)
    x = layers.MaxPool2D(pool_size=(2, 2), padding='same')(x)
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')(x)
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')(x)
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')(x)
    x = layers.PReLU()(x)
    x = layers.MaxPool2D(pool_size=(2, 2), padding='same')(x)
    x = layers.Flatten()(x)
    # Define output layer
    output = layers.Dense(4)(x)
    model = tf.keras.Model(
        inputs=[maze_input],
        outputs=[output]
    )
    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt, loss='mse')
    return model

##### Section: Main
# Create input and output folder
makeDir(MAZE_INPUT_FOLDER_PATH)
makeDir(OUTPUT_FOLDER_PATH)

MAZE_FILE_PATHS = getMazeFilePaths(MAZE_INPUT_FOLDER_PATH)

model = build_model()

main = Main(model, weights_file=MODEL_INPUT_FOLDER_PATH + "1612996735_final_model.h5")
print("Starting prediction for " + str(len(MAZE_FILE_PATHS)) + " mazes")

for maze_file_path in MAZE_FILE_PATHS:
    START_CELL, TARGET_CELL, MAZE, MAZE_SEED = importMaze(maze_file_path)

    nrows, ncols = MAZE.shape # Get count of dimension

    INPUT_FILE_NAME = str(ncols) + "x1x" + str(nrows) + "_" + str(MAZE_SEED)
    print("maze seed: " + str(MAZE_SEED))
    print("Start Cell:")
    print(START_CELL)
    print("Target Cell:")
    print(TARGET_CELL)
    print(MAZE)

    qmaze = Qmaze(MAZE, START_CELL, TARGET_CELL) # Construct environment/game from numpy array: maze

    predict_time = main.predict(qmaze)
    print("Prediction finished after: " + str(predict_time))