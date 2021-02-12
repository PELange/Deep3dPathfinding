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

MAZE_INPUT_FOLDER_PATH = os.path.realpath(__file__) + "/../../output/TL/mazeFOVs/"
OUTPUT_FOLDER_PATH = os.path.realpath(__file__) + "/../../output/TL/dataGen/"
TRAINING_STATS_FILE_PATH = OUTPUT_FOLDER_PATH + "training_statistics" + ".txt"

# Tweaking factors (Hyper parameters)
N_EPOCHS_UPDATE_TARGET_MODEL = 10 # In what episode interval do the weights of the targetmodel get updated from the onlinemodel
MAZE_SIZE_N_EPISODE_PERCENTILE = 0.5 # At how many steps should the NN stop the episode, depending on maze.size (1.0 = 100% of maze.size)
N_MIN_PATH_NOT_DECREASED_ABORT = 100 # When the NN doesn't find any better path for N episodes, the training aborts

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
            if '.json' in file:
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

# Normalize prev_envstate and envstate for the neural network, so the calculations are faster
def normalize_envstate(envstate):
    for i in range(len(envstate[0])):
        envstate[0][i] = envstate[0][i] / 6
    return envstate

# Checks if a cell not a barrier
def check_maze_cell_valid(maze, row, col):
    if maze[row][col] == barrier_mark:
        return False
    else:
        return True

# Extend 5x5 Maze with outer lane, for all possible directions, when target is not in current FOV
# Returns a 7x7 Maze
def extend_maze_to_7x7(maze):
    maze = np.vstack([[12.0, 12.0, 12.0, 12.0, 12.0], maze])
    maze = np.vstack([maze, [12.0, 12.0, 12.0, 12.0, 12.0]])
    maze = np.column_stack([[12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0], maze])
    maze = np.column_stack([maze, [12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0]])
    return maze

# Save generated Q-Value-FOV-Tuple in file
def saveData(maze, target_cell, q_values, timestamp):
    maze_copy = copy.deepcopy(maze)
    maze_copy[target_cell[0]][target_cell[1]] = target_mark
    data_file_path = OUTPUT_FOLDER_PATH + str(timestamp) + "_TL_data.h5"
    maze_q_value_tuple = np.concatenate((maze_copy.flatten(), q_values.flatten()))
    print("Saving tuple:")
    print(maze_q_value_tuple)
    with h5py.File(data_file_path, "a") as dataset_file:
        new_key = str(len(dataset_file.keys()))
        dataset_file.create_dataset(new_key, shape=maze_q_value_tuple.shape, data=maze_q_value_tuple)

def loadData(file_name):
    data_file_path = OUTPUT_FOLDER_PATH + file_name
    fovs = deque()
    qValues_list = deque()

    if os.path.isfile(data_file_path):
        with h5py.File(data_file_path, "r") as dataset_file:
            for key in dataset_file.keys():
                dataset = dataset_file[key]
                fov = np.array(dataset[0:49].reshape((1, -1)))
                qValues = np.array(dataset[49:53].reshape((1, -1)))
                fovs.append(fov)
                qValues_list.append(qValues)
    
    return np.asarray(fovs), np.asarray(qValues_list)

##### Section: Environment
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
        self.total_reward = 0
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

    # Get reward based on agent mode and agent position
    def get_reward(self):
        agent_row, agent_col, mode = self.agent
        tar_row, tar_col = self.target

        # Target cell
        if agent_row == tar_row and agent_col == tar_col:
            return 1.0
        # Agent is stuck somehow (This should never happen)
        if mode == 'blocked':
            return self.min_reward - 1        
        # Walking trough walls or going out of bounds
        if mode == "invalid":
            return -0.75
        # Visited cell
        if (agent_row, agent_col) in self.visited:
            return -0.25
        # Unvisited cell
        if mode == 'valid':
            return -0.04

    # Update envstate and game status according to given action and get reward
    def act(self, action, n_steps):
        self.update_state(action)
        reward = self.get_reward()
        self.total_reward += reward
        status = self.game_status(n_steps)
        # envstate = self.observe()
        agent_pos = self.getAgentPos()
        return agent_pos, reward, status

    # Returns the state of the maze as 1D-Array -> the complete maze, with the current location of the agent
    def observe(self):
        canvas = self.draw_env()
        envstate = canvas.reshape((1, -1)) # Reshapes the canvas into a one dimensional array (two dimensional with one entry)
        return envstate

    # "Draw" environment as 2D-Array with agent in it
    def draw_env(self):
        canvas = np.copy(self._maze)
        # draw the agent
        row, col, _ = self.agent
        canvas[row, col] = agent_mark
        return canvas

    # Get current game status
    def game_status(self, n_steps):
        # Return lose if reward exceeds minimum threshold
        if n_steps >= self._maze.size * MAZE_SIZE_N_EPISODE_PERCENTILE:
            return 'lose'
        agent_row, agent_col, _ = self.agent
        target_row, target_col = self.target
        # Return win if agent reached target
        if agent_row == target_row and agent_col == target_col:
            return 'win'

        return 'not_over'

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
    
    def getAgentPos(self):
        agent_row, agent_col, _ = self.agent
        agent_pos = np.array([agent_row, agent_col], dtype="float32").reshape((1, -1))
        return agent_pos


##### Section: Experiences
# This class saves experiences in the replay buffer (memory) and predicts the next targets using that memory
class ReplayBuffer(object):
    def __init__(self, max_memory, discount=0.95):
        self.max_memory = max_memory
        self.discount = discount
        self.memory = deque()

    # Saves current step taken into memory
    def remember(self, experience):
        # experience = [envstate, action, reward, envstate_next, game_over]
        # envstate == flattened 1d maze cells info, including agent cell
        self.memory.append(experience)
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    # Predict the next targets using a random batch from memory
    def predict_targets(self, model, target_model, batch_size=10):
        env_size = self.memory[0][0].shape[1] # envstate 1d size (1st element of experience)
        mem_size = len(self.memory)
        batch_size = min(mem_size, batch_size)
        agent_poss = np.zeros((batch_size, env_size))
        agent_poss_next = np.zeros((batch_size, env_size))

        targets = np.zeros((batch_size, num_actions))
        Q_sa = np.zeros((batch_size, num_actions))
        Q_saMax = np.zeros((batch_size))
        game_overs = np.zeros((batch_size))
        rewards = np.zeros((batch_size))
        actions = np.zeros((batch_size), dtype=np.int32)
        # Iterate over random experiences stored in memory
        for i, j in enumerate(np.random.choice(range(mem_size), batch_size, replace=False)):
            agent_pos, action, reward, agent_pos_next, game_over = self.memory[j]
            agent_poss[i] = agent_pos
            agent_poss_next[i] = agent_pos_next
            game_overs[i] = game_over
            rewards[i] = reward
            actions[i] = action

        # There should be no target values for actions not taken.
        targets = model.predict_on_batch(agent_poss) # Q-Values for every action taken in this state (6 Results)

        # Q_sa = derived policy = max quality env/action = max_a' Q(s', a')
        # Use target model here to prevent network to be unstable (freeze, diverge, oszillate (schwanken))
        Q_sa = target_model.predict_on_batch(agent_poss_next) # Highest Q-Value for the best action of the next step in the future

        for i in range(len(Q_sa)):
            Q_saMax[i] = np.max(Q_sa[i])

            if game_overs[i]:
                targets[i, actions[i]] = rewards[i] # Save current reward without future reward when action returns in game over
            else:
                # reward + gamma * max_a' Q(s', a')
                targets[i, actions[i]] = rewards[i] + self.discount * Q_saMax[i]

        return agent_poss, targets

##### Section: Main Class
class Main(object):
    def __init__(self, **opt):
        self.n_episode = opt.get('episodes', 15000)
        self.max_memory = opt.get('max_memory', 1000)
        self.batch_size = opt.get('batch_size', 32)
        self.epsilon_decrease_rate = 1 / (self.n_episode / 10)
    
    def qtrain(self, model, qmaze, target):
        win_count = 0
        epsilon = 1 # Exploration factor, 10% of all episodes are exploration episodes, the rest is exploitation
        start_time = datetime.datetime.now()
        path_found = False

        self.final_actions = list() # List to save the final actions that represent the best found path
        self.final_coords = list()

        n_min_path_not_decreased = 0

        # Create target model from online model
        target_model = keras.models.clone_model(model)

        # Initialize replay_buffer replay object
        replay_buffer = ReplayBuffer(max_memory=self.max_memory)

        current_actions = list() # list to save the current path
        current_coords = list() # list to save only coordinates of the tiles the agent took
        min_path_size = math.inf # current minimal path size
        n_till_training = 0

        # Run the path finding algorithm for n_episodes
        for episode in range(self.n_episode):
            avg_loss = 0.0
            qmaze.reset(START_CELL)
            game_over = False

            # Update target_model every X episodes
            if episode % N_EPOCHS_UPDATE_TARGET_MODEL == 0:
                target_model.set_weights(model.get_weights())

            # get initial agent_pos
            agent_pos = qmaze.getAgentPos()
            agent_pos = normalize_envstate(agent_pos) # Normalize agent pos input for NN for better performance

            current_actions.clear()
            current_coords.clear()
            n_steps = 0
            episode_reward_sum = 0
            # Start the game loop
            while not game_over:
                n_steps += 1
                # Get valid actions for current envstate
                valid_actions = qmaze.valid_actions()
                if not valid_actions: break
                prev_agent_pos = agent_pos
                # Get next action                
                action = self.explore_or_exploit(model, prev_agent_pos, epsilon, valid_actions)

                # Apply action, get reward and new envstate
                agent_pos, reward, game_status = qmaze.act(action, n_steps)
                agent_pos = normalize_envstate(agent_pos) # Normalize agent pos input for NN for better performance
                # Set break condition if game is over
                if game_status == 'win':
                    win_count += 1
                    game_over = True
                    path_found = True
                elif game_status == 'lose':
                    game_over = True
                else:
                    game_over = False       

                # Store experience
                experience = [prev_agent_pos, action, reward, agent_pos, game_over]
                # Only add action to current actions list if it is a valid action
                agent_row, agent_col, _ = qmaze.agent
                agent_coords = "[" + str(agent_col) + "," + str(agent_row) + "]"
                if (action in valid_actions):
                    current_actions.append(str(action) + "_VALID; " + agent_coords)
                    current_coords.append(agent_col)
                    current_coords.append(agent_row)
                else:
                    current_actions.append(str(action) + "_INVALID; " + agent_coords)
                replay_buffer.remember(experience)
                episode_reward_sum += reward

                # Only start training when replay buffer is filled to 1/50 of it's size
                if len(replay_buffer.memory) >= self.max_memory / 50 and n_till_training >= 3:
                    n_till_training = 0
                    # Predict targets using the target_model
                    inputs, targets = replay_buffer.predict_targets(model, target_model, batch_size=self.batch_size)
                    # Train neural network model
                    model.fit(
                        inputs,
                        targets,
                        epochs=8,
                        batch_size=self.batch_size,
                        verbose=0, 
                    )    
                else:
                    n_till_training += 1

            # Decrease epsilon if still in exploration mode
            if (epsilon < 0.05 or epsilon < self.epsilon_decrease_rate): epsilon = 0.05
            if (epsilon > 0.05): epsilon -= self.epsilon_decrease_rate

            dt = datetime.datetime.now() - start_time
            t = format_time(dt.total_seconds())
            
            # Count how many steps the minimal path length did not change
            if min_path_size <= len(current_actions): n_min_path_not_decreased += 1

            # Save min_path_size, shortest path and it's coords
            if (min_path_size > len(current_actions)): 
                n_min_path_not_decreased = 0
                min_path_size = len(current_actions)
                self.final_actions.clear()
                self.final_coords.clear()
                # Append the start cell to the path coordinates
                self.final_coords.append(START_CELL[1])
                self.final_coords.append(START_CELL[0])
                for action in current_actions:
                    self.final_actions.append(action)
                for coord in current_coords:
                    self.final_coords.append(coord)
                    
            template = "{}: Episode: {:03d}/{:d} | Loss: {:.5f} | Reward Sum: {:.3f} | Path Length: {:d} | Minimal Path Length {:d} | Win Count: {:d} | time: {}"

            print(template.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"), episode + 1, self.n_episode, avg_loss, episode_reward_sum, len(current_actions), len(self.final_actions), win_count, t))

            if n_min_path_not_decreased >= N_MIN_PATH_NOT_DECREASED_ABORT: 
                print("No better path found for " + str(N_MIN_PATH_NOT_DECREASED_ABORT) +  " Iterations, abort training")
                self.saveTrainingStatistics(path_found, t, target)
                return self.get_center_QValues(model), path_found

        dt = datetime.datetime.now() - start_time
        t = format_time(dt.total_seconds())
        self.saveTrainingStatistics(path_found, t, target)
        return self.get_center_QValues(model), path_found

    # Calculate the next step based on the epsilon greedy policy
    # Either random action (explore) or best action (exploit) depending on epsilon
    def explore_or_exploit(self, model, state, epsilon, valid_actions):
        if np.random.rand() < epsilon:
            return random.choice(valid_actions)
        else:
            return np.argmax(model.predict(state)[0])

    # Gets the Q-Values for the Center Position of this FOV
    def get_center_QValues(self, model):
        # return model.predict(np.array([3.0, 3.0]).reshape(1, -1))[0] # Not normalized pos
        return model.predict(np.array([0.5, 0.5]).reshape(1, -1))[0] # Normalized pos for (3.0, 3.0)

    def saveTrainingStatistics(self, path_found, time, target):
        with open(TRAINING_STATS_FILE_PATH, "a") as outfile:
            if path_found:
                outfile.write("Target: " + "[" + str(target[1])  + "," + str(target[0]) + "]" + "; Path Length: " + str(len(self.final_actions)) 
                + "; Path leads to Target: " + str(path_found) + "; Elapsed Time: " + str(time) + "\n")
            else:
                outfile.write("Target: " + "[" + str(target[1])  + "," + str(target[0]) + "]" + "; Path Length: " + str(len(self.final_actions)) 
                + "; Path leads to Target: " + str(path_found) + "; --- Retry number: " + str(RETRY_COUNT) + ". Abort after 10 tries. Elapsed Time: " + str(time) + "\n")

##### Section: Model
def build_model():
    # 49 = MAZE_SIZE because they have all the dimension of 7x7
    model = tf.keras.models.Sequential([
        keras.layers.Dense(49, input_shape=(2,)),
        keras.layers.PReLU(),
        keras.layers.Dense(49),
        keras.layers.PReLU(),
        keras.layers.Dense(num_actions) # num_actions defines how many outputs this neural network should have (in this case: 4)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

##### Section: Main Function
start_time = datetime.datetime.now()
timestamp = str(time.time()).split('.')[0] # Get current timeStamp for saving files
# Create input and output folder
makeDir(MAZE_INPUT_FOLDER_PATH)
makeDir(OUTPUT_FOLDER_PATH)

if not os.path.isfile(TRAINING_STATS_FILE_PATH):
    open(TRAINING_STATS_FILE_PATH, "w")

mazes_blocked_count = 0
data_tuple_count = 0
main = Main(episodes=500, max_memory=10000, batch_size=32)
MAZE_FILE_PATHS = getMazeFilePath(MAZE_INPUT_FOLDER_PATH)
print("Starting training for " + str(len(MAZE_FILE_PATHS)) + " mazes")
with open(TRAINING_STATS_FILE_PATH, "a") as outfile:
    outfile.write("----- Starting training for " + str(len(MAZE_FILE_PATHS)) + " mazes -----\n")
    outfile.write( "-----------------------------------------------------------------------------------------------------------------------\n")

for maze_file_path in MAZE_FILE_PATHS:
    start_time_inner = datetime.datetime.now()
    target_count = 0

    MAZE, MAZE_SEED = importMaze(maze_file_path)

    MAZE = extend_maze_to_7x7(MAZE)
    print("maze seed: " + str(MAZE_SEED))
    print("Current Maze:")
    print(MAZE)

    nrows, ncols = np.shape(MAZE) # Get count of dimension
    maze_name = maze_file_path.rsplit('/', 1)[1][:-5]

    with open(TRAINING_STATS_FILE_PATH, "a") as outfile:
        outfile.write("----- Starting training for maze: " + maze_name + " -----\n")

    MAZE_SIZE = MAZE.size

    # Skip maze if center cell is a wall
    if not check_maze_cell_valid(MAZE, 3, 3):
        print("Maze: " + maze_file_path + " has a barrier in the middle, continue with next maze")
        mazes_blocked_count += 1
        with open(TRAINING_STATS_FILE_PATH, "a") as outfile:
            outfile.write("----- Maze: " + maze_name + " has a barrier in the middle, aborting training iteration and continue with next maze -----\n")
    else:
        START_CELL = (3, 3)
        # Iterate over all possible targets and train generate training samples for them
        for row in range(0, nrows):
            for col in range(0, ncols):
                if not check_maze_cell_valid(MAZE, row, col) or (row == 3 and col == 3):
                    print("Target is wall or center of maze, continue with next target. Row: " + str(row) + " Col: " + str(col))
                else:
                    print("Starting new training")                    
                    target_count += 1
                    target_cell = (row, col)
                    tf.keras.backend.clear_session() # Clear cache of tf.keras, otherwise all models are saved in this cache
                    model = build_model()

                    qmaze = Qmaze(MAZE, START_CELL, target_cell) # Construct environment/game from numpy array: maze (see above)

                    print("target cell:")
                    print(target_cell)
                    path_found = False
                    center_qvalues = np.zeros((4))
                    RETRY_COUNT = 1
                    # Retry training 10 times until a path was found
                    while not path_found and RETRY_COUNT <= 10:
                        center_qvalues, path_found = main.qtrain(model, qmaze, target_cell)
                        if path_found == False:
                            tf.keras.backend.clear_session() # Clear cache of tf.keras, otherwise all models are saved in this cache
                            model = build_model()
                            RETRY_COUNT += 1
                    saveData(MAZE, target_cell, center_qvalues, timestamp)

        dt = datetime.datetime.now() - start_time_inner
        t = format_time(dt.total_seconds())
        with open(TRAINING_STATS_FILE_PATH, "a") as outfile:
            outfile.write("----- Training finished for maze: " + maze_name + " in: " + t + "; " + str(target_count) + " out of 49 cells were valid targets -----\n")
            outfile.write( "-----------------------------------------------------------------------------------------------------------------------\n")
        data_tuple_count += target_count

dt = datetime.datetime.now() - start_time
t = format_time(dt.total_seconds())
with open(TRAINING_STATS_FILE_PATH, "a") as outfile:
    outfile.write("----- Training finished for: " + str(len(MAZE_FILE_PATHS)) + " mazes in: " + t + "-----\n")
    outfile.write("----- " + str(mazes_blocked_count) + " out of " + str(len(MAZE_FILE_PATHS)) + " mazes had a barrier in the middle and were skipped -----\n")
    outfile.write("----- Generated " + str(data_tuple_count) + " data tuples -----\n")
    outfile.write( "-----------------------------------------------------------------------------------------------------------------------\n")

# LEFT = 0
# FWD = 1
# RIGHT = 2
# BACKWARD = 3