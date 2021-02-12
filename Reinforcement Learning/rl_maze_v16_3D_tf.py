from __future__ import print_function

from math import *
import abc
import copy
import os, sys, time, datetime, json, random
import numpy as np
import pandas as pd
import tensorflow as tf
import jsbeautifier
opts = jsbeautifier.default_options()
opts.indent_size = 2

from collections import deque
from os import walk

tf.compat.v1.enable_v2_behavior()
tf.get_logger().setLevel('ERROR')

##################################################
#TensorFlow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD , Adam, RMSprop
from tensorflow.keras.layers import PReLU

##################################################
#TF-Agents
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.environments import tf_py_environment
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import wrappers
from tf_agents.environments import utils
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.agents.dqn import dqn_agent
from tf_agents.policies import random_tf_policy
from tf_agents.policies import policy_saver
from tf_agents.networks import q_network
from tf_agents.metrics import tf_metrics
from tf_agents.metrics import py_metrics
from tf_agents.drivers import dynamic_step_driver
from tf_agents.drivers import py_driver
from tf_agents.specs import array_spec
from tf_agents.utils import common
from tf_agents.eval  import metric_utils

#### Zeitstempel für die Datei
timestamp = str(time.time()).split('.')[0]


agent_dqn = True                      # False => ddqn_agent       // True => dqn_agent
optimizer_RMSprop = True               # False => Adam optimizer   // True => RMSprop opitimizer
###################################################################
#### Hyperparameter
num_iterations = 250000                 # Anzahl an Steps
                                           
initial_collect_steps = 1000            # Anazhl an Steps, die vor dem Training in die Policy eingespeist werden 
replay_buffer_capacity = 100000         # Größe des Replaybuffers

#fc_layer_params = (100,)               # aus Tutorial - Anzahl und Größe der Schichten des Neuronalen Netzes
fc_layer_params = [32,64,128]           # statische Größe
                                        # aktuell werden, die Parameter dynamisch berechnet abhängig von der Maze
target_update_period = 25   
batch_size = 128                        # @param
learning_rate = 5e-3                    # vorheriger Wert (1e-5) war zu niedrig

training_with_duration = False          # training mit maximaler Anzahl an Steps pro Epoche 
episode_duration = 200                  # Maximale anzahl der Episoden

num_eval_episodes = 1                   # Anzahl der Episoden um das Average auszurechnen
eval_interval = 1000                    # AusgabenTrigger
log_interval = 500                      # AusgabenTrigger
#####################################################################



def importMaze(maze_file):
    with open(maze_file) as json_file:
        data = json.load(json_file)
        # print(data)

        # Get agent and target pos and convert them to correct dimension
        agent_pos = data['agentStart']
        target_pos = data['agentGoal']
        start_cell = (agent_pos[0], agent_pos[2], agent_pos[1])
        target_cell = (target_pos[0], target_pos[2], target_pos[1])
        maze_seed = data['levelseed']
        # Get dimension of maze
        dimension = data['xdimensions']
        colDim = dimension[0]
        rowDim = dimension[2]
        depthDim = dimension[1]
        dim = [colDim, rowDim, depthDim]
        
        # Get level
        levelRaw = np.array(data['exportlevel'])
        
        # Change coding of start and target cell to empty cells
        for it in range(len(levelRaw)):
            if (levelRaw[it] == 14):
                levelRaw[it] = 12        
            levelRaw[it] -= 11
            if (levelRaw[it] > 0):
                levelRaw[it] += 1
        # convert Elements to TF-Agents
        # 0 = free, 1 = player, 2 = wall, 3 = Teleporter 
        for it in range(len(levelRaw)):
            if (levelRaw[it] == 2):
                levelRaw[it] = 0
            elif (levelRaw[it] == 0):
                levelRaw[it] = 2
        
        level3D = np.full((colDim,rowDim,depthDim), 0)
        it = 0
        for z in range(depthDim):
            for y in range(rowDim):
                for x in range(colDim):
                    level3D[x,y,z] = levelRaw[it]
                    it += 1
    return start_cell, target_cell, level3D, dim, maze_seed

# Create given directory if not exists
def makeDir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def getMazeFilePath(input_folder_path):
    _, _, files = next(walk(input_folder_path))
    if len(files) <= 0:
        raise Exception("no maze file to read found")
    return input_folder_path + files[0]

# Measure distance using manhattan distance
# Difference in depth gets measured by distance to next best elevator
def manhattan_distance(p1,p2):
    print("p1")
    print(p1)
    print("p2")
    print(p2)
    return abs(p1[0] - p2[0]) * ((nrows * ncols) / 2) + abs(p1[1] - p2[1]) + abs(p1[2] - p2[2])

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
    mazeMin = min(envstate[0])
    mazeMax = max(envstate[0])
    for i in range(len(envstate[0])):
        envstate[0][i] = (envstate[0][i] - mazeMin) / (mazeMax - mazeMin)
    return envstate

def printLevelLegendOn(level):
    print("#################################################################")
    print("level im Grundzustand")
    print("P = Agent- / Startposition")
    print("Z = Zielpositions")
    print("O = freie Zelle")
    print("W = Wand")
    print("T = Treppe\n")
    printLevel(level)
    print("#################################################################\n\n")

def printLevel(level):
    string_obs = np.array((level), dtype=np.unicode_)
    string_obs = np.where(string_obs=="2","W", string_obs)      #Wand
    string_obs = np.where(string_obs=="3","T", string_obs)      #Teleporter
    string_obs = np.where(string_obs=="0"," ", string_obs)      #Freie Zelle
    string_obs = np.where(string_obs=="5","Z", string_obs)      #Ziel
    string_obs = np.where(string_obs=="4","P", string_obs)      #Agent

    
    for i in range(string_obs.shape[2]):
        lane = string_obs[:, :, i]
        observe_2d = pd.DataFrame(lane)
        observe_2d.columns = [''] * len(observe_2d.columns)
        observe_2d = observe_2d.to_string(index=False)
        print("Etage:",i,"{}".format(observe_2d))


#### Level laden und Variablen speichern
input_folder_path = '../../inputs/mazeToRead/'
makeDir(input_folder_path)
maze_file_path = getMazeFilePath(input_folder_path)
print(maze_file_path)
startCell, targetCell, maze, dim, maze_seed = importMaze(maze_file_path)


#### Hyperparameter - 
layer = dim[0] + dim[1] + dim[2]
#fc_layer_params = [layer, layer * 2, layer * 4]
#fc_layer_params = layer = (100,)


#### Environment Parameter
BACKWARD = 0
FORWARD = 1
LEFT = 2
RIGHT = 3
DOWN = 4
UP = 5

tf_barrier_mark = 2
tf_elevator_mark = 3

unvisited = 0
visited = 1

#### Hyperparamter - Rewards
move_valid_value = -0.04
move_illegal = -6
move_to_already_visited_cell_value = -0.25
move_to_target = maze.size
min_reward_trigger = -600            #-(30 * (dim[0] + dim[1] + dim[2]))

proximity_reward_allowed = False                 # reward = -(distance agent ziel / max diagonale) 
proximity_reward_ration = 1                     #0.0 kein reward ; 1.0 voller reward
avg_distance = np.abs(targetCell[0] - startCell[0]) + np.abs(targetCell[1] - startCell[1]) + np.abs(targetCell[2] - startCell[2])
max_distance = maze.size
max_steps = 70

class GridWorldEnv(py_environment.PyEnvironment):
    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=5, name='action')

        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(6,), dtype=np.int32, minimum=[0,0,0,0,0,0],maximum=[dim[0],dim[1],dim[2],dim[0],dim[1],dim[2]],name='observation')
        self._step_iterator = 0
        self._state=[startCell[0],startCell[1],startCell[2],targetCell[0],targetCell[1],targetCell[2]] 
        self._episode_ended = False
        self.visitedMaze = np.full((dim[0],dim[1],dim[2]),unvisited)
        self.visitedMaze[startCell[0],startCell[1],startCell[2]] = visited
        self.accumulated_reward = 0
        self.min_reward = min_reward_trigger

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._step_iterator = 0
        self._state=[startCell[0],startCell[1],startCell[2],targetCell[0],targetCell[1],targetCell[2]]
        self.visitedMaze = np.full((dim[0],dim[1],dim[2]),unvisited)
        self.visitedMaze[startCell[0],startCell[1],startCell[2]] = visited
        self._episode_ended = False
        self.accumulated_reward = 0
        return ts.restart(np.array(self._state, dtype=np.int32))

    def _step(self, action):
        if self._episode_ended or self._step_iterator > max_steps or self.accumulated_reward < self.min_reward:
            return self.reset()

        self._step_iterator += 1
        
        move_reward = self.move(action)
        move_reward += self.visited_reward()
        move_reward += self.distance_reward()
        if proximity_reward_allowed == True:
            move_reward += self.proximity_reward()
        
        self.accumulated_reward += move_reward

        if self.game_over():
            self._episode_ended = True
            move_reward += move_to_target

        if self._episode_ended or self._step_iterator > max_steps or self.accumulated_reward < self.min_reward:
            return ts.termination(np.array(self._state, dtype=np.int32), move_reward)
        else:
            return ts.transition(
                np.array(self._state, dtype=np.int32), reward=move_reward, discount=0.9)


    def move(self, action):
        row, col, aisle = self._state[0],self._state[1],self._state[2]
        move_reward = move_valid_value

        move_is_valid = False 
        if action == BACKWARD:  # [-1,0,0]
            if row - 1 >= 0:
                if maze[row - 1,col,aisle] != tf_barrier_mark:  #not blocked
                    self._state[0] -= 1
                    move_is_valid = True
        if action == FORWARD:   # [+1,0,0]
            if row + 1 < dim[0]:
                if maze[row + 1,col,aisle] != tf_barrier_mark:  #not blocked
                    self._state[0] += 1
                    move_is_valid = True
        if action == LEFT:      # [0,-1,0]
            if col - 1 >= 0:
                if maze[row,col - 1,aisle] != tf_barrier_mark:  #not blocked
                    self._state[1] -= 1
                    move_is_valid = True
        if action == RIGHT:     # [0,+1,0]
            if col + 1  < dim[1]:
                if maze[row,col + 1,aisle] != tf_barrier_mark:  #not blocked
                    self._state[1] += 1
                    move_is_valid = True
        if action == DOWN:      # [0,0,-1]
            if aisle - 1  >= 0:
                if maze[row,col,aisle - 1] == tf_elevator_mark and maze[row,col,aisle] == tf_elevator_mark:
                    self._state[2] -= 1
                    move_is_valid = True
        if action == UP:        # [0,0,+1]
            if aisle + 1  < dim[2]:
                if maze[row,col,aisle + 1] == tf_elevator_mark and maze[row,col,aisle] == tf_elevator_mark:
                    self._state[2] += 1
                    move_is_valid = True
        if(move_is_valid == False):
            move_reward += move_illegal
            self._episode_ended = True
        return move_reward

        
    def proximity_reward(self):
        actual_distance = np.abs(self._state[3] - self._state[0]) + np.abs(self._state[4] - self._state[1]) + np.abs(self._state[5] - self._state[2])
        #ratio = actual_distance / avg_distance
        ratio = actual_distance / max_distance
        reward = 0

        reward = -np.abs(ratio)
        return reward
        
    def distance_reward(self):
        actual_distance = np.abs(self._state[3] - self._state[0]) + np.abs(self._state[4] - self._state[1])
        reward = -(actual_distance * 0.01)
        return reward
        
    def visited_reward(self):
        row, col, aisle = self._state[0],self._state[1],self._state[2]
        reward = 0
        if(self.visitedMaze[row,col,aisle] == unvisited):
            reward = 0
        else:
            reward = move_to_already_visited_cell_value
        self.visitedMaze[row,col,aisle] = visited
        return reward
        
    def game_over(self):
        row, col, aisle = self._state[0],self._state[1],self._state[2] 
        frow, fcol, faisle = self._state[3],self._state[4],self._state[5]
        return row==frow and col==fcol and aisle==faisle    
        
        
#Validierung für die Environment. wird mit Zufallsaktionen berechnet
#if __name__ == '__main__':
if(True):
    env = GridWorldEnv()
    utils.validate_py_environment(env, episodes=5)

    tl_env = wrappers.TimeLimit(env, duration=50)

    #randomRoute
    print("validation for the enviroment")
    print("random Route")
    time_step = tl_env.reset()
    #print("action:", action, " observation:", time_step.observation, " reward:", time_step.reward)
    rewards = time_step.reward

    for i in range(10):
        action = np.random.choice([0,1,2,3,4,5])
        time_step = tl_env.step(action)
        print("action:", action, " observation:", time_step.observation, " reward:", time_step.reward)
        rewards += time_step.reward

    print("reward: ", rewards)

#zusätzlich: Kontrolle der movemethode und rewards
    print("\nDimensionalitätstest mit für Tiefe Breite Höhe und entsprechende Actions")
    time_step = tl_env.reset()
    rewards = 0
    #pr2 = (RIGHT, LEFT, FORWARD, BACKWARD, RIGHT, RIGHT, RIGHT, RIGHT, UP, DOWN, DOWN)
    pr = (LEFT, LEFT, LEFT, DOWN, DOWN, FORWARD, LEFT, LEFT,LEFT, FORWARD,FORWARD, UP,UP,FORWARD,FORWARD,LEFT,LEFT)
    for i in range(len(pr)):
       action = pr[i]
       time_step = tl_env.step(action)
       print("action:", action, " observation:", time_step.observation, " reward:", time_step.reward)
       rewards += time_step.reward
    print("reward: ", rewards)



#### Trainings- und Evaluations-environment
if(training_with_duration == True):
#### Environment mit fester Episodendauer. Sinnvoll bei Problemen, die keine oder schlechte Abbruch-Kriterien haben
    train_py_env = wrappers.TimeLimit(GridWorldEnv(), duration=episode_duration)
    eval_py_env = wrappers.TimeLimit(GridWorldEnv(), duration=episode_duration)
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
else:
#### Training ohne Episodendauer
    train_env = tf_py_environment.TFPyEnvironment(GridWorldEnv())
    eval_env = tf_py_environment.TFPyEnvironment(GridWorldEnv())


#### Netzwerk
q_net = q_network.QNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=fc_layer_params)

train_step_counter = tf.compat.v2.Variable(0)

#### Erstellen des TF-Agenten

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
if(optimizer_RMSprop == True):
    optimizer = tf.keras.optimizers.RMSprop(lr=learning_rate, rho=0.95, momentum=0.0, epsilon=0.00001, centered=True)

update_period = 4
epsilon_fn = tf.keras.optimizers.schedules.PolynomialDecay(
                 initial_learning_rate=1.0, 
                 decay_steps=num_iterations // update_period,# die Formel lässt das Epsilon sinken
                 end_learning_rate=0.01)
                 
#### ddqn - Agent mit Adam Optimierer. Erzielt in Tests bessere Ergebnisse                 
tf_agent = dqn_agent.DdqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        target_update_period=target_update_period,
        td_errors_loss_fn=tf.keras.losses.Huber(reduction="none"),
        gamma=0.99,
        train_step_counter=train_step_counter,
        epsilon_greedy=lambda: epsilon_fn(train_step_counter)
        )
        
if(agent_dqn == True):
#### dqn - Agent
    tf_agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn = dqn_agent.common.element_wise_squared_loss,
        train_step_counter=train_step_counter
        )
        
tf_agent.initialize()

#### Hilfsfunktion für TF-Agent
def collect_step(environment, policy):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    replay_buffer.add_batch(traj)

def compute_avg_return(environment, policy, num_episodes=10):

    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
            total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

#### Policy, Replaybuffer, Observer und Metriken
eval_policy = tf_agent.policy
collect_policy = tf_agent.collect_policy

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_capacity)



train_metrics = [
            tf_metrics.NumberOfEpisodes(),
            tf_metrics.EnvironmentSteps(),
            tf_metrics.AverageReturnMetric(),
            tf_metrics.AverageEpisodeLengthMetric(),
]
         
#### dataSet und driver
dataset = replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=batch_size,
    num_steps=2).prefetch(3)

replay_observer = [replay_buffer.add_batch]

initial_collect_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())
initial_driver = dynamic_step_driver.DynamicStepDriver(
    train_env,
    initial_collect_policy,
    observers=replay_observer + train_metrics,
    num_steps=1)
    
driver = dynamic_step_driver.DynamicStepDriver(
           train_env,
           collect_policy,
           observers=replay_observer + train_metrics,
   num_steps=1)


#### Training - grundstätzlich Setup
iterator = iter(dataset)
print("average untrainiert: ", compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes))
driver.run = common.function(driver.run)
tf_agent.train = common.function(tf_agent.train)
tf_agent.train_step_counter.assign(0)

final_time_step, policy_state = driver.run()

policy_state = tf_agent.collect_policy.get_initial_state(train_env.batch_size)
for i in range(initial_collect_steps):
    final_time_step, _ = initial_driver.run(final_time_step, policy_state)

episode_len = []
step_len = []
start_time = datetime.datetime.now()    

for i in range(num_iterations):
    final_time_step, _ = driver.run(final_time_step, policy_state)
    experience, _ = next(iterator)
    train_loss = tf_agent.train(experience=experience)
    step = tf_agent.train_step_counter.numpy()

    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss.loss))
        episode_len.append(train_metrics[3].result().numpy())
        step_len.append(step)
        print('Average episode length: {}'.format(np.floor(train_metrics[3].result().numpy())))

    if step % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
        print('Episode = {0}: Average Return = {1}'.format(step, np.floor(avg_return)))
#plt.plot(step_len, episode_len)
#plt.xlabel('Episodes')
#plt.ylabel('Average Episode Length (Steps)')
#plt.show()

#### Dauer
dt = datetime.datetime.now() - start_time
t = format_time(dt.total_seconds())

#### Berechnung des Ergebnispfad 
def compute_evaluated_path(environment, policy, num_episodes=10, game_status='lose'):
    actions = []
    for episode in range(num_episodes):
        time_step = environment.reset()
        while not time_step.is_last():
            pos = time_step.observation.numpy()
            actions.append(pos[-1,0])
            actions.append(pos[-1,2])
            actions.append(pos[-1,1])
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)  
        pos = time_step.observation.numpy()
        cell = [pos[-1,0],pos[-1,1],pos[-1,2]]
        if(pos[-1,0] == pos[-1,3] and pos[-1,1] == pos[-1,4] and pos[-1,2] == pos[-1,5]):
            print("Target erfolgreich gefunden")
            game_status = 'win'
        actions.append(pos[-1,0])
        actions.append(pos[-1,2])
        actions.append(pos[-1,1])             
    return actions, game_status

#### Anzeige der Evaluierung
def compute_viz(environment, policy, num_episodes=10):  
    #einmalig level im Grundzustand ausgeben
    viz_maze = copy.deepcopy(maze)
    viz_maze[startCell[0], startCell[1],startCell[2]] = 4         
    viz_maze[targetCell[0],targetCell[1],targetCell[2]] = 5
    printLevelLegendOn(viz_maze)

    total_return = 0.0
    for episode in range(num_episodes):
        viz_maze = copy.deepcopy(maze)
        time_step = environment.reset()
        episode_return = 0.0
        step = 0
        while not time_step.is_last():
            pos = time_step.observation.numpy()
            if(step < 1):                                       #Ziel einmalig speichern
                viz_maze[pos[-1,3],pos[-1,4],pos[-1,5]] = 5
            viz_maze[pos[-1,0],pos[-1,1],pos[-1,2]] = 4         #Agentposition speichern

            episode_return += time_step.reward
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            step += 1

        pos = time_step.observation.numpy()                     #Letzte position aufnehmen
        viz_maze[pos[-1,0],pos[-1,1],pos[-1,2]] = 4             

        episode_return += time_step.reward
        print('episode: ', episode, '  -- steps: ',step)
        printLevel(viz_maze)
        print("Reward: {} \n".format(episode_return))
        total_return += episode_return

    avg_return = total_return / num_episodes
    
    return avg_return.numpy()[0] 

finale_avg_return = compute_viz(eval_env, tf_agent.policy, 1)
final_game_status = 'lose'
final_actions, final_game_status = compute_evaluated_path(eval_env, tf_agent.policy, 1, final_game_status)
print("final average: ", finale_avg_return)

def save_files(final_actions, final_game_status, timestamp, epoch_reward_sum):
        outdir_path = os.path.realpath(__file__) + "/../../../output/v15/"
        makeDir(outdir_path)
        input_file_name = str(dim[1]) + "x" + str(dim[2]) + "x" + str(dim[0]) + "_" + str(maze_seed)

        h5file = outdir_path + input_file_name + "_model" + ".h5"
        json_file = outdir_path + input_file_name + "_model" + ".json"
        path_file = outdir_path + timestamp + "_" + input_file_name + "_path_info" + ".txt"
        coord_file = outdir_path + timestamp + "_" + input_file_name + "_path" + ".json"

        # Save weights of model into file
        #model.save_weights(h5file, overwrite=True)

        #with open(json_file, "w") as outfile:
            #json.dump(model.to_json(), outfile)

        # Output Data of path into file
        with open(path_file, "w") as outfile:
            outfile.write("Maze Size: " + str(dim[1]) + "x" + str(dim[2]) + "x" + str(dim[0]))
            outfile.write("\nPath Length: " + str(len(final_actions)))
            outfile.write("\nReward Sum: " + str(epoch_reward_sum))
            outfile.write("\nElapsed Time to find this path: " + str(t))
            outfile.write("\nGame Status: " + str(final_game_status))
            outfile.write("\nStart:" + str(startCell))
            outfile.write("\nTarget:" + str(targetCell))
            outfile.write("\nFinal best path:\n")

            for action in final_actions:
                #action = format_path(action)
                outfile.write("%s\n" % str(action))

        # Output coords of path in seperate file
        print(final_game_status)
        if (final_game_status == 'win'):
            coord_data = []
            for coord in final_actions:
                coord_data.append(int(coord))
            print(maze_file_path)
            if maze_file_path is not None:
                with open(maze_file_path, "r") as import_file, open(coord_file, "w") as export_file:
                    import_data = json.load(import_file)
                    export_data = dict(import_data)
                    export_data['resultPath'] = coord_data
                    json.dump(export_data, export_file, indent=3)
            # If no file imported, just dump the resultPath; for testing only
            else:
                with open(coord_file, "w") as export_file:
                    export_data = dict()
                    export_data['resultPath'] = coord_data
                    json.dump(export_data, export_file, indent=3)

save_files(final_actions, final_game_status, timestamp, finale_avg_return)
#template = "{}: Epoch: {:03d}/{:d} | Loss: {:.5f} | Reward Sum: {:.3f} | Path Length: {:d} | Minimal Path Length {:d} | Win Count: {:d} | time: {}"