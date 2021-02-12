from __future__ import print_function
import os, sys, time, datetime
import numpy as np
from numpy.core.fromnumeric import reshape
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')
from os import walk
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque
import h5py
import csv

##### Section: Defining GLOBALS

os.environ['TF_NUM_INTEROP_THREADS'] = '6'
os.environ['TF_NUM_INTRAOP_THREADS'] = '6'

tf.config.threading.set_intra_op_parallelism_threads(6)
tf.config.threading.set_inter_op_parallelism_threads(6)

# Possible cell marks
target_mark = 9.0
agent_mark = 10.0 # Current agent position
barrier_mark = 11.0 # Cell where the agent cannot stand
empty_mark = 12.0 # Passable cell for the agent

OUTPUT_FOLDER_PATH = os.path.realpath(__file__) + "/../../output/TL/dataGen/test/"
MODEL_FOLDER_PATH = os.path.realpath(__file__) + "/../../model/"
MAX_EPOCHS = 10000
timestamp = str(time.time()).split('.')[0] # Get current timeStamp for saving files

epoch_loss_csv_path = MODEL_FOLDER_PATH + timestamp + "_loss_history_epoch.csv"
batch_loss_csv_path = MODEL_FOLDER_PATH + timestamp + "_loss_history_batch.csv"

##### Section: Loss Callback Class
class LossCallback(keras.callbacks.Callback):
    def on_batch_end(self, batch, logs):
        with open(batch_loss_csv_path, "a+") as loss_file: 
            loss = logs['loss']
            loss_file.write(str(loss) + "\n")

    def on_epoch_end(self, epoch, logs=None):
        with open(epoch_loss_csv_path, "a+") as loss_file: 
            loss = logs['loss']
            loss_file.write(str(loss) + "\n")
        
##### Section: Utility functions

# Get all mazes out of input folder
def getDataFilePaths(data_folder_path):
    _, _, files = next(walk(data_folder_path))
    file_paths = list()
    if len(files) <= 0:
        raise Exception("no data file to read found")
    else:
        for file in files:
            if "data.h5" in file:
                file_paths.append(data_folder_path + file)
    return np.array(file_paths)

# Create given directory if not exists
def makeDir(path):
    if not os.path.exists(path):
        os.makedirs(path)

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
    mazeMin = target_mark
    mazeMax = empty_mark
    for i in range(len(envstate[0])):
        envstate[0][i] = (envstate[0][i] - mazeMin) / (mazeMax - mazeMin)
    return envstate

def loadData(data_file_path, fovs, qValues_list):
    if os.path.isfile(data_file_path):
        with h5py.File(data_file_path, "r") as dataset_file:
            for key in dataset_file.keys():
                dataset = dataset_file[key]
                fov = np.array(dataset[0:49].reshape((1, -1)))
                qValues = np.array(dataset[49:53].reshape((4)))
                fov = normalize_envstate(fov).reshape((7, 7, 1))
                fovs.append(fov)
                qValues_list.append(qValues * 100) # Multiply qValues by 100 for better Loss-Rate reduction
    
    return fovs, qValues_list

def buildDataPipeline(fovs, qValues_list, max_batch):
    fovs_arr = np.asarray(fovs)
    qValues_arr = np.asarray(qValues_list)

    dataset = tf.data.Dataset.from_tensor_slices((fovs_arr, qValues_arr))
    dataset = dataset.shuffle(buffer_size=len(fovs))
    dataset = dataset.repeat(5)
    dataset = dataset.batch(batch_size=max_batch, drop_remainder=True)

    return dataset

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
    # x = layers.Dropout(0.2)(x)
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

# Decide if using cpu or gpu
with tf.device('/gpu:0'):
# with tf.device('/cpu:0'):

    ##### Section: Main Function
    start_time = datetime.datetime.now()
    sample_count = 0
    # Create input and output folder
    makeDir(MODEL_FOLDER_PATH)
    makeDir(OUTPUT_FOLDER_PATH)

    model = build_model()
        
    # Get all data files with suffix "data.h5"
    DATA_FILE_PATHS = getDataFilePaths(OUTPUT_FOLDER_PATH)
    # Create dequeues for loading data
    fovs = deque()
    qValues_list = deque()
    # Iterate over all data files and train the model
    for data_file_path in DATA_FILE_PATHS:
        fovs, qValues_list = loadData(data_file_path, fovs, qValues_list)
    print(model.summary())
    dataset = buildDataPipeline(fovs, qValues_list, 64)
    model_path = MODEL_FOLDER_PATH + timestamp + "_model.h5"
    # Stop training after 250 consecutive epochs where loss didn't decrease
    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='loss', patience=100)
    # Save weights of model into file after every X batch_steps
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        save_weights_only=True,
        monitor='loss',
        save_best_only=True,
        save_freq=100)

    print("Starting training for " + str(len(fovs)) + " data samples")
    history = model.fit(dataset, epochs=MAX_EPOCHS, callbacks=[early_stopping_callback, model_checkpoint_callback, LossCallback()])
    sample_count = len(fovs)


dt = datetime.datetime.now() - start_time
t = format_time(dt.total_seconds())

print("----- Training model finished after " + t + " for " + str(sample_count) + " data tuples-----\n")