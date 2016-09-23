'''(MODIFIED)'''

'''This script was highly inspired from the following tutorial: https://yanpanlau.github.io/2016/07/10/FlappyBird-Keras.html'''

'''Play flappy bird using the deep-Q learning algorithm'''

# which game are we playing?
# Note that the flappy bird environment is included in the Github repo
ENV_NAME = "flappy_bird"


loaded_model = "flappy_bird-convo_model.json"
loaded_weights = "flappy_bird-convo_model.h5"
model_name = ENV_NAME + "-convo_model"
arch_name = model_name + ".json"

Train = False
nb_actions = 2  # number of valid actions
gamma = 0.99  # decay rate of past Observations
explore = 3000000  # frames over which to anneal epsilon
observe = 5000  # how many steps to observe before training?
final_epsilon = 0.0001  # final value of epsilon
initial_epsilon = 0.05  # starting value of epsilon
memory = 50000  # number of previous transitions to remember
batch = 50  # size of minibatch


print("Environment:", ENV_NAME)
print("Training:", Train)

# access locations file
exec(open("Locations.py").read())
print("locations successfully read.")


#import dependencies
import pygame
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque

import json
from keras import initializations
from keras.initializations import normal, identity
from keras.models import Sequential, model_from_json
from keras.regularizers import l2
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam, rmsprop


img_rows, img_cols = 80, 80
img_channels = 4  # We stack 4 frames


# access locations file
exec(open("Locations.py").read())
print("locations successfully read.")


# define various modules for reproducibility
def save_weights(which_model, filename):
    run = os.path.join(runs_dir, filename)
    which_model.save_weights(run, overwrite=True)
    print("weights saved as", filename)


def save_architecture(archname, string):
    arch = os.path.join(arch_dir, archname)
    open(arch, 'w').write(string)
    print("architecture saved as", archname)


# construct the model
def construct_model():
    if loaded_model is not None:
        # load previous architecture
        model_name = loaded_model
        model_location = os.path.join(arch_dir, loaded_model)
        model = model_from_json(open(model_location).read())
        if loaded_weights is not None:
            # load previous weights
            weights_location = os.path.join(runs_dir, loaded_weights)
            model.load_weights(weights_location)

    else:
        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=(
            img_rows, img_cols, img_channels)))
        model.add(Convolution2D(nb_filter=32,
                                nb_row=3,
                                nb_col=3,
                                init='normal',
                                border_mode='valid'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3),
                               strides=(2, 2),
                               border_mode='valid'))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(64, 3, 3,
                                init='normal',
                                border_mode='valid'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3),
                               strides=(2, 2),
                               border_mode='valid'))

        model.add(Flatten())
        model.add(Dense(output_dim=512,
                        input_dim=True,
                        init='normal',  # initialize the weights
                        bias=True,
                        W_regularizer=l2(0.01)
                        ))
        model.add(Activation('relu'))

        model.add(Dense(output_dim=nb_actions, init='normal'))

        # save model
        json_string = model.to_json()
        save_architecture(arch_name, json_string)
    print("model constructed.")
    return model


# preprocess the images
def preprocess(color):
    x = skimage.color.rgb2gray(color)
    x = skimage.transform.resize(x, (80, 80))
    x = skimage.exposure.rescale_intensity(x, out_range=(0, 255))
    return x


# train the model
def train_model():
    flappy = game.GameState()
    D = deque()

    # initialize values
    action_index = 0
    t = 0
    loss = 0
    epsilon = initial_epsilon
    # initial action. Note that [0, 1] = flap and [1, 0] = don't flap
    action_t0 = np.zeros(nb_actions)
    action_t0[0] = 1

    x_t0_colored, r_t0, terminal = flappy.frame_step(action_t0)
    x_t0 = preprocess(x_t0_colored)
    # going forward, the frames will be replaced with incoming images
    s_0 = np.stack((x_t0, x_t0, x_t0, x_t0), axis=0)
    s_t = s_0.reshape(1, s_0.shape[2], s_0.shape[1], s_0.shape[0])

    # train the model
    while True:
        state = "Observing"
        loss = 0
        max_Q = 0
        action_index = 0
        r_t = 0
        a_t = np.zeros([nb_actions])

        if random.random() <= epsilon:
            print("----------Random Action----------")
            action_index = random.randrange(nb_actions)
            a_t[action_index] = 1
        else:
            # input a stack of 4 images, get the prediction of the Q-function
            q = model.predict(s_t)
            # choose action with highest prediction of the Q-function
            action_index = np.argmax(q)
            a_t[action_index] = 1

        # run the selected action and observed next state and reward
        x_t_colored, r_t, terminal = flappy.frame_step(a_t)
        x_t = preprocess(x_t_colored)
        x_t = x_t.reshape(1, x_t.shape[0], x_t.shape[1], 1)
        s_t1 = np.append(x_t, s_t[:, :, :, :3], axis=3)

        # store the transition in D
        D.append((s_t, action_index, r_t, s_t1, terminal))
        if len(D) > memory:
            D.popleft()

        # train on minibatches. We only train after observation period is over
        # so that there are sufficient number of frames to sample from
        if t > observe and terminal:
            # sample a minibatch to train on
            minibatch = random.sample(D, batch)

            inputs = np.zeros((batch, s_t.shape[1], s_t.shape[
                              2], s_t.shape[3]))  # 50, 80, 80, 4
            targets = np.zeros((inputs.shape[0], nb_actions))  # 50, 2

            # Now we do the experience replay
            for i in range(0, len(minibatch)):
                state_t = minibatch[i][0]
                action_t = minibatch[i][1]
                reward_t = minibatch[i][2]
                state_t1 = minibatch[i][3]
                terminal = minibatch[i][4]
                inputs[i:i + 1] = state_t
                targets[i] = model.predict(state_t)
                Q_sa = model.predict(state_t1)
                max_Q = np.max(Q_sa)

                if terminal:
                    targets[i, action_t] = reward_t
                else:
                    targets[i, action_t] = reward_t + gamma * max_Q

            loss += model.train_on_batch(inputs, targets)

            # save progress every 1000 iterations
            print("Minibatch complete. Loss: ", loss)
            print("saving weights.")
            save_weights(model, model_name + ".h5")

        s_t = s_t1
        t = t + 1
        # reduce epsilon gradually
        if epsilon > final_epsilon:
            epsilon -= (initial_epsilon - final_epsilon) / explore

        if t > observe:
            state = "Training"

        print("TIMESTEP", t, "/ STATE", state,
              "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, "/ TERMINAL", terminal)


# test the model
def test_model():
    flappy = game.GameState()

    # INITIAL
    # initialize values
    action_index = 0
    t = 0
    loss = 0
    epsilon = final_epsilon
    # initial action. Note that [0, 1] = flap and [1, 0] = don't flap
    action_t0 = np.zeros(nb_actions)
    action_t0[0] = 1

    x_t0_colored, r_t0, terminal = flappy.frame_step(action_t0)
    x_t0 = preprocess(x_t0_colored)
    # going forward, the frames will be replaced with incoming images
    s_0 = np.stack((x_t0, x_t0, x_t0, x_t0), axis=0)
    s_t = s_0.reshape(1, s_0.shape[2], s_0.shape[1], s_0.shape[0])

    # train the model
    while True:
        state = "Testing"
        loss = 0
        max_Q = 0
        action_index = 0
        r_t = 0
        a_t = np.zeros([nb_actions])

        # input a stack of 4 images, get the prediction of the Q-function
        q = model.predict(s_t)
        # choose action with highest prediction of the Q-function
        action_index = np.argmax(q)
        a_t[action_index] = 1

        # run the selected action and observed next state and reward
        x_t_colored, r_t, terminal = flappy.frame_step(a_t)
        x_t = preprocess(x_t_colored)
        x_t = x_t.reshape(1, x_t.shape[0], x_t.shape[1], 1)
        s_t1 = np.append(x_t, s_t[:, :, :, :3], axis=3)

        s_t = s_t1
        t = t + 1

        print("TIMESTEP", t, "/ STATE", state,
              "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, "/ TERMINAL", terminal)


# construct model
model = construct_model()
# compile the model
model.compile(loss='mse', optimizer='rmsprop')


if Train:
    train_model()
else:
    test_model()
