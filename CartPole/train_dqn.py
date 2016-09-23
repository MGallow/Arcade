'''This script is modified from the original dqn_cartpole.py script in the examples section of the Keras-rl package. See: https://github.com/matthiasplappert/keras-rl'''

ENV_NAME = "CartPole-v0"

Internet = True  # connected to the internet (Slack)?
Watch = False
Test = False
Record = False
loaded_model = None
loaded_weights = None
nb_steps = 500000
nb_test_episodes = 10
nb_steps_warmup = 10
model_name = ENV_NAME + "-original_model"
arch_name = model_name + ".json"


print("Using Internet:", Internet)
print("Environment:", ENV_NAME)
print("Watching:", Watch)
print("Testing:", Test)
print("Record:", Record)

# access locations file
exec(open("Locations.py").read())
print("locations successfully read.")


# dependencies
import os
import sys
import numpy as np
import gym
from PIL import Image
from keras.models import Sequential, model_from_json
from keras.regularizers import l2
from keras.initializations import normal, uniform
from keras.optimizers import SGD, rmsprop, Adam
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense

sys.path.append("..")
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor


def save_weights(which_agent, filename):
    run = os.path.join(runs_dir, filename)
    which_agent.save_weights(run, overwrite=True)
    print("weights saved as", filename)


def save_architecture(archname, string):
    arch = os.path.join(arch_dir, archname)
    open(arch, 'w').write(string)
    print("architecture saved as", archname)


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n


# Next, we build the model.
def construct_model():
    if loaded_model is not None:
        # load previous architecture
        model_location = os.path.join(arch_dir, loaded_model)
        model = model_from_json(open(model_location).read())
        if loaded_weights is not None:
            # load previous weights
            weights_location = os.path.join(runs_dir, loaded_weights)
            model.load_weights(weights_location)
    else:
        model = Sequential()
        model.add(Flatten(input_shape=(1,) +
                          env.observation_space.shape))  # (1, 4)
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(nb_actions))
        model.add(Activation('linear'))
        json_string = model.to_json()
        save_architecture(arch_name, json_string)
    print("model constructed.")
    return model


#callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]


# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
def train_model():

    # construct model
    model = construct_model()

    # configure and compile agent
    memory = SequentialMemory(limit=50000)
    policy = BoltzmannQPolicy()
    agent = DQNAgent(model=model,
                     nb_actions=nb_actions,
                     memory=memory,
                     nb_steps_warmup=nb_steps_warmup,
                     target_model_update=1e-2,
                     policy=policy)
    agent.compile(optimizer=Adam(lr=1e-3), metrics=['mae'])  # 'rmsprop

    # train the model
    agent.fit(env, nb_steps=nb_steps, visualize=Watch, verbose=2)

    # After training is done, we save the final weights.
    save_weights(which_agent=agent, filename=model_name + ".h5f")

    # Do we want to test the model?
    if Test:
        if Record:
            vid_location = os.path.join(video_dir, model_name)
            env.monitor.start(
                vid_location, video_callable=lambda count: count % 1 == 0, force=True)
            agent.test(env, nb_episodes=nb_test_episodes, visualize=True)
            env.monitor.close()
        else:
            agent.test(env, nb_episodes=nb_test_episodes)

    if Internet:
        # send message to slack when finished
        slck = os.path.join(slack_dir, "Slack.py")
        exec(open(slck).read())
        print("message sent to Slack")
    return model


# execute the training!
train_model = train_model()


print("train_dqn.py complete.")
