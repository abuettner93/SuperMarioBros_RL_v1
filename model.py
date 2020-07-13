# INITIALIZATION: libraries, parameters, network...
import keras
from keras.models import model_from_json
from keras.models import Sequential  # One layer after the other
from keras.layers import Dense, Flatten, \
    Conv2D  # Dense layers are fully connected layers, Flatten layers flatten out multidimensional inputs
from collections import deque  # For storing moves
import time
import random
from skimage import io

import numpy as np


class Model:
    def __init__(self, environment, model_type="DNN", filename=""):
        self.env = environment.env
        # Parameters
        self.games_to_train_on = 1
        self.model_type = model_type
        self.observetime = 1000  # Number of timesteps we will be acting on the game and observing results
        self.D = deque(
            maxlen=100000)  # Register where the actions will be stored

        self.epsilon = 0.7  # Probability of doing a random move
        self.epsilon_min = 0.3
        self.gamma = 0.99  # Discounted future reward. How much we care about steps further in time
        self.mb_size = 50  # Learning minibatch size
        self.filename = filename
        self.model = self.define_ml_model(self.model_type, self.filename)
        # print((2,) + self.env.observation_space.shape)

    def define_ml_model(self, model_type, filename=""):
        if model_type == "CNN":
            # Create CNN-assisted network
            model = Sequential()
            model.add(Conv2D(filters=32, kernel_size=(8, 8), strides=4, activation="relu", input_shape=(2, 240, 256)))
            model.add(Conv2D(filters=64, kernel_size=(4, 4), strides=2, activation="relu"))
            model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation="relu"))
            model.add(Dense(512, activation="relu"))
            model.add(Dense(self.env.action_space.n))
            model.summary()

        elif model_type == "DNN":
            # Create network. Input is two consecutive game states, output is Q-values of the possible moves.
            model = Sequential()
            model.add(Dense(20, input_shape=(2,) + self.env.observation_space.shape, kernel_initializer='uniform',
                            activation='relu'))
            model.add(Flatten())  # Flatten input so as to have no problems with processing
            model.add(Dense(18, kernel_initializer='uniform', activation='relu'))
            model.add(Dense(10, kernel_initializer='uniform', activation='relu'))
            model.add(Dense(self.env.action_space.n, kernel_initializer='uniform',
                            activation='linear'))  # Same number of outputs as possible actions
            opt = keras.optimizers.Adam(learning_rate=0.005)
            model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
            model.summary()

        elif model_type == "load":
            # load json and create model
            json_file = open(f"{filename}.json", 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)

            # load weights into new model
            model.load_weights(f"{filename}.h5")
            print("Loaded model from disk")

            model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
            model.summary()

        else:
            print("model type given is not defined!")
            exit()
        return model

    def train_model(self, total_games=30):
        for game_ind in range(total_games):
            observation = self.env.reset()  # Game begins
            obs = np.expand_dims(observation,
                                 axis=0)  # (Formatting issues) Making the observation the first element of a batch of inputs
            state = np.stack((obs, obs), axis=1)
            done = False
            #     for t in range(observetime):
            counter = 0
            epsilon = self.epsilon
            max_frames = 1000
            obs_start = time.time()

            while not done and counter < max_frames:
                # epsilon = (self.epsilon_min - self.epsilon)/max_frames*(counter) + epsilon
                if np.random.rand() <= self.epsilon:
                    action = np.random.randint(0, self.env.action_space.n, size=1)[0]
                else:
                    Q = self.model.predict(state)  # Q-values predictions
                    action = np.argmax(Q)  # Move with highest Q-value is the chosen one
                observation_new, reward, done, info = self.env.step(
                    action)  # See state of the game, reward... after performing the action

                obs_new = np.expand_dims(observation_new, axis=0)  # (Formatting issues)
                state_new = np.append(np.expand_dims(obs_new, axis=0), state[:, :1, :],
                                      axis=1)  # Update the input with the new state of the game
                self.D.append((state, action, reward, state_new, done))  # 'Remember' action and consequence
                state = state_new  # Update state
                if done:
                    self.env.reset()  # Restart game if it's finished
                    obs = np.expand_dims(observation,
                                         axis=0)  # (Formatting issues) Making the observation the first element of a batch of inputs
                    state = np.stack((obs, obs), axis=1)
                counter += 1
            obs_end = time.time()

            # SECOND STEP: Learning from the observations (Experience replay)
            #     print(epsilon)
            if game_ind % self.games_to_train_on == 0:
                train_start = time.time()
                for num in range(3):
                    mb_size = min(len(self.D), 50)
                    print("mb_size", mb_size)
                    minibatch = random.sample(self.D, mb_size)  # Sample some moves

                    inputs_shape = (mb_size,) + state.shape[1:]
                    inputs = np.zeros(inputs_shape)
                    targets = np.zeros((mb_size, self.env.action_space.n))

                    reward_mean = np.mean(minibatch, axis=0)[2]
                    # print(reward_mean)

                    for i in range(0, mb_size):
                        if self.model_type == "CNN":
                            rgb_weights = [0.2980, 0.5870, 0.1140]
                            state = np.dot(minibatch[i][0][..., :3], rgb_weights)
                            state_new = np.dot(minibatch[i][3][..., :3], rgb_weights)
                        else:
                            state = minibatch[i][0]
                            state_new = minibatch[i][3]

                        action = minibatch[i][1]
                        reward = minibatch[i][2] - reward_mean
                        done = minibatch[i][4]

                        # Build Bellman equation for the Q function
                        inputs[i:i + 1] = np.expand_dims(state, axis=0)
                        targets[i] = self.model.predict(state)
                        Q_sa = self.model.predict(state_new)

                        if done:
                            targets[i, action] = reward
                        else:
                            targets[i, action] = reward + self.gamma * np.max(Q_sa)
                    # Train network to output the Q function
                    history = self.model.train_on_batch(inputs, targets)
                    # print("HISTORY", history)
                train_end = time.time()

                model_json = self.model.to_json()
                with open(self.filename + ".json", "w+") as json_file:
                    json_file.write(model_json)
                self.model.save_weights(self.filename + '.h5')

            #     print('Learning Finished')
            print(
                f"Trained on game {game_ind}! Observation time: {obs_end - obs_start}s\tTrain time: {train_end - train_start}s")

        print("Training complete, running a real game now!")
