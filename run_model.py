from environment import Environment
from model import Model
import numpy as np
import time
environment = Environment(version="SuperMarioBros-v2")

model = Model(environment, model_type="DNN", filename="saved_models/main_model_right_only_rom_v2")

total_games = 20

start = time.time()
model.train_model(total_games=total_games)
end = time.time()
print(f"Trained {total_games} games in t={(end-start)/60}mins")


observation = model.env.reset()
obs = np.expand_dims(observation, axis=0)
state = np.stack((obs, obs), axis=1)
done = False
tot_reward = 0.0
#     counter = 0
while not done:
    #         print(counter)
    model.env.render()  # Uncomment to see game running
    Q = model.model.predict(state)
    action = np.argmax(Q)
    #         print(game_ind, t, action)
    observation, reward, done, info = model.env.step(action)
    obs = np.expand_dims(observation, axis=0)
    state = np.append(np.expand_dims(obs, axis=0), state[:, :1, :], axis=1)
    tot_reward += reward
#         counter += 1
print('Game ended! Total reward: {}, {}'.format(reward, tot_reward))

model.env.close()
