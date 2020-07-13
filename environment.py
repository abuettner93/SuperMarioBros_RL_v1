from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT


class Environment:
    def __init__(self, version='SuperMarioBros-v0', movement_type=RIGHT_ONLY):
        self.env = gym_super_mario_bros.make(version)
        self.env = JoypadSpace(self.env, movement_type)
