import copy
import math
import retro
import gymnasium
import pyglet
import time
import numpy as np
import random
import os
import glob
import logging

class RenderSkip(gymnasium.Wrapper):
    def __init__(self, env, render_every=4):
        super().__init__(env)
        self.render_every = render_every
        self.render_count = 0

    def step(self, action):
        want_render = False
        if self.render_count % self.render_every == 0:
            want_render = True
        self.render_count += 1
        return self.env.step(action, want_render=want_render)

    #def render(self):
        #if self.render_count %self.render_every == 0:
            #return super().render()
        #self.render_count += 1


class StochasticFrameSkip(gymnasium.Wrapper):
    '''
    Taken from <https://stable-retro.farama.org/>.
    '''
    def __init__(self, env, n, stickprob):
        gymnasium.Wrapper.__init__(self, env)
        self.n = n
        self.stickprob = stickprob
        self.curac = None
        self.rng = np.random.RandomState()
        self.supports_want_render = False #hasattr(env, "supports_want_render")

    def reset(self, **kwargs):
        self.curac = None
        return self.env.reset(**kwargs)

    def step(self, ac):
        terminated = False
        truncated = False
        totrew = 0
        for i in range(self.n):
            # First step after reset, use action
            if self.curac is None:
                self.curac = ac
            # First substep, delay with probability=stickprob
            elif i == 0:
                if self.rng.rand() > self.stickprob:
                    self.curac = ac
            # Second substep, new action definitely kicks in
            elif i == 1:
                self.curac = ac
            if self.supports_want_render and i < self.n - 1:
                ob, rew, terminated, truncated, info = self.env.step(
                    self.curac,
                    want_render=False,
                )
            else:
                ob, rew, terminated, truncated, info = self.env.step(self.curac)
            totrew += rew
            if terminated or truncated:
                break
        return ob, totrew, terminated, truncated, info


class ObserveVariables(gymnasium.Wrapper):
    '''
    Remove the screen from the observations,
    and use the variables defined in the game's `data.json`.
    This makes an MlpPolicy suitable for training instead of the CnnPolicy,
    which uses much less compute.
    The downside is that it requires manually adding quality features to `data.json`.
    '''
    def __init__(self, env):
        super().__init__(env)
        low = int(self.observation_space.low_repr)
        high = int(self.observation_space.high_repr)
        shape = len(self.unwrapped.data.lookup_all()),
        dtype = self.observation_space.dtype
        self.observation_space = gymnasium.spaces.Box(low, high, shape, dtype, seed=0)
        logging.info(f'ObserveVariables.shape={shape}')

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        return self.observation_space.sample(), info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        newobservation = np.array(list(self.env.data.lookup_all().values()))
        return newobservation, reward, terminated, truncated, info


class RandomStateReset(gymnasium.Wrapper):
    '''
    Whenever reset() is called, restart at a random state instead of the scenario's predefined state.
    This is useful when all the predefined states have some bias on what good behavior is,
    and starting at random states can smooth over that bias.
    '''
    def __init__(self, env, path, globstr=None):
        super().__init__(env)
        self.path = path
        if globstr:
            self.globstr = globstr

    def reset(self, **kwargs):
        results = super().reset(**kwargs)
        states = [os.path.basename(filepath).split('.')[0] for filepath in glob.glob(self.path + '/' + self.globstr)]
        newstate = random.choice(states)
        logging.info(f"newstate={newstate}")
        self.unwrapped.load_state(newstate, inttype=retro.data.Integrations.ALL)
        return results
