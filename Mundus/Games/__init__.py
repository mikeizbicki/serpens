import copy
import glob
import logging
import math
import numpy as np
import os
import pprint
import pyglet
import re
import time

import retro
import gymnasium
from gymnasium import spaces
from Mundus.Agent.Zelda import *
from Mundus.Retro import *
from Mundus.Object import *
from Mundus.util import *
from Mundus.Wrappers import *
from Mundus.Games.Zelda import *
import Mundus.Games.Zelda


def make_game_env(
        game='Zelda',
        action_space='all',
        render_mode='human',
        fork_emulator=False,
        interactive=False,
        reset_state=None,
        **kwargs):
    '''
    Create a Zelda environment.

    NOTE:
    I've arranged everything into a single function here
    to avoid inconsistencies between the train/play code that were happening.
    This style doesn't follow the standard for stable-retro,
    and could probably be improved.
    '''
    if 'seed' in kwargs:
        seed = kwargs['seed']
    else:
        seed = 0

    # set the folder where retro.make() looks for ROMs
    custom_path = os.path.join(os.getcwd(), 'custom_integrations')
    retro.data.Integrations.add_custom_path(custom_path)

    if action_space == 'DISCRETE':
        use_restricted_actions = retro.Actions.DISCRETE
    elif action_space == 'MULTI_DISCRETE':
        use_restricted_actions = retro.Actions.MULTI_DISCRETE
    else:
        use_restricted_actions = retro.Actions.ALL

    # create the environment
    env = retro.make(
            game=game + '-Nes',
            #game='Zelda-Nes',
            #game='MyMario-Nes',
            #game='GauntletII-Nes',
            #game='Zelda-PerilsOfDarkness-Nes',
            inttype=retro.data.Integrations.ALL,
            #state='overworld_07',
            render_mode=render_mode,
            use_restricted_actions=use_restricted_actions,
            )
    if fork_emulator:
        env = ForkedRetroEnv(env)
    if 'Zelda' in game:
        env = ZeldaWrapper(env, **kwargs)
        if interactive:
            env = ZeldaInteractive(env)
            env = Agent(env, SimpleNavigator, RandomObjective)
            env = UnstickLink(env)
    else:
        env = RetroKB(env, **kwargs)

    if reset_state is not None:
        path = f'custom_integrations/{env.unwrapped.gamename}'
        env = RandomStateReset(env, path, reset_state, seed)

    # apply zelda-specific action space
    if 'zelda-' in action_space:
        kind = action_space.split('-')[1]
        env = ZeldaActionSpace(env, kind)

    logging.debug(f"env.action_space={env.action_space}")
    logging.debug(f"env.action_space.n={getattr(env.action_space, 'n', None)}")
    return env

