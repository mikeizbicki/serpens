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
from collections import OrderedDict, defaultdict
import pprint

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ObjectCNN(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gymnasium.Space,
        features_dim: int = 64,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "NatureCNN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        print(f"observation_space={observation_space}")
        asd
        self.cnn = nn.Sequential(
            nn.Conv1d(n_input_channels, features_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(features_dim, features_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(features_dim, features_dim, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


class KnowledgeBase:
    def __init__(self):
        self.items = defaultdict(lambda: {})
        self.columns = set()

    def __setitem__(self, name, val):
        self.items[name] = val
        for k in val.keys():
            self.columns.add(k)

    def to_observation(self):
        columns = sorted(self.columns)
        stuff = []
        for item, val in self.items.items():
            stuff.append([])
            for column in columns:
                stuff[-1].append(val[column])
        #return stuff
        stuff += [[0] * 6] * (20 - len(stuff))
        return np.array(stuff, dtype=np.float16)

    def display(self):
        columns = sorted(self.columns)
        columns_lens = [max(len(column), 5) for column in columns]
        item_len = 20 #max([len(item) for item in self.items])
        ret = ''
        ret += ' '*item_len
        for column, column_len in zip(columns, columns_lens):
            ret += f' {column:>{column_len}}'
        ret += '\n'
        for item,val in self.items.items():
            ret += f'{item:>{item_len}}'
            for column, column_len in zip(columns, columns_lens):
                s = '-'
                if column in val:
                    s = val[column]
                if isinstance(s, (np.floating, float)):
                    ret += f' {s:>+0.{column_len-3}f}'
                else:
                    ret += f' {s:>{column_len}}'

            ret += '\n'
        return ret

def _ram2kb(ram):
    '''
    Enemy Type:
    There are many different memory locations that seem to store the enemy type;
    I'm not sure about what is different about them,
    and the particular location I chose may be wrong in some subtle situations.
    The values represent:
    00= NOTHING
    01= B. LION GUYS
    02= R. LION GUYS
    03= B. MOBLINS
    04= R. MOBLINS
    05= B. RATS
    06= R. RATS
    07= SLOW R. OCTOROCKS
    08= FAST R. OCTOROCKS
    09= SLOW B. OCTOROCKS
    0A= FAST B. OCTOROCKS
    0B= R. DARKNUTS
    0C= B. DARKNUTS
    0D= B. TECHTIKES
    0E= R. TECHTIKES
    0F= R. LEVERS
    10= B. LEVERS
    11= ZORAS
    12= VAMPIERS
    13= BIG SLIMES
    14= LIL, SLIMES
    15= LIL, SLIMES
    16= POLS VOICES
    17= LIKE LIKES
    18= ????
    19= ???
    1A= PEAHATS
    1B= B.BATS
    1C= R. BATS
    1D= B. BATS
    1E= ????
    1F= FALLING ROCKS
    20= R. LEVERS
    21= MUMMIES
    22= ????
    23= B. WIZROBES
    24= R. WIZROBES
    25= ????
    26= ????
    27= WALL MASTERS
    28= ROPES (SNAKES)
    29= ????
    2A= STALFOSE
    2B= BUBBLES
    2C= B. BUBBLES
    2D= R. BUBBLES
    2E= ????
    2F= FAIRY AT POND
    30= ????
    31= 3 DODONGOS
    32= 1 DODONGO
    33= B. GHOMA
    34= R. GHOMA
    35= RUPIE STASH
    36= GRUMBLE
    37= ZELDA
    38= DIGDOGER?
    39= DIGDOGER?
    3A= 2 R. WORMS
    3B= 2 B. WORMS
    3C= MANHANDELA
    3D= AQUAMENTIS
    3E= GANNON
    3F= FIRE
    '''
    kb = KnowledgeBase()

    # link info
    item = {}
    item['x'] = ram[112]
    item['y'] = ram[132]
    item['type'] = -1
    item['state'] = ram[172]
    item['direction'] = ram[152]
    if ram[1648] == 0:
        item['health'] = 0.0
    else:
        partial = 0.5
        if ram[1648] > 127:
            partial = 0
        item['health'] = float(ram[1647] - math.floor(ram[1647] / 16) * 16 - partial + 1)
    kb['link'] = item

    # sword info
    item = {}
    item['state'] = ram[185]
    item['direction'] = ram[165]
    item['type'] = -2
    item['health'] = 0
    item['x'] = ram[125]
    item['y'] = ram[145]
    if item['state'] != 0:
        kb['sword_melee'] = item

    item = {}
    item['state'] = ram[186]
    item['direction'] = ram[166]
    item['type'] = -2
    item['health'] = 0
    item['x'] = ram[126]
    item['y'] = ram[146]
    if item['state'] != 0:
        kb['sword_proj'] = item

    # enemy info
    for i in range(6):
        item = {}
        #item['countdown'] = ram[41+i]
        item['state'] = ram[173+i]
        item['direction'] = ram[153+i]
        item['type'] = ram[848+i]
        item['x'] = ram[113+i]
        item['y'] = ram[133+i]
        rawhealth = ram[1158+i]
        if rawhealth > 0 and (rawhealth < 16 or rawhealth >= 128):
            item['health'] = 0
        else:
            item['health'] = rawhealth//16
        if item['type'] > 0:
            kb[f'enemy_{i}'] = item

    # projectile info
    for i in range(6):
        item = {}
        item['direction'] = ram[159+i]
        item['state'] = ram[179+i]
        item['type'] = -3
        item['health'] = 0
        item['x'] = ram[119+i]
        item['y'] = ram[139+i]
        if item['state'] != 0:
            kb[f'projectile_{i}'] = item

    # normalize and center all item positions
    for item, val in kb.items.items():
        # center on link
        if item != 'link':
            link_x = int(ram[112])
            link_y = int(ram[132])
        else:
            link_x = 0
            link_y = 0
        kb.items[item]['x'] -= link_x
        kb.items[item]['y'] -= link_y

        # translate screen coordinates to map coordinates
        # (i.e. account for the black bar on top)
        kb.items[item]['y'] -=  61

        # normlalize
        kb.items[item]['x'] /= 240
        kb.items[item]['y'] /= 160
        # NOTE:
        # the NES x resolution is 256,
        # but all sprites are 16 pixels wide,
        # so there are 240 available for positioning;
        # the y resolution is 224,
        # but the top vertical bar uses 56 pixels,
        # and only 8 pixels are reserved for "link positioning"
        # because he does not go "partially off the edge" on the top screen

    return kb


class RetroWithRam(gymnasium.Wrapper):
    '''
    The .get_ram() function is relatively expensive.
    The only purpose of this wrapper is to store the result of .get_ram() in an attribute
    to prevent multiple calls.
    '''
    def step(self, action):
        self.ram2 = self.ram
        self.ram = self.env.get_ram()
        return super().step(action)


class ZeldaWrapper(RetroWithRam):
    '''
    '''
    def __init__(
            self,
            env,
            stdout_debug=False,
            use_full_subtiles=False,
            link_view_radius=2,
            center_xy=True,
            output_link_view=False,
            normalize_output=True,
            reorder_outputs=True,
            clean_outputs=True,
            no_render_skipped_frames=True,
            skip_boring_frames=True,
            ):

        # bookkeeping
        super().__init__(env)
        self.stdout_debug = stdout_debug
        self.use_full_subtiles = use_full_subtiles
        self.link_view_radius = link_view_radius
        self.center_xy = center_xy
        self.output_link_view = output_link_view
        self.observations_keys = None
        self.normalize_output = normalize_output
        self.reorder_outputs = reorder_outputs
        self.clean_outputs = clean_outputs
        self.no_render_skipped_frames=no_render_skipped_frames
        self.skip_boring_frames = skip_boring_frames 
        self.mousex = 0
        self.mousey = 0

        # these variables track what information in the info dict will be included in the observations 
        self.keep_prefixes = ['enemy', 'link_char', 'link_sword', 'projectile']
        self.keep_suffixes = ['_x', '_y', 'drop', 'count', '_direction_', 'health', 'state', 'type']

        # switch the terminal to the alternate screen
        if self.stdout_debug:
            print('\u001B[?1049h')

        # create a new observation space
        #info = self.generate_info()
        #observations = self.info_to_observations(info)
        #if normalize_output:
            #low = -1
            #high = 1
        #else:
            #low = -256
            #high = 256
        #shape = [len(observations)]
        #self.observation_space = gymnasium.spaces.Box(low, high, shape, dtype, seed=0)
        dtype = np.float16
        low = -1
        high = 1
        shape = [20, 6]
        self.observation_space = gymnasium.spaces.Box(low, high, shape, dtype, seed=0)
        #shape = [6]
        #self.observation_space = gymnasium.spaces.Sequence(
                #gymnasium.spaces.Box(low, high, shape, dtype, seed=0),
                #seed=0,
                #stack=True)
        logging.info(f'observations.shape={shape}')

    def close(self):
        # switch the terminal away from the alternate screen
        if self.stdout_debug:
            print('\u001B[?1049l')

        # cleanly close superclass
        super().close()

    def reset(self, **kwargs):
        self.episode_reward_dict = {}
        self.episode_reward = 0
        obs, info = super().reset(**kwargs)
        return self.observation_space.sample(), info

    def step(self, action):

        observation, reward, terminated, truncated, info = super().step(action)
        #self.ram2 = self.ram
        #self.ram = self.env.get_ram()

        kb = _ram2kb(self.ram)
        observation = kb.to_observation()
        terminated = any([
            _gamestate_all_enemies_dead(self.ram),
            _gamestate_hearts(self.ram) <= 0,
            _gamestate_is_screen_scrolling(self.ram),
            _gamestate_is_cave_enter(self.ram),
            ])

        info['is_success'] = _gamestate_all_enemies_dead(self.ram)
        info['summary_gamestate_hearts(self.ram)'] = _gamestate_hearts(self.ram)
        info['summary_gamestate_is_screen_scrolling(self.ram)'] = _gamestate_is_screen_scrolling(self.ram)
        info['summary_gamestate_is_cave_enter(self.ram)'] = _gamestate_is_cave_enter(self.ram)
        info['summary_gamestate_all_enemies_dead(self.ram)'] = _gamestate_all_enemies_dead(self.ram)

        reward_dict = {
            'scrolling': 0,
            'enemy_hit': 0,
            'link_hit': 0,
            'link_dead': 0,
            'all_enemies': 0,
            }
        if self.ram2 is not None:
            if _gamestate_is_screen_scrolling(self.ram) or _gamestate_is_cave_enter(self.ram):
                reward_dict['scrolling'] = -2
            reward_dict['enemy_hit'] = max(0, _gamestate_all_enemies_health(self.ram2) - _gamestate_all_enemies_health(self.ram))
            if _gamestate_all_enemies_health(self.ram) == 0 and _gamestate_all_enemies_health(self.ram2) != 0:
                reward_dict['all_enemies'] = 2
            if _gamestate_hearts(self.ram) == 0 and _gamestate_hearts(self.ram2) != 0:
                reward_dict['link_dead'] = -2
            reward_dict['link_hit'] = -1 * round(2*max(0, _gamestate_hearts(self.ram2) - _gamestate_hearts(self.ram)))
        
        reward = sum(reward_dict.values())

        self.episode_reward += reward
        self.episode_reward_dict = {k: self.episode_reward_dict.get(k, 0) + reward_dict.get(k, 0) for k in set(self.episode_reward_dict) | set(reward_dict)}

        if self.stdout_debug:
            text = ''
            text += '\x1b[2J' # clear the screen
            #text += _observations_to_str(new_observations)
            text += kb.display()
            #text += f'\ntotal_variables: {len(new_observations)}'
            text += f'\nterminated = {terminated}'
            text += f'\nepisode_reward = {self.episode_reward:0.4f}'
            text += f'\nepisode_reward_dict = {pprint.pformat(self.episode_reward_dict)}'
            text += f'\nmouse (x, y) = {self.mousex:0.2f}, {self.mousey:0.2f}'
            print(text)

        # skip the boring frames
        if self.skip_boring_frames:

            # record the environment's render mode so that we can restore it after skipping
            render_mode = self.env.render_mode
            if self.no_render_skipped_frames:
                self.env.render_mode = None
            skipped_frames = 0

            while any([
                _gamestate_is_screen_scrolling(self.ram),
                _gamestate_is_drawing_text(self.ram),
                _gamestate_is_cave_enter(self.ram),
                _gamestate_is_inventory_scroll(self.ram),
                _gamestate_hearts(self.ram) <= 0,
                _gamestate_is_openning_scene(self.ram),
                ]):
                # normally we will not press any action to the environment;
                # but if link has died, we need to press the "continue" button;
                # to do this, we alternate between no action and pressing "start" every frame
                skipped_frames += 1
                skipaction = [False]*8
                if _gamestate_hearts(self.ram) <= 0 and skipped_frames%2 == 0:
                    skipaction[3] = True
                super().step(skipaction)
                #self.ram = self.env.get_ram()
            self.env.render_mode = render_mode

        # return step results
        return observation, reward, terminated, truncated, info

# The _gamestate functions modified from gym-zelda-1 repo;
# that repo uses the nesgym library, which is a pure python nes emulator
# and about 10x slower than the gymretro environment;
def _gamestate_is_screen_scrolling(ram):
    SCROLL_GAME_MODES = {4, 6, 7}
    return ram[0x12] in SCROLL_GAME_MODES

def _gamestate_is_drawing_text(ram):
    return ram[0x0605] == 0x10

def _gamestate_is_cave_enter(ram):
    return ram[0x0606] == 0x08

def _gamestate_is_inventory_scroll(ram):
    return 65 < ram[0xfc]

def _gamestate_is_openning_scene(ram):
    return bool(ram[0x007c])

def _gamestate_hearts(ram):
    link_full_hearts = 0x0f & ram[0x066f]
    link_partial_hearts = ram[0x0670] / 255
    return link_full_hearts + link_partial_hearts

def _gamestate_all_enemies_dead(ram):
    return all([ram[848+i] == 0 for i in range(6)])

def _gamestate_all_enemies_health(ram):
    healths = []
    for i in range(6):
        rawhealth = ram[1158+i]
        if rawhealth > 0 and (rawhealth < 16 or rawhealth >= 128):
            healths.append(0)
        else:
            healths.append(rawhealth//16)
    return sum(healths)

def _gamestate_screen_change(ram):
    return ram[18] == 5
