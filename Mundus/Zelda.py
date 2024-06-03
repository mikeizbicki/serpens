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


class LSTMPool(torch.nn.LSTM):
    def forward(self, x):
        x = x.reshape([x.shape[0], x.shape[2], x.shape[1]])
        out = super().forward(x)
        out = out[1][0]
        out = out.reshape(out.shape[1], out.shape[0] * out.shape[2])
        return out

class MaxPool(torch.nn.Module):
    def forward(self, x):
        out = torch.max(x, 2)[0]
        return out

class MeanPool(torch.nn.Module):
    def forward(self, x):
        out = torch.mean(x, 2)
        return out


class ObjectCnn(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gymnasium.Space,
        features_dim: int = 64,
        pooling: str = 'mean',
    ) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv1d(n_input_channels, features_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(features_dim, features_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(features_dim, features_dim, kernel_size=1),
            nn.ReLU(),
        )
        
        if pooling == 'lstm':
            self.pool = LSTMPool(features_dim, features_dim, batch_first=True, bidirectional=True)
        elif pooling == 'max':
            self.pool = MaxPool()
        elif pooling == 'mean':
            self.pool = MeanPool()
        else:
            raise ValueError(f'pooling type "{pooling}" not supported')

        # Compute shape by doing one forward pass
        with torch.no_grad():
            out = self.cnn(torch.as_tensor(observation_space.sample()[None]).float())
            out = self.pool(out)
            n_flatten = out.shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
            )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        ret = self.cnn(observations)
        ret = self.pool(ret)
        ret = self.linear(ret)
        return ret


class KnowledgeBase:
    def __init__(self, max_objects = 20):
        self.items = defaultdict(lambda: {})
        self.columns = set()
        self.max_objects = max_objects

    def __setitem__(self, name, val):
        if len(self.items) >= self.max_objects:
            logging.warning('len(self.items) >= self.max_objects')
        else:
            self.items[name] = val
            for k, v in val.items():
                self.columns.add(k)
                # NOTE: we convert all integer values away from numpy types to python int
                # because the numpy types are unsigned and small;
                # this can result in hard-to-debug integer overflow problems
                if not isinstance(v, (np.floating, float)):
                    val[k] = int(v)

    def to_observation(self):
        columns = sorted(self.columns)
        stuff = []
        for item, val in self.items.items():
            stuff.append([])
            for column in columns:
                stuff[-1].append(val[column])
        # FIXME:
        # currently we pad the observation space
        stuff += [[0] * len(self.columns)] * (self.max_objects - len(stuff))
        return np.array(stuff, dtype=np.float16).T

    def get_observation_space(self):
        dtype = np.float16
        low = -1
        high = 1
        observation = self.to_observation()
        shape = observation.shape
        return gymnasium.spaces.Box(low, high, shape, dtype, seed=0)

    def display(self):
        columns = sorted(self.columns)
        columns_lens = [max(len(column), 5) for column in columns]
        item_len = 12 #max([len(item) for item in self.items])
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

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.ram2 = None
        self.ram = self.env.get_ram()
        return obs, info


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
            render_kb=True,
            scenario='attack',
            ):

        # bookkeeping
        super().__init__(env)
        self.stdout_debug = stdout_debug
        self.use_full_subtiles = use_full_subtiles
        self.link_view_radius = link_view_radius
        self.center_xy = center_xy
        self.output_link_view = output_link_view
        self.normalize_output = normalize_output
        self.reorder_outputs = reorder_outputs
        self.clean_outputs = clean_outputs
        self.no_render_skipped_frames=no_render_skipped_frames
        self.skip_boring_frames = skip_boring_frames 
        self.render_kb = render_kb
        self.mouse = None

        # create a new observation space
        self.ram = self.env.get_ram()
        kb = self.generate_knowledge_base()
        self.observation_space = kb.get_observation_space()
        logging.info(f'self.observation_space.shape={self.observation_space.shape}')

        # define the scenarios
        self.scenarios = {}
        self.scenarios['attack'] = {
            'terminated': lambda ram: any([
                _gamestate_all_enemies_dead(self.ram),
                _gamestate_link_killed(self.ram),
                _gamestate_is_screen_scrolling(self.ram),
                _gamestate_is_cave_enter(self.ram),
                ]),
            'is_success': lambda ram: any([
                _gamestate_all_enemies_dead(self.ram),
                ]),
            'reward': {
                'link_l1dist': 0,
                'button_push': 0,
                },
            }

        self.scenarios['suicide'] = copy.deepcopy(self.scenarios['attack'])
        self.scenarios['suicide']['is_success'] = lambda ram: any([
                _gamestate_link_killed(self.ram),
                ])
        self.scenarios['suicide']['reward'] = {
            'link_killed': -2,
            'link_hit': -1,
            'link_l1dist': 0,
            }

        self.scenarios['follow'] = copy.deepcopy(self.scenarios['attack'])
        #self.scenarios['follow']['reward'] = defaultdict(lambda: 0)
        self.scenarios['follow']['reward']['link_l1dist'] = 1
        #self.scenarios['follow']['reward']['button_push'] = 1

        def _step_follow_enemy(self, kb):
            enemies = []
            for k in kb.items:
                if 'enemy' in k:
                    enemies.append(kb.items[k])
            if len(enemies) == 0:
                self.mouse = None
            else:
                link = kb.items['link']
                self.mouse = {}
                dist = 9999
                for i, enemy in enumerate(enemies):
                    newdist = abs(enemy['x'] - link['x']) + abs(enemy['y'] - link['y'])
                    if newdist < dist:
                        dist = newdist
                        self.mouse['x'] = enemy['x']
                        self.mouse['y'] = enemy['y']
        self.scenarios['follow_enemy'] = copy.deepcopy(self.scenarios['follow'])
        self.scenarios['follow_enemy']['step'] = _step_follow_enemy

        self.scenarios['repel_enemy'] = copy.deepcopy(self.scenarios['follow_enemy'])
        self.scenarios['repel_enemy']['reward']['link_l1dist'] = -1

        def _step_follow_random(self, kb):
            if random.random() <= 1 / (60 * 5):
                x = random.randrange(0, 255) - 8
                y = random.randrange(60, 240) - 8
                self.mouse = {'x': x, 'y': y}
        self.scenarios['follow_random'] = copy.deepcopy(self.scenarios['follow'])
        self.scenarios['follow_random']['step'] = _step_follow_random

        self.scenarios['repel_random'] = copy.deepcopy(self.scenarios['follow_random'])
        self.scenarios['repel_random']['reward']['link_l1dist'] = -1

        self.scenario = self.scenarios[scenario]


    def reset(self, **kwargs):
        self.episode_summary_dict = {}
        self.episode_reward_dict = {}
        self.episode_reward = 0
        obs, info = super().reset(**kwargs)
        return self.observation_space.sample(), info

    def step(self, action):

        # step the emulator
        render_mode = self.env.render_mode
        self.env.render_mode = 'rgb_array'
        observation, reward, terminated, truncated, info = super().step(action)
        self.env.render_mode = render_mode

        # compute the scenario information
        kb = self.generate_knowledge_base()
        observation = kb.to_observation()

        terminated = self.scenario['terminated'](self.ram)
        info['is_success'] = self.scenario['is_success'](self.ram)
        if 'step' in self.scenario:
            self.scenario['step'](self, kb)

        summary_dict = self.get_rewards()
        reward_dict = {}
        reward = 0
        for k, v in summary_dict.items():
            reward_k = summary_dict[k] * self.scenario['reward'].get(k, 1)
            reward += reward_k
            reward_dict[k] = reward_k

        self.episode_reward += reward
        self.episode_summary_dict = {k: self.episode_summary_dict.get(k, 0) + summary_dict.get(k, 0) for k in set(self.episode_summary_dict) | set(summary_dict)}
        self.episode_reward_dict = {k: self.episode_reward_dict.get(k, 0) + reward_dict.get(k, 0) for k in set(self.episode_reward_dict) | set(reward_dict)}
        for k in summary_dict.keys():
            info['summary_' + k] = self.episode_summary_dict[k]
            info['reward_' + k] = self.episode_reward_dict[k]

        # render the environment
        if self.render_kb:
            for k, v in kb.items.items():
                if k != 'mouse':
                    # NOTE: we apply min/max to all values;
                    # some objects can be located off screen,
                    # and trying to render them directly crashes without clipping
                    xmin = max(0, min(239, v['x'] - 8))
                    xmax = max(0, min(239, v['x'] + 8))
                    ymin = max(0, min(223, v['y'] - 8))
                    ymax = max(0, min(223, v['y'] + 8))
                    self.env.img[ymin:ymax+1, xmin] = 255
                    self.env.img[ymin:ymax+1, xmax] = 255
                    self.env.img[ymin, xmin:xmax+1] = 255
                    self.env.img[ymax, xmin:xmax+1] = 255
                else:
                    xmin = max(0, min(239, v['x'] - 4))
                    xmax = max(0, min(239, v['x'] + 4))
                    ymin = max(0, min(223, v['y'] - 4))
                    ymax = max(0, min(223, v['y'] + 4))
                    self.env.img[ymin:ymax+1, xmin:xmax+1] = [255, 0, 255]
        if render_mode == 'human':
            self.env.render()

        if self.stdout_debug:
            text = ''
            text += kb.display()
            text += f'\nterminated = {terminated}'
            text += f'\nepisode_reward = {self.episode_reward:0.4f}'
            text += f'\nepisode_reward_dict = {pprint.pformat(self.episode_reward_dict)}'
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
                # NOTE: normally we will not press any action to the environment;
                # but if link has died, we need to press the "continue" button;
                # to do this, we alternate between no action and pressing "start" every frame
                skipped_frames += 1
                skipaction = [False]*8
                if _gamestate_hearts(self.ram) <= 0 and skipped_frames%2 == 0:
                    skipaction[3] = True
                super().step(skipaction)
            self.env.render_mode = render_mode

        # return step results
        return observation, reward, terminated, truncated, info

    def generate_knowledge_base(self):
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
        ram = self.ram

        if self.mouse is not None:
            item = {}
            item['x'] = self.mouse['x']
            item['y'] = self.mouse['y']
            item['type'] = -5
            item['state'] = 0
            item['direction'] = 0
            item['health'] = 0
            kb['mouse'] = item

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

        # We add the tile information last.
        # the screen is divided into an 11x16 grid,
        # and each tile is divided into 2x2 subtiles;
        # therefore, the subtile grid is 22x32 (=0x2c0).
        # the leftmost, rightmost, and bottommost subtiles do not get displayed,
        # so only a 21x30 grid of subtiles is displayed
        subtiles = ram[0x530+0x800:0x530+0x2c0+0x800].reshape([32,22])

        # NOTE:
        # accessing the memory through get_state() allows changing the state;
        # the code below is usefull for modifying the grid in a running game;
        # it does not update the pictures (which are drawn at the beginning of the screen),
        # but does update how link/enemies behave on the tiles
        #import code
        #code.interact(local=locals())
        # >>> tiles = self.env.unwrapped.em.get_state()[14657:14657+11*4*16]
        # >>> newstate = b'\x00'*88; state = self.env.unwrapped.em.get_state(); state = state[:14657]+newstate+state[14657+len(newstate):]; self.env.unwrapped.em.set_state(state)

        include_background = True
        use_full_subtiles = False
        view_radius = 2
        ignore_tile_set = set([0x26, 0x24, 0x76])
        if include_background:
            if use_full_subtiles:
                tiles = subtiles
                tile_size = 8
            else:
                tiles = subtiles[::2,::2]
                tile_size = 16

            link_x = ram[112]
            link_y = ram[132]
            link_tile_x = (link_x+8)//tile_size
            link_tile_y = (link_y-56)//tile_size
            if not use_full_subtiles:
                link_tile_y = (link_y-48)//tile_size

            mapstr = ''
            for y in range(tiles.shape[1]):
                for x in range(tiles.shape[0]):
                    if abs(x - link_tile_x) + abs(y - link_tile_y) <= view_radius:
                        item = {}
                        item['x'] = x*tile_size 
                        item['y'] = y*tile_size + 61
                        item['type'] = -4
                        item['state'] = tiles[x, y]
                        item['direction'] = 0
                        item['health'] = 0
                        if tiles[x, y] not in ignore_tile_set:
                            kb[f'tile_{x:0>2d}_{y:0>2d}'] = item
                        prefix = '*'
                    else:
                        prefix = ' '
                    if (x, y) == (link_tile_x, link_tile_y):
                        mapstr += prefix + '# '
                    else:
                        if tiles[x, y] in ignore_tile_set:
                            mapstr += prefix + '  '
                        else:
                            mapstr += prefix + f'{hex(tiles[x, y])[2:]:2}'
                mapstr += '\n'
            #print(f"{mapstr}")

        # normalize and center all item positions
        kb.columns.add('relx')
        kb.columns.add('rely')
        for item, val in kb.items.items():
            # center on link
            if item != 'link':
                link_x = int(ram[112])
                link_y = int(ram[132])
            else:
                link_x = 0
                link_y = 0
            kb.items[item]['relx'] = kb.items[item]['x'] - link_x
            kb.items[item]['rely'] = kb.items[item]['y'] - link_y

            # normalize
            kb.items[item]['relx'] /= 240
            kb.items[item]['rely'] /= 160
            # NOTE:
            # the NES x resolution is 256,
            # but all sprites are 16 pixels wide,
            # so there are 240 available for positioning;
            # the y resolution is 224,
            # but the top vertical bar uses 56 pixels,
            # and only 8 pixels are reserved for "link positioning"
            # because he does not go "partially off the edge" on the top screen

        return kb


    ########################################
    # MARK: reward functions
    ########################################

    def get_rewards(self):
        ret = {}
        prefix = '_rewardfunc_'
        reward_functions = [f for f in dir(self) if f.startswith(prefix)]
        for reward_function in reward_functions:
            f = getattr(self, reward_function)
            if self.ram2 is not None:
                ret[reward_function[len(prefix):]] = f()
            else:
                ret[reward_function[len(prefix):]] = 0
        return ret

    def _rewardfunc_link_killed(self):
        return _gamestate_hearts(self.ram) == 0 and _gamestate_hearts(self.ram2) != 0

    def _rewardfunc_link_hit(self):
        return -1 * round(2*max(0, _gamestate_hearts(self.ram2) - _gamestate_hearts(self.ram)))

    def _rewardfunc_enemy_alldead(self):
        return (_gamestate_all_enemies_health(self.ram) == 0 and _gamestate_all_enemies_health(self.ram2) != 0) and not (_gamestate_is_screen_scrolling(self.ram) or _gamestate_is_cave_enter(self.ram))

    def _rewardfunc_screen_scrolling(self):
        return - int(_gamestate_is_screen_scrolling(self.ram) and not _gamestate_is_screen_scrolling(self.ram2))

    def _rewardfunc_screen_cave(self):
        return - int(_gamestate_is_cave_enter(self.ram) and not _gamestate_is_cave_enter(self.ram2))

    def _rewardfunc_enemy_hit(self):
        return max(0, _gamestate_all_enemies_health(self.ram2) - _gamestate_all_enemies_health(self.ram))

    def _rewardfunc_enemy_killed(self):
        return max(0, _gamestate_total_enemies(self.ram2) - _gamestate_total_enemies(self.ram))

    def _rewardfunc_link_l1dist(self):
        if self.mouse is None:
            return 0
        else:
            safe_radius = 0
            link_x = self.ram[112]
            link_y = self.ram[132]
            l1dist = abs(self.mouse['x'] - link_x) + abs(self.mouse['y'] - link_y)
            score = -max(0, l1dist-safe_radius) / 10000
            return score

    def _rewardfunc_button_push(self):
        return -min(1, abs(self.ram[0xfa] - self.ram2[0xfa])) / 1000

    def _rewardfunc_add_bomb(self):
        return max(0, self.ram[1624] - self.ram2[1624])

    def _rewardfunc_add_bombmax(self):
        return max(0, self.ram[1660] - self.ram2[1660])

    def _rewardfunc_add_keys(self):
        return max(0, self.ram[1646] - self.ram2[1646])

    def _rewardfunc_add_ruppees(self):
        return max(0, self.ram[1645] - self.ram2[1645])

    def _rewardfunc_add_clock(self):
        return max(0, self.ram[1644] - self.ram2[1644])

    def _rewardfunc_add_heart(self):
        return max(0, _gamestate_hearts(self.ram) - _gamestate_hearts(self.ram2))

########################################
# MARK: _gamestate
########################################

# The _gamestate functions modified from gym-zelda-1 repo;
# that repo uses the nesgym library, which is a pure python nes emulator
# and about 10x slower than the gymretro environment;
def _gamestate_screen_change(ram):
    return _gamestate_is_screen_scrolling(ram) or _gamestate_is_cave_enter(ram)

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

def _gamestate_link_killed(ram):
    return _gamestate_hearts(ram) <= 0

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

def _gamestate_total_enemies(ram):
    healths = []
    for i in range(6):
        rawhealth = ram[1158+i]
        if rawhealth > 0 and (rawhealth < 16 or rawhealth >= 128):
            healths.append(0)
        else:
            healths.append(rawhealth//16)
    return sum([1 for health in healths if health > 0])

def _gamestate_screen_change(ram):
    return ram[18] == 5
