import copy
import math
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
import retro
import gymnasium
from stable_baselines3.common.type_aliases import TensorDict
from gymnasium import spaces
from Mundus.Object import *


class RewardExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        **kwargs,
    ) -> None:
        # NOTE:
        # (this trick is copied from the SB3 MultiPolicy implementation.)
        # we do not know features_dim here before going over all the items,
        # so put something there temoporarily;
        # update features_dim later
        super().__init__(observation_space, features_dim=1)

        object_space = spaces.Dict({
            'objects_discrete': observation_space['objects_discrete'],
            'objects_continuous': observation_space['objects_continuous'],
            })
        extractors = {
            'objects': ObjectCnn(object_space, **kwargs),
            'rewards': nn.Flatten(),
            }
        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        from stable_baselines3.common.preprocessing import get_flattened_obs_dim
        dim_objects = 64 # FIXME
        dim_reward = get_flattened_obs_dim(observation_space['rewards'])
        self._features_dim = dim_objects + dim_reward

    def forward(self, observations: TensorDict) -> torch.Tensor:
        encoded_tensor_list = [
                self.extractors['objects']({
                    'objects_discrete': observations['objects_discrete'],
                    'objects_continuous': observations['objects_continuous'],
                    }),
                self.extractors['rewards'](observations['rewards']),
            ]
        return torch.cat(encoded_tensor_list, dim=1)


class RetroWithRam(gymnasium.Wrapper):
    '''
    The .get_ram() function is relatively expensive.
    The only purpose of this wrapper is to store the result of .get_ram() in an attribute
    to prevent multiple calls.
    '''
    def __init__(self, env):
        super().__init__(env)
        self.ram = None
        self.ram2 = None

    def step(self, action):
        class IntegerRam():
            def __init__(self, ram):
                self.ram = ram
            def __getitem__(self, key):
                if isinstance(key, slice):
                    return self.ram[key]
                elif isinstance(key, int):
                    return int(self.ram[key])
                else:
                    raise ValueError(f'IntegerRam; type={type(key)}, value={key} ')
        self.ram2 = self.ram
        self.ram = IntegerRam(self.env.get_ram())
        return super().step(action)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.ram2 = None
        self.ram = self.env.get_ram()
        return obs, info


# non-hex numbers are for underworld
ignore_tile_set = [0x26, 0x24, 0x76, 126, 104, 116]

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
            seed=None,
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
        self.scenario = scenario
        self.seed = seed
        self.random = random.Random(self.seed)

        # create a new observation space
        self.ram = self.env.get_ram()
        kb = self.generate_knowledge_base()
        kb_obs = kb.get_observation_space()
        self.observation_space = spaces.Dict({
            'objects_discrete': kb_obs['objects_discrete'],
            'objects_continuous': kb_obs['objects_continuous'],
            'rewards': spaces.Box(-1, 1, [len(self.get_rewards().keys())], np.float16),
            })
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
                'button_push': 1,
                },
            }

        self.scenarios['noitem'] = copy.deepcopy(self.scenarios['attack'])
        self.scenarios['noitem']['reward'] = {
            'add_bomb': -1,
            'add_clock': -1,
            'add_heart': -1,
            'add_keys': -1,
            'add_ruppees': -1,
            }

        self.scenarios['noattack'] = copy.deepcopy(self.scenarios['attack'])
        self.scenarios['noattack']['is_success'] = lambda ram: any([
            not _gamestate_link_killed(self.ram),
            ])
        self.scenarios['noattack']['reward'] |= {
            'enemy_hit': -1,
            'enemy_killed': -1,
            'enemy_alldead': -1,
            'add_bomb': -1,
            'add_clock': -1,
            'add_heart': -1,
            'add_keys': -1,
            'add_ruppees': -1,
            }

        self.scenarios['screen_cave'] = copy.deepcopy(self.scenarios['noattack'])
        self.scenarios['screen_cave']['is_success'] = lambda ram:any([
            self._rewardfunc_screen_cave(),
            ])
        self.scenarios['screen_cave']['reward'] |= {
            'screen_cave': -1,
            }

        self.scenarios['scroll'] = copy.deepcopy(self.scenarios['noattack'])
        self.scenarios['scroll']['is_success'] = lambda ram:any([
            self._rewardfunc_screen_scrolling(),
            ])
        self.scenarios['scroll']['reward'] |= {
            'screen_scrolling': -1,
            'screen_scrolling_north': -1,
            'screen_scrolling_south': -1,
            'screen_scrolling_east': -1,
            'screen_scrolling_west': -1,
            }

        self.scenarios['screen_scrolling_north'] = copy.deepcopy(self.scenarios['noattack'])
        self.scenarios['screen_scrolling_north']['is_success'] = lambda ram:any([
            self._rewardfunc_screen_scrolling_north(),
            ])
        self.scenarios['screen_scrolling_north']['reward'] |= {
            'screen_scrolling': 0,
            'screen_scrolling_north': -1,
            'screen_scrolling_south': 1,
            'screen_scrolling_east': 1,
            'screen_scrolling_west': 1,
            }

        self.scenarios['screen_scrolling_south'] = copy.deepcopy(self.scenarios['noattack'])
        self.scenarios['screen_scrolling_south']['is_success'] = lambda ram:any([
            self._rewardfunc_screen_scrolling_south(),
            ])
        self.scenarios['screen_scrolling_south']['reward'] |= {
            'screen_scrolling': 0,
            'screen_scrolling_north': 1,
            'screen_scrolling_south': -1,
            'screen_scrolling_east': 1,
            'screen_scrolling_west': 1,
            }

        self.scenarios['screen_scrolling_east'] = copy.deepcopy(self.scenarios['noattack'])
        self.scenarios['screen_scrolling_east']['is_success'] = lambda ram:any([
            self._rewardfunc_screen_scrolling_east(),
            ])
        self.scenarios['screen_scrolling_east']['reward'] |= {
            'screen_scrolling': 0,
            'screen_scrolling_north': 1,
            'screen_scrolling_south': 1,
            'screen_scrolling_east': -1,
            'screen_scrolling_west': 1,
            }

        self.scenarios['screen_scrolling_west'] = copy.deepcopy(self.scenarios['noattack'])
        self.scenarios['screen_scrolling_west']['is_success'] = lambda ram:any([
            self._rewardfunc_screen_scrolling_west(),
            ])
        self.scenarios['screen_scrolling_west']['reward'] |= {
            'screen_scrolling': 0,
            'screen_scrolling_north': 1,
            'screen_scrolling_south': 1,
            'screen_scrolling_east': 1,
            'screen_scrolling_west': -1,
            }

        self.scenarios['suicide'] = copy.deepcopy(self.scenarios['attack'])
        self.scenarios['suicide']['is_success'] = lambda ram: any([
            _gamestate_link_killed(self.ram),
            ])
        self.scenarios['suicide']['reward'] = {
            'link_killed': -1,
            'link_hit': -1,
            }

        self.scenarios['danger'] = copy.deepcopy(self.scenarios['attack'])
        self.scenarios['danger']['is_success'] = lambda ram: any([
            not _gamestate_link_killed(self.ram),
            ])
        self.scenarios['suicide']['reward'] |= {
            'link_killed': 4,
            'link_hit': -1,
            }

        self.scenarios['follow'] = copy.deepcopy(self.scenarios['attack'])
        self.scenarios['follow']['terminated'] = lambda ram: any([
            self._rewardfunc_link_onmouse(),
            _gamestate_link_killed(self.ram),
            _gamestate_is_screen_scrolling(self.ram),
            _gamestate_is_cave_enter(self.ram),
            ])
        self.scenarios['follow']['success'] = lambda ram: any([
            self._rewardfunc_link_onmouse(),
            ])
        self.scenarios['follow']['reward'] |= {
            'link_onmouse': 1,
            'link_l1dist' : 1,
            'enemy_hit': 0,
            'enemy_killed': 0,
            'enemy_alldead': 0,
            'add_bomb': 0,
            'add_clock': 0,
            'add_heart': 0,
            'add_keys': 0,
            'add_ruppees': 0,
            }

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
            #if self.stepcount%300 == 0:
            if random.random() < 1/300:
                if self.mouse is None or random.random() < 0.2:
                    mode = 'passable_tile'
                else:
                    mode = 'neighbor'
                subtiles = self.ram[0x530+0x800:0x530+0x2c0+0x800].reshape([32,22])
                tiles = subtiles[::2,::2]
                if mode == 'neighbor':
                    curx = round(self.mouse['x'] / 16)
                    cury = round((self.mouse['y'] - 56 - 8 + 3) / 16)
                    indexes = []
                    if curx > 2:
                        indexes.append([curx-1, cury])
                    if curx < 14:
                        indexes.append([curx+1, cury])
                    if cury > 2:
                        indexes.append([curx, cury-1])
                    if cury < 9:
                        indexes.append([curx, cury+1])
                    newindex = random.choice(indexes)
                    x = newindex[0] * 16 
                    y = newindex[1] * 16 + 56 + 8 - 3
                    self.mouse = {'x': x, 'y': y}
                elif mode == 'passable_tile':
                    indexes = np.argwhere(np.isin(tiles, ignore_tile_set))
                    if len(indexes) > 0:
                        newindex = random.choice(indexes)
                        x = newindex[0] * 16 
                        y = newindex[1] * 16 + 56 + 8 - 3
                        self.mouse = {'x': x, 'y': y}
                elif mode == 'randomxy':
                    x = random.randrange(0, 255) - 8
                    y = random.randrange(60, 240) - 8
                    self.mouse = {'x': x, 'y': y}
                else:
                    raise ValueError('should not happen')
        self.scenarios['follow_random'] = copy.deepcopy(self.scenarios['follow'])
        #self.scenarios['follow_random']['terminated'] = lambda ram: any([
                #_gamestate_link_killed(self.ram),
                #_gamestate_is_screen_scrolling(self.ram),
                #_gamestate_is_cave_enter(self.ram),
                #])
        #self.scenarios['follow_random']['is_success'] = lambda ram: not any([
                #_gamestate_link_killed(self.ram),
                #_gamestate_is_screen_scrolling(self.ram),
                #_gamestate_is_cave_enter(self.ram),
                #])
        self.scenarios['follow_random']['step'] = _step_follow_random

        self.scenarios['repel_random'] = copy.deepcopy(self.scenarios['follow_random'])
        self.scenarios['repel_random']['reward']['link_l1dist'] = -1

    def reset(self, **kwargs):
        self.episode_summary_dict = {}
        self.episode_reward_dict = {}
        self.episode_reward = 0
        if self.scenario is not None:
            self.episode_scenario = self.scenario
        else:
            self.episode_scenario = random.choice(sorted(self.scenarios.keys()))
        self.stepcount = 0
        obs, info = super().reset(**kwargs)
        # FIXME: randomly initializing link's position should be behind a flag
        self._random_link_position()
        return self.observation_space.sample(), info

    def step(self, action):
        self.stepcount += 1

        # step the emulator
        render_mode = self.env.render_mode
        self.env.render_mode = 'rgb_array'
        observation, reward, terminated, truncated, info = super().step(action)
        self.env.render_mode = render_mode

        # compute the scenario information
        kb = self.generate_knowledge_base()
        kb_obs = kb.to_observation()

        scenario = self.scenarios[self.episode_scenario]
        terminated = scenario['terminated'](self.ram)
        info['is_success'] = scenario['is_success'](self.ram)
        if 'step' in scenario:
            scenario['step'](self, kb)

        summary_dict = self.get_rewards()
        reward_dict = {}
        reward = 0
        for k, v in summary_dict.items():
            reward_k = summary_dict[k] * scenario['reward'].get(k, 1)
            reward += reward_k
            reward_dict[k] = reward_k

        self.episode_reward += reward
        self.episode_summary_dict = {k: self.episode_summary_dict.get(k, 0) + summary_dict.get(k, 0) for k in set(self.episode_summary_dict) | set(summary_dict)}
        self.episode_reward_dict = {k: self.episode_reward_dict.get(k, 0) + reward_dict.get(k, 0) for k in set(self.episode_reward_dict) | set(reward_dict)}
        for k in summary_dict.keys():
            info['summary_' + k] = self.episode_summary_dict[k]
            info['reward_' + k] = self.episode_reward_dict[k]

        info['scenario_success_' + self.episode_scenario] = scenario['is_success'](self.ram)
        info['scenario_count_' + self.episode_scenario] = 1

        observation = {
            'objects_discrete': kb_obs['objects_discrete'],
            'objects_continuous': kb_obs['objects_continuous'],
            'rewards': [v for k, v in sorted(summary_dict.items())]
            }

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

    def _set_mem(self, assignment_dict):
        '''
        Change the values of memory.
        The input `assignment_dict` has keys as memory addresses and values as integers between 0-255.
        '''
        offset = 93
        i0 = 0
        chunks = []
        state = self.env.em.get_state()
        for k, v in sorted(assignment_dict.items()):
            assert v >= 0 and v <= 255
            i = k + offset
            chunks.append(state[i0:i] + bytes([v]))
            i0 = i + 1
        chunks.append(state[i0:])
        state1 = b''.join(chunks)
        self.env.em.set_state(state1)

        # NOTE:
        # if the state update is malformed for some reason,
        # then the state won't update;
        # this simple assert checks that the state update actually happened;
        # it might be too slow for an inner loop or interfere with a long training run;
        # so it should probably be put behind a flag
        if True:
            state = self.env.em.get_state()
            for k, v in assignment_dict.items():
                assert state[k+offset] == v

    def generate_knowledge_base(self, include_background=False):
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
        kb = KnowledgeBase(keys={
            'objects_discrete': ['type', 'direction', 'state'],
            'objects_continuous': ['relx', 'rely', 'x', 'y', 'health'],
            })
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

        # add tile information last
        # FIXME:
        # this code isn't currently being run, and not sure how buggy it is
        subtiles = self._get_subtiles()
        use_full_subtiles = False
        view_radius = 2
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
                    if abs(int(x) - int(link_tile_x)) + abs(int(y) - int(link_tile_y)) <= view_radius:
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

    def _get_subtiles(self):
        # The screen is divided into an 11x16 grid,
        # and each tile is divided into 2x2 subtiles;
        # therefore, the subtile grid is 22x32 (=0x2c0).
        # the leftmost, rightmost, and bottommost subtiles do not get displayed,
        # so only a 21x30 grid of subtiles is displayed
        subtiles = self.ram[0x530+0x800:0x530+0x2c0+0x800].reshape([32,22])
        # NOTE:
        # accessing the memory through get_state() allows changing the state;
        # the code below is usefull for modifying the grid in a running game;
        # it does not update the pictures (which are drawn at the beginning of the screen),
        # but does update how link/enemies behave on the tiles
        #import code
        #code.interact(local=locals())
        # >>> tiles = self.env.unwrapped.em.get_state()[14657:14657+11*4*16]
        # >>> newstate = b'\x00'*88; state = self.env.unwrapped.em.get_state(); state = state[:14657]+newstate+state[14657+len(newstate):]; self.env.unwrapped.em.set_state(state)

        return subtiles
    
    def _random_link_position(self):
        subtiles = self._get_subtiles()
        valid_positions = []
        for x in range(32):
            for y in range(22):
                if subtiles[x, y] in ignore_tile_set:
                    valid_positions.append([x*8, y*8+60])
        position = self.random.choice(valid_positions)
        self._set_mem({112: position[0], 132: position[1]})


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
        return - int(_gamestate_hearts(self.ram) == 0 and _gamestate_hearts(self.ram2) != 0)

    def _rewardfunc_link_hit(self):
        return -1 * round(2*max(0, _gamestate_hearts(self.ram2) - _gamestate_hearts(self.ram)))

    def _rewardfunc_enemy_alldead(self):
        return (_gamestate_all_enemies_health(self.ram) == 0 and _gamestate_all_enemies_health(self.ram2) != 0) and not (_gamestate_is_screen_scrolling(self.ram) or _gamestate_is_cave_enter(self.ram))

    def _rewardfunc_screen_scrolling(self):
        return - int(_gamestate_is_screen_scrolling(self.ram) and not _gamestate_is_screen_scrolling(self.ram2))

    def _rewardfunc_screen_scrolling_north(self):
        return int(self.ram[132] == 61) * self._rewardfunc_screen_scrolling()

    def _rewardfunc_screen_scrolling_south(self):
        return int(self.ram[132] == 221) * self._rewardfunc_screen_scrolling()

    def _rewardfunc_screen_scrolling_west(self):
        return int(self.ram[112] == 0) * self._rewardfunc_screen_scrolling()

    def _rewardfunc_screen_scrolling_east(self):
        return int(self.ram[112] == 240) * self._rewardfunc_screen_scrolling()

    def _rewardfunc_screen_cave(self):
        return - int(_gamestate_is_cave_enter(self.ram) and not _gamestate_is_cave_enter(self.ram2))

    def _rewardfunc_enemy_hit(self):
        return max(0, _gamestate_all_enemies_health(self.ram2) - _gamestate_all_enemies_health(self.ram))

    def _rewardfunc_enemy_killed(self):
        return max(0, _gamestate_total_enemies(self.ram2) - _gamestate_total_enemies(self.ram))

    def _rewardfunc_link_onmouse(self):
        link_x = self.ram[112]
        link_y = self.ram[132]
        if self.mouse is None:
            return False
        else:
            return self.mouse['x'] == link_x and self.mouse['y'] == link_y

    def _rewardfunc_link_l1dist(self):
        if self.mouse is None:
            return 0
        else:
            safe_radius = 8
            link_x = self.ram[112]
            link_y = self.ram[132]
            l1dist = abs(self.mouse['x'] - link_x) + abs(self.mouse['y'] - link_y)
            score = -max(0, l1dist-safe_radius) / 1000 / 60
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


################################################################################
# FIXME:
# This class was taken from another open source project;
# it reduces the action space to speed up training;
# I need to incorporate this trick and lookup the cite
class ZeldaActionSpace(gymnasium.ActionWrapper):
    """A wrapper that shrinks the action space down to what's actually used in the game."""
    def __init__(self, env, kind):
        super().__init__(env)

        # movement is always allowed
        self.actions = [['LEFT'], ['RIGHT'], ['UP'], ['DOWN'],
                        ['LEFT', 'A'], ['RIGHT', 'A'], ['UP', 'A'], ['DOWN', 'A'],
                        ['LEFT', 'B'], ['RIGHT', 'B'], ['UP', 'B'], ['DOWN', 'B'],
                        ['UP', 'LEFT', 'B'], ['UP', 'RIGHT', 'B'], ['DOWN', 'LEFT', 'B'], ['DOWN', 'RIGHT', 'B']
                        ]

        num_action_space = 4
        if kind != 'move-only':
            num_action_space += 4

        if kind in ('directional-item', 'diagonal-item', 'all'):
            num_action_space += 4

        if kind in ('diagonal-item', 'all'):
            num_action_space += 4

        assert isinstance(env.action_space, gym.spaces.MultiBinary)

        self.button_count = env.action_space.n
        self.buttons = env.unwrapped.buttons

        self._decode_discrete_action = []
        for action in self.actions:
            arr = np.array([False] * self.button_count)

            for button in action:
                arr[self.buttons.index(button)] = True

            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(num_action_space)

    def action(self, action):
        if isinstance(action, list) and len(action) and isinstance(action[0], str):
            arr = np.array([False] * self.button_count)
            for button in action:
                arr[self.buttons.index(button)] = True

            return arr

        return self._decode_discrete_action[action].copy()
