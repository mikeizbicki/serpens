import copy
import glob
import logging
import math
import numpy as np
import os
import pprint
import pyglet
import random
import re
import time

import torch
import retro
import gymnasium
from gymnasium import spaces
from Mundus.Object import *
from Mundus.util import *

# load font for writing to image
from PIL import Image, ImageDraw, ImageFont
pil_font = ImageFont.load_default()


def make_zelda_env(
        action_space='all',
        render_mode='human',
        **kwargs):
    '''
    Create a Zelda environment.

    NOTE:
    I've arranged everything into a single function here
    to avoid inconsistencies between the train/play code that were happening.
    This style doesn't follow the standard for stable-retro,
    and could probably be improved.
    '''

    # set the folder where retro.make() looks for ROMs
    custom_path = os.path.join(os.getcwd(), 'custom_integrations')
    retro.data.Integrations.add_custom_path(custom_path)

    if action_space == 'DISCRETE':
        use_restricted_actions = retro.Actions.DISCRETE
    elif action_space == 'MULTI_DISCRETE':
        use_restricted_actions = retro.Actions.MULTI_DISCRETE
    elif action_space == 'MULTI_DISCRETE':
        use_restricted_actions = retro.Actions.MULTI_DISCRETE
    else:
        use_restricted_actions = retro.Actions.ALL

    # create the environment
    env = retro.make(
            game='Zelda-Nes',
            inttype=retro.data.Integrations.ALL,
            state='overworld_07',
            render_mode=render_mode,
            use_restricted_actions=use_restricted_actions,
            )
    env = ZeldaWrapper(env, **kwargs)

    # apply zelda-specific action space
    if 'zelda-' in action_space:
        kind = action_space.split('-')[1]
        env = ZeldaActionSpace(env, kind)

    logging.debug(f"env.action_space={env.action_space}")
    logging.debug(f"env.action_space.n={getattr(env.action_space, 'n', None)}")
    return env


class RetroWithRam(gymnasium.Wrapper):
    '''
    The .get_ram() function is relatively expensive.
    The only purpose of this wrapper is to store the result of .get_ram() in an attribute
    to prevent multiple calls.
    '''

    class _IntLookup():
        '''
        By default, self.ram is a np.array with dtype=uint8.
        This can result in overflow/underflows which are obnoxious to debug.
        This class converts all results to integers before returning to avoid these bugs.
        '''
        def __init__(self, ram):
            self.ram = ram
            self.mouse = None

        def __getitem__(self, key):
            if isinstance(key, slice):
                return self.ram[key]
            elif isinstance(key, int):
                return int(self.ram[key])
            else:
                raise ValueError(f'IntegerLookup; type={type(key)}, value={key} ')


    def __init__(self, env):
        super().__init__(env)
        self.ram2 = None
        self.ram = self._IntLookup(self.env.get_ram())

    def _update_ram(self):
        self.ram2 = self.ram
        self.ram = self._IntLookup(self.env.get_ram())

    def step(self, action):
        self._update_ram()
        return super().step(action)

    def reset(self, **kwargs):
        self.ram = self.env.get_ram()
        obs, info = super().reset(**kwargs)
        self.ram2 = None
        self.ram = self.env.get_ram()
        return obs, info


class ZeldaWrapper(RetroWithRam):
    '''
    Wrapper for extracting useful info from the RAM of original NES Zelda.
    '''

    ########################################
    # MARK: tasks
    ########################################
    tasks = {}
    tasks['attack'] = {
        'terminated': [
            '_ramstate_all_enemies_dead',
            '_ramstate_link_killed',
            '_ramstate_is_screen_scrolling',
            '_ramstate_is_cave_enter',
            ],
        'is_success': [
            '_ramstate_all_enemies_dead',
            ],
        'reward': {
            'link_killed': -2,
            'link_hit': -1,
            'enemy_alldead': 2,
            'screen_scrolling': -2,
            'screen_cave': -2,
            'enemy_hit': 1,
            'enemy_killed': 1,
            'add_bomb': 1,
            'add_keys': 1,
            'add_ruppees': 1,
            'add_clock': 1,
            'add_heart': 1,
            },
        'pseudoreward': {
            'button_push': -0.001,
            'button_arrow_nomove': -0.01,
            }
        }
    tasks['attack_noitem'] = copy.deepcopy(tasks['attack'])
    tasks['attack_noitem']['reward'] |= {
        'add_bomb': -1,
        'add_clock': -1,
        'add_heart': -1,
        'add_keys': -1,
        'add_ruppees': -1,
        }
    tasks['noattack'] = copy.deepcopy(tasks['attack'])
    tasks['noattack'] |= {
        'is_success': [
            '_ramstate_link_alive',
            ],
        'reward': {
            'link_killed': -2,
            'link_hit': -1,
            'enemy_alldead': -2,
            'screen_scrolling': -2,
            'screen_cave': -2,
            'enemy_hit': -1,
            'enemy_killed': -1,
            },
        }

    ####################
    # mouse tasks
    ####################

    def _step_onmouse_random(self, kb):
        if self.ram.mouse is None or self.random.random() < 1/300:
            if self.ram.mouse is None or self.random.random() < 0.2:
                mode = 'passable_tile'
            else:
                mode = 'neighbor'
            subtiles = self.ram[0x530+0x800:0x530+0x2c0+0x800].reshape([32,22])
            tiles = subtiles[::2,::2]
            if mode == 'neighbor':
                curx = round(self.ram.mouse['x'] / 16)
                cury = round((self.ram.mouse['y'] - 56 - 8 + 3) / 16)
                indexes = []
                if curx > 2:
                    indexes.append([curx-1, cury])
                if curx < 14:
                    indexes.append([curx+1, cury])
                if cury > 2:
                    indexes.append([curx, cury-1])
                if cury < 9:
                    indexes.append([curx, cury+1])
                newindex = self.random.choice(indexes)
                x = newindex[0] * 16 
                y = newindex[1] * 16 + 56 + 8 - 3
                self.ram.mouse = {'x': x, 'y': y}
            elif mode == 'passable_tile':
                indexes = np.argwhere(np.isin(tiles, ignore_tile_set))
                if len(indexes) > 0:
                    newindex = self.random.choice(indexes)
                    x = newindex[0] * 16 
                    y = newindex[1] * 16 + 56 + 8 - 3
                    self.ram.mouse = {'x': x, 'y': y}
            elif mode == 'randomxy':
                x = self.random.randrange(0, 255) - 8
                y = self.random.randrange(60, 240) - 8
                self.ram.mouse = {'x': x, 'y': y}
            else:
                raise ValueError('should not happen')
            self.mouse = self.ram.mouse
    tasks['onmouse_random'] = {
        'terminated': [
            '_ramstate_link_onmouse',
            '_ramstate_link_killed',
            '_ramstate_is_screen_scrolling',
            '_ramstate_is_cave_enter',
            ],
        'is_success': [
            '_ramstate_link_onmouse',
            ],
        'reward': {
            'link_killed': -2,
            'link_hit': -1,
            'link_onmouse': 2,
            },
        'pseudoreward': {
            'link_l1dist_decrease' : 0.01,
            'link_l1dist_increase' : -0.01,
            },
        'step': _step_onmouse_random
        }

    def _step_onmouse_enemy(self, kb):
        enemies = []
        for k in kb.items:
            if 'enemy' in k:
                enemies.append(kb.items[k])
        if len(enemies) == 0:
            mouse = None
        else:
            link = kb.items['link']
            mouse = {
                'x': enemies[0]['x'],
                'y': enemies[0]['y'],
                }
            self.mouse = mouse
            self.ram.mouse = mouse
    tasks['onmouse_enemy'] = copy.deepcopy(tasks['onmouse_random'])
    tasks['onmouse_enemy']['reward'] = {
        'link_killed': -2,
        'link_hit': -1,
        'link_onmouse': 2,
        'mouse_enemy_hit': 1,
        'nomouse_enemy_hit': -0.1,
        'mouse_enemy_killed': 1,
        'screen_scrolling': -2,
        'screen_cave': -2,
        }
    tasks['onmouse_enemy']['step'] = _step_onmouse_enemy

    ####################
    # screen change tasks
    ####################

    def _step_screen(self, kb, direction):
        if direction == 'NORTH':
            mouse = {'x': 120, 'y': 40}
        if direction == 'SOUTH':
            mouse = {'x': 120, 'y': 240}
        if direction == 'WEST':
            mouse = {'x': -20, 'y': 140}
        if direction == 'EAST':
            mouse = {'x': 260, 'y': 140}
        self.mouse = mouse
        self.ram.mouse = mouse

    tasks['screen_north'] = copy.deepcopy(tasks['noattack'])
    tasks['screen_north']['is_success'] = [
        '_ramstate_screen_scrolling_north'
        ]
    tasks['screen_north']['reward'] |= {
        'screen_scrolling_north': 4,
        }
    tasks['screen_north']['pseudoreward'] = {
        'link_l1dist_decrease' : 0.01,
        'link_l1dist_increase' : -0.01,
        }
    tasks['screen_north']['is_valid'] = lambda ram: _isscreen_changeable(ram, 'NORTH')
    tasks['screen_north']['step'] = lambda self, kb: self._step_screen(kb, 'NORTH')

    tasks['screen_south'] = copy.deepcopy(tasks['noattack'])
    tasks['screen_south']['is_success'] = [
        '_ramstate_screen_scrolling_south'
        ]
    tasks['screen_south']['reward'] |= {
        'screen_scrolling_south': 4,
        }
    tasks['screen_south']['pseudoreward'] = {
        'link_l1dist_decrease' : 0.01,
        'link_l1dist_increase' : -0.01,
        }
    tasks['screen_south']['is_valid'] = lambda ram: _isscreen_changeable(ram, 'SOUTH')
    tasks['screen_south']['step'] = lambda self, kb: self._step_screen(kb, 'SOUTH')

    tasks['screen_east'] = copy.deepcopy(tasks['noattack'])
    tasks['screen_east']['is_success'] = [
        '_ramstate_screen_scrolling_east'
        ]
    tasks['screen_east']['reward'] |= {
        'screen_scrolling_east': 4,
        }
    tasks['screen_east']['pseudoreward'] = {
        'link_l1dist_decrease' : 0.01,
        'link_l1dist_increase' : -0.01,
        }
    tasks['screen_east']['is_valid'] = lambda ram: _isscreen_changeable(ram, 'EAST')
    tasks['screen_east']['step'] = lambda self, kb: self._step_screen(kb, 'EAST')

    tasks['screen_west'] = copy.deepcopy(tasks['noattack'])
    tasks['screen_west']['is_success'] = [
        '_ramstate_screen_scrolling_west'
        ]
    tasks['screen_west']['reward'] |= {
        'screen_scrolling_west': 4,
        }
    tasks['screen_west']['pseudoreward'] = {
        'link_l1dist_decrease' : 0.01,
        'link_l1dist_increase' : -0.01,
        }
    tasks['screen_west']['is_valid'] = lambda ram: _isscreen_changeable(ram, 'WEST')
    tasks['screen_west']['step'] = lambda self, kb: self._step_screen(kb, 'WEST')


    def __init__(
            self,
            env,
            stdout_debug=False,
            debug_assert=False,
            use_subtiles=False,
            no_render_skipped_frames=True,
            skip_boring_frames=True,
            render_kb=True,
            task='attack',
            seed=None,
            reset_method='link enemy map',
            frames_without_attack_threshold=None,
            fast_termination=False,
            ):

        # bookkeeping
        super().__init__(env)
        self.stdout_debug = stdout_debug
        self.debug_assert = debug_assert
        self.use_subtiles = use_subtiles
        self.no_render_skipped_frames=no_render_skipped_frames
        self.skip_boring_frames = skip_boring_frames 
        self.render_kb = render_kb
        self.seed = seed
        self.random = None
        self.random_reset = random.Random(self.seed)
        self.mouse = None
        self.reset_method = reset_method
        self.fast_termination = fast_termination
        if frames_without_attack_threshold is None:
            self.frames_without_attack_threshold = math.inf
        else:
            self.frames_without_attack_threshold = frames_without_attack_threshold

        # self.valid_tasks contains all the tasks that match the task glob
        self.task = task
        pattern = re.compile(self.task)
        self.valid_tasks = sorted([task for task in self.tasks.keys() if pattern.match(task)])

        # create a new observation space
        kb = generate_knowledge_base(self.ram, self.ram2)
        self.observation_space = kb.get_observation_space()
        self.observation_space['rewards'] = self.observation_space['events']
        logging.info(f'self.observation_space.shape={self.observation_space.shape}')
        logging.info(f"self.observation_space.keys()={self.observation_space.keys()}")
        for k in self.observation_space:
            logging.info(f"self.observation_space[k].shape={self.observation_space[k].shape}")

    def reset(self, **kwargs):
        self.random = random.Random(self.random_reset.randint(0, 2**30))
        self.episode_event_multiset = MultiSet()
        self.episode_reward_multiset = MultiSet()
        self.episode_pseudoreward_multiset = MultiSet()
        self.episode_reward = 0
        self.episode_pseudoreward = 0
        self.episode_task = self.random.choice(self.valid_tasks)
        self.stepcount = 0
        self.frames_without_attack = 0
        self.max_frames_without_attack = 0
        self.skipped_frames = 0
        self.mouse = None
        obs, info = super().reset(**kwargs)

        # reset map/link/enemy location
        valid_coords = []
        if 'map' in self.reset_method:
            valid_map_coords = [
                0x1e, 0x1f, 0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x38, 0x3a, 0x3b, 0x3d, 0x3e, 0x3f, 0x48, 0x49, 0x4a, 0x4c, 0x4d, 0x4e, 0x4f, 0x51, 0x52, 0x53, 0x5d, 0x58, 0x59, 0x5a, 0x5b, 0x5b, 0x5e, 0x5f, 0x61, 0x63, 0x63, 0x64, 0x65, 0x66, 0x67, 0x69, 0x6a, 0x6b, 0x6f, 0x70, 0x71, 0x73, 0x73, 0x76, 0x78, 0x79, 0x7a, 0x7b, 0x7c, 0x7d
                ]
            hard_coords = [0x10, 0x11, 0x20, 0x12, 0x13, 0x14, 0x15, 0x25, 0x26, 0x70, 0x50, 0x60]
            valid_coords += valid_map_coords
        if 'spider' in self.reset_method:
            valid_coords += [0x76, 0x79, 0x7a, 0x4a, 0x2c]
        if 'octo' in self.reset_method:
            valid_coords += [0x68, 0x78, 0x58, 0x57, 0x67, 0x66, 0x49, 0x64]
        if valid_coords != []:
            new_coord = self.random.choice(valid_coords)
            self._set_map_coordinates_eb(new_coord)

        isvalid = self.tasks[self.episode_task].get('is_valid', lambda ram:True)(self.ram)
        if not isvalid:
            self.episode_task = 'attack'

        if 'link' in self.reset_method:
            self._set_random_link_position()
        if 'enemy' in self.reset_method:
            self._set_random_enemy_positions()
        return self.observation_space.sample(), info

    def step(self, action):
        self.stepcount += 1

        # step the emulator
        render_mode = self.env.render_mode
        self.env.render_mode = 'rgb_array'
        observation, reward, terminated, truncated, info = super().step(action)
        self.env.render_mode = render_mode

        # compute the task information
        self.ram.mouse = self.mouse # FIXME: this should be in only one spot
        kb = generate_knowledge_base(self.ram, self.ram2)
        kb_obs = kb.to_observation()
        task = self.tasks[self.episode_task]

        kb_obs['rewards'] = np.array([task['reward'].get(k, 0) for k in sorted(kb.events)])

        terminated = any([globals()[fname](self.ram) for fname in task['terminated']])
        info['is_success'] = any([globals()[fname](self.ram) for fname in task['is_success']])
        if 'step' in task:
            task['step'](self, kb)

        # potentially end task early if we're going too long
        if _ramstate_all_enemies_health(self.ram) == _ramstate_all_enemies_health(self.ram2) and _ramstate_hearts(self.ram) == _ramstate_hearts(self.ram2):
            self.frames_without_attack += 1
        else:
            self.frames_without_attack = 0
        self.max_frames_without_attack = max(self.max_frames_without_attack, self.frames_without_attack)

        info['misc_max_frames_without_attack'] = self.max_frames_without_attack
        info['misc_truncated_noattack'] = 0
        if self.frames_without_attack >= self.frames_without_attack_threshold:
            truncated = True
            info['misc_truncated_noattack'] = 1

        # compute logging information
        event_multiset = MultiSet(kb.events)
        reward_multiset = MultiSet({k: v * event_multiset[k] for k, v in task['reward'].items()})
        pseudoreward_multiset = MultiSet({k: v * event_multiset[k] for k, v in task['pseudoreward'].items()})
        reward = sum(reward_multiset.values())
        pseudoreward = sum(pseudoreward_multiset.values())
        self.episode_reward += reward
        self.episode_pseudoreward += pseudoreward
        self.episode_event_multiset += event_multiset
        self.episode_reward_multiset += reward_multiset
        for k in event_multiset.keys():
            info['event_' + k] = self.episode_event_multiset[k]
            info['reward_' + k] = self.episode_reward_multiset[k]
            info['reward_' + k] = self.episode_pseudoreward_multiset[k]

        info['task_success_' + self.episode_task] = info['is_success']
        info['task_count_' + self.episode_task] = 1

        # render the environment
        if self.render_kb:
            
            # draw bounding box around objects
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

            # write the task to the screen
            img_pil = Image.fromarray(self.env.img)
            draw = ImageDraw.Draw(img_pil)
            draw.text(
                [0,0],
                self.episode_task,
                font=pil_font,
                fill=(255, 0, 255),
                antialiasing=False,
                )
            self.env.img = np.array(img_pil)

        if render_mode == 'human':
            self.env.render()

        if self.stdout_debug:
            text = ''
            text += kb.display()
            #text += kb.info['mapstr']
            text += f'\n{format_dict_pretty(self.episode_event_multiset)}'
            text += f'\n'
            text += f'\nepisode_reward = {self.episode_reward:0.4f}'
            text += f'\nterminated = {terminated}'
            text += f'\nmap_coord = {hex(self.ram[0xEB])}'
            print(text)

        # skip the boring frames
        if not (self.fast_termination and (terminated or truncated)) and self.skip_boring_frames:
            while any([
                _ramstate_is_screen_scrolling(self.ram),
                _ramstate_is_drawing_text(self.ram),
                _ramstate_is_cave_enter(self.ram),
                _ramstate_is_inventory_scroll(self.ram),
                _ramstate_hearts(self.ram) <= 0,
                _ramstate_is_openning_scene(self.ram),
                ]):
                self.skipped_frames += 1
                # NOTE:
                # but if link has died, we need to press the "continue" button;
                # to do this, we alternate between no action and pressing "start" every frame
                # if link has not died, we will not press any action to the environment;
                # in order to make our button presses compatible with any action_space,
                # we need to manually set the buttons and step the interpreter;
                # the code below is very similar to the code in retro.RetroEnv.step()

                skipbuttons = []
                if _ramstate_hearts(self.ram) <= 0 and self.skipped_frames%2 == 0:
                    skipbuttons = ['START']
                self._set_buttons(skipbuttons)

                self.env.em.step()
                self.env.data.update_ram()
                self._update_ram()
                ob = self.env._update_obs()
                if self.env.render_mode == "human" and not self.no_render_skipped_frames:
                    self.env.render()

        info['misc_step'] = self.stepcount
        info['misc_skipped_frames'] = self.skipped_frames
        info['misc_episode_reward'] = self.episode_reward
        info['misc_episode_pseudoreward'] = self.episode_pseudoreward

        # return step results
        return kb_obs, reward + pseudoreward, terminated, truncated, info

    ########################################
    # MARK: debug code
    ########################################

    def save_state(statename):
        '''
        Create a file that stores the emulator's state.
        The file is gzip compressed so that it is compatible with being loaded directly in gym-retro.
        '''
        filename = f'custom_integrations/Zelda-Nes/{statename}.state'
        import gzip
        with gzip.open(filename, 'wb') as f:
            f.write(self.unwrapped.em.get_state())

    def load_state(statename):
        '''
        Load a state file.
        This is useful for debuging purposes to load in the middle of a game run.
        '''
        filename = f'custom_integrations/Zelda-Nes/{statename}.state'
        import gzip
        with gzip.open(filename, 'rb') as f:
            self.unwrapped.em.set_state(f.read())

    ########################################
    # MARK: change game state
    ########################################

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

        if self.debug_assert:
            state = self.env.em.get_state()
            for k, v in assignment_dict.items():
                assert state[k+offset] == v

    def _set_buttons(self, buttons):
        mask = [1 if button in buttons else 0 for button in self.unwrapped.buttons]
        masp = np.array(mask, dtype=np.uint8)
        self.unwrapped.em.set_button_mask(mask, 0)

    def _set_map_coordinates_eb(self, eb):
        '''
        The RAM position 0xEB stores the screen coordinates.
        (Y-axis is the high nibble with max of 7, X-axis is the low nibble.)
        Setting 0xEB directly causes strange behavior.
        So our strategy is to:
        compute how many screens left/right and up/down we need to move;
        then perform the screen transitions with the _move_map() function.
        '''
        # NOTE:
        # there seems to be a bug in the _move_map function;
        # sometimes it does not correctly move the map the first time it is called;
        # we work around this bug by first calling it in all the directions,
        # then calculating our current position and the location we need to move to;
        # this seems to work, but adds overhead
        self._move_map('LEFT')
        self._move_map('RIGHT')
        self._move_map('DOWN')
        self._move_map('UP')

        x0 = eb % 0x10
        x1 = int(self.ram[0xEB]) % 0x10
        for i in range(x1 - x0):
            self._move_map('LEFT')
        for i in range(x0 - x1):
            self._move_map('RIGHT')

        y0 = eb // 0x10
        y1 = int(self.ram[0xEB]) // 0x10
        for i in range(y0 - y1):
            self._move_map('DOWN')
        for i in range(y1 - y0):
            self._move_map('UP')

        if self.debug_assert:
            assert eb == self.ram[0xEB], f'eb={hex(eb)}, self.ram[0xEB]={hex(self.ram[0xEB])}'

    def _move_map(self, direction):
        '''
        Scroll the map in the specified direction.
        This function first teleports link to the edge of the screen,
        and then presses the corresponding button to trigger the screen transition.
        '''

        # move link to edge of screen
        if direction == 'LEFT':
            self._set_mem({112: 0, 132: 140})
        elif direction == 'RIGHT':
            self._set_mem({112: 240, 132: 140})
        elif direction == 'UP':
            self._set_mem({112: 120, 132: 61})
        elif direction == 'DOWN':
            self._set_mem({112: 120, 132: 221})
        else:
            raise ValueError(f'direction={direction}')

        # walk through edge
        for i in range(10):
            self._set_buttons([direction])
            self.env.em.step()
            self.env.data.update_ram()
            self._update_ram()

        # skip until frame transition done
        while _ramstate_is_screen_scrolling(self.ram):
            self.env.em.step()
            self.env.data.update_ram()
            self._update_ram()

        # after switching screens, link might be inside a wall;
        # so we move him to a random position
        self._set_random_link_position()

    def _set_random_link_position(self, choice=None):
        subtiles = _get_subtiles(self.ram)
        valid_positions = []
        for x in range(32):
            for y in range(1, 22):
                if subtiles[x, y] in ignore_tile_set:
                    valid_positions.append([x*8, y*8+60-8])
        if choice is None:
            position = self.random.choice(valid_positions)
        else:
            position = valid_positions[choice]
        self._set_mem({112: position[0], 132: position[1]})

    def _set_random_enemy_positions(self):
        subtiles = _get_subtiles(self.ram)
        valid_positions = []
        for x in range(32):
            for y in range(1, 22):
                if subtiles[x, y] in ignore_tile_set:
                    valid_positions.append([x*8, y*8+60-8])
        positions = self.random.sample(valid_positions, 6)

        update_dict = {}
        for i in range(6):
            update_dict.update(
                    { 113+i: positions[0+i][0]              # x
                    , 133+i: positions[0+i][1]              # y
                    # FIXME:
                    # the following sets the direction, but fails sometimes for spiders?
                    #, 153+i: self.random.choice([1,2,4,8])  # dir
                    # FIXME:
                    # the following lines "should" set the state and type of monsters;
                    # but there's a problem with rendering them;
                    # not sure if the monsters actually behave correctly or not.
                    #, 848+i: i+6
                    })

        self._set_mem(update_dict)

########################################
# MARK: knowledge base
########################################

def generate_knowledge_base(ram, ram2, include_background=True, use_subtiles=False):
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
        'objects_continuous': ['relx', 'rely', 'x', 'y', 'dx', 'dy', 'health'],
        })

    if ram.mouse is not None:
        item = {}
        item['x'] = ram.mouse['x']
        item['y'] = ram.mouse['y']
        item['dx'] = ram.mouse['x'] - ram2.mouse['x'] if ram2 is not None and ram2.mouse is not None else 0
        item['dy'] = ram.mouse['y'] - ram2.mouse['y'] if ram2 is not None and ram2.mouse is not None else 0
        item['type'] = -5
        item['state'] = 0
        item['direction'] = 0
        item['health'] = 0
        kb['mouse'] = item

    # link info
    item = {}
    item['x'] = ram[112]
    item['y'] = ram[132]
    item['dx'] = ram[112] - ram2[112] if ram2 is not None else 0
    item['dy'] = ram[132] - ram2[132] if ram2 is not None else 0
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
    item['dx'] = ram[125] - ram2[125] if ram2 is not None else 0
    item['dy'] = ram[145] - ram2[145] if ram2 is not None else 0
    if item['state'] != 0:
        kb['sword_melee'] = item

    item = {}
    item['state'] = ram[186]
    item['direction'] = ram[166]
    item['type'] = -2
    item['health'] = 0
    item['x'] = ram[126]
    item['y'] = ram[146]
    item['dx'] = ram[126] - ram2[126] if ram2 is not None else 0
    item['dy'] = ram[146] - ram2[146] if ram2 is not None else 0
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
        item['dx'] = ram[113+i] - ram2[113+i] if ram2 is not None else 0
        item['dy'] = ram[133+i] - ram2[133+i] if ram2 is not None else 0
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
        item['dx'] = ram[119+i] - ram2[119+i] if ram2 is not None else 0
        item['dy'] = ram[139+i] - ram2[139+i] if ram2 is not None else 0
        if item['state'] != 0:
            kb[f'projectile_{i}'] = item

    # add tile information last
    subtiles = _get_subtiles(ram)
    view_radius = 2
    if include_background:
        if use_subtiles:
            tiles = subtiles
            tile_size = 8
        else:
            tiles = subtiles[::2,::2]
            tile_size = 16
        kb.info['tiles'] = tiles
        kb.info['tile_size'] = tile_size

        link_x = ram[112]
        link_y = ram[132]
        link_tile_x = (link_x+8)//tile_size
        link_tile_y = (link_y-56)//tile_size
        if not use_subtiles:
            link_tile_y = (link_y-48)//tile_size

        mapstr = ''
        for y in range(tiles.shape[1]):
            for x in range(tiles.shape[0]):
                if abs(int(x) - int(link_tile_x)) + abs(int(y) - int(link_tile_y)) <= view_radius:
                    item = {}
                    item['x'] = x*tile_size 
                    item['y'] = y*tile_size + 61
                    item['dx'] = 0
                    item['dy'] = 0
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
        kb.info['mapstr'] = mapstr

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

    # add events
    kb.events = get_events(ram, ram2)

    return kb

# FIXME:
# Zelda has two tilemaps, one for the overworld and one for the underworld.
# Each tile is represented as a single byte.
# The overworld and underworld have different sets of passable tiles,
# and only a small set of passable tiles is listed below.
# These should be expanded to the full set,
# and the code should be adjusted to be aware of if it's in over/underworld.
# The lists below is a small list where the passable tiles seem compatible in both situations and capture most of the information.
passable_tiles_overworld = [0x24, 0x26, 0x76, 0x84]
passable_tiles_underworld = [0x68, 0x74, 0x7e]
ignore_tile_set = passable_tiles_overworld + passable_tiles_underworld

def _get_subtiles(ram):
    # The screen is divided into an 11x16 grid,
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

    return subtiles


def _isscreen_changeable(ram, direction):
    '''
    Return True if Link can change screens in the specified direction.
    This function is used to determine if the `screen_*` set of tasks are possible.

    FIXME:
    Link can also sometimes enter a cave.
    We should detect when this is possible and add a new task for entering caves.
    '''
    subtiles = _get_subtiles(ram)
    if direction == 'NORTH':
        return any(np.isin(subtiles[:,0], ignore_tile_set))
    if direction == 'SOUTH':
        return any(np.isin(subtiles[:,21], ignore_tile_set))
    if direction == 'EAST':
        return any(np.isin(subtiles[0,:], ignore_tile_set))
    if direction == 'WEST':
        return any(np.isin(subtiles[31,:], ignore_tile_set))
    assert False, 'should not happen'
    

########################################
# MARK: event functions
########################################

def get_events(ram, ram2):
    prefix = '_event_'
    if not hasattr(get_events, 'event_names'):
        get_events.event_names = [f for f in globals() if f.startswith(prefix)]
        get_events.event_names.sort()
    ret = {}
    for event_name in get_events.event_names:
        f = globals()[event_name]
        if ram2 is not None:
            ret[event_name[len(prefix):]] = f(ram, ram2)
        else:
            ret[event_name[len(prefix):]] = 0
    return ret

def _event_link_killed(ram, ram2):
    return int(_ramstate_hearts(ram) == 0 and _ramstate_hearts(ram2) != 0)

def _event_link_hit(ram, ram2):
    return round(2*max(0, _ramstate_hearts(ram2) - _ramstate_hearts(ram)))

def _event_enemy_alldead(ram, ram2):
    return int((_ramstate_all_enemies_health(ram) == 0 and _ramstate_all_enemies_health(ram2) != 0) and not (_ramstate_is_screen_scrolling(ram) or _ramstate_is_cave_enter(ram)))

def _event_screen_scrolling(ram, ram2):
    return int(_ramstate_is_screen_scrolling(ram) and not _ramstate_is_screen_scrolling(ram2))

def _event_screen_scrolling_north(ram, ram2):
    return int(ram[132] == 61) * _event_screen_scrolling(ram, ram2)

def _event_screen_scrolling_south(ram, ram2):
    return int(ram[132] == 221) * _event_screen_scrolling(ram, ram2)

def _event_screen_scrolling_west(ram, ram2):
    return int(ram[112] == 0) * _event_screen_scrolling(ram, ram2)

def _event_screen_scrolling_east(ram, ram2):
    return int(ram[112] == 240) * _event_screen_scrolling(ram, ram2)

def _event_screen_cave(ram, ram2):
    return int(_ramstate_is_cave_enter(ram) and not _ramstate_is_cave_enter(ram2))

def _event_nomouse_enemy_hit(ram, ram2):
    return _event_enemy_hit(ram, ram2) and not _event_mouse_enemy_hit(ram, ram2)

def _event_mouse_enemy_killed(ram, ram2):
    return _event_enemy_killed(ram, ram2) and _event_mouse_enemy_hit(ram, ram2)

def _event_mouse_enemy_hit(ram, ram2):
    if ram is None or ram.mouse is None or ram2 is None or ram2.mouse is None:
        return 0
    mindist = 999
    mini = None
    for i in range(6):
        dist = abs(ram[113+i] - ram.mouse['x']) + abs(ram[133+i] - ram.mouse['y'])
        if dist < mindist:
            mindist = dist
            mini = i
    return max(0, _ramstate_enemy_health(ram2, mini) - _ramstate_enemy_health(ram, mini))

def _ramstate_enemy_health(ram, i):
    rawhealth = ram[1158+i]
    if rawhealth > 0 and (rawhealth < 16 or rawhealth >= 128):
        return 0
    return rawhealth//16

def _event_enemy_hit(ram, ram2):
    return max(0, _ramstate_all_enemies_health(ram2) - _ramstate_all_enemies_health(ram))

def _event_enemy_killed(ram, ram2):
    return max(0, _ramstate_total_enemies(ram2) - _ramstate_total_enemies(ram))

def _event_link_onmouse(ram, ram2):
    return int(_ramstate_link_onmouse(ram) and not _ramstate_link_onmouse(ram2))

def _event_link_l1dist_decrease(ram, ram2):
    if ram.mouse is None or not hasattr(ram2, 'mouse') or ram2.mouse is None:
        return 0
    else:
        x1 = ram[112]
        y1 = ram[132]
        l1dist1 = abs(ram.mouse['x'] - x1) + abs(ram.mouse['y'] - y1)
        x2 = ram2[112]
        y2 = ram2[132]
        l1dist2 = abs(ram2.mouse['x'] - x2) + abs(ram2.mouse['y'] - y2)
        return min(max(l1dist2 - l1dist1, 0), 1)

def _event_link_l1dist_increase(ram, ram2):
    if ram.mouse is None or not hasattr(ram2, 'mouse') or ram2.mouse is None:
        return 0
    else:
        x1 = ram[112]
        y1 = ram[132]
        l1dist1 = abs(ram.mouse['x'] - x1) + abs(ram.mouse['y'] - y1)
        x2 = ram2[112]
        y2 = ram2[132]
        l1dist2 = abs(ram2.mouse['x'] - x2) + abs(ram2.mouse['y'] - y2)
        return min(max(l1dist1 - l1dist2, 0), 1)

def _event_button_push(ram, ram2):
    return min(1, abs(ram[0xfa] - ram2[0xfa]))

def _event_button_arrow_nomove(ram, ram2):
    arrow1 = ram[0xfa] & 15
    arrow2 = ram2[0xfa] & 15
    x1 = ram[112]
    y1 = ram[132]
    x2 = ram2[112]
    y2 = ram2[132]
    return arrow1 and x1 == x2 and y1 == y2

def _event_add_bomb(ram, ram2):
    return max(0, ram[1624] - ram2[1624])

def _event_add_bombmax(ram, ram2):
    return max(0, ram[1660] - ram2[1660])

def _event_add_keys(ram, ram2):
    return max(0, ram[1646] - ram2[1646])

def _event_add_ruppees(ram, ram2):
    return max(0, ram[1645] - ram2[1645])

def _event_add_clock(ram, ram2):
    return max(0, ram[1644] - ram2[1644])

def _event_add_heart(ram, ram2):
    return max(0, _ramstate_hearts(ram) - _ramstate_hearts(ram2))

########################################
# MARK: _ramstate
########################################

def _ramstate_link_onmouse(ram):
    if not hasattr(ram, 'mouse') or ram.mouse is None:
        return 0
    else:
        safe_radius = 2
        link_x = ram[112]
        link_y = ram[132]
        l1dist = abs(ram.mouse['x'] - link_x) + abs(ram.mouse['y'] - link_y)
        return l1dist <= safe_radius

# The _ramstate functions modified from gym-zelda-1 repo;
# that repo uses the nesgym library, which is a pure python nes emulator
# and about 10x slower than the gymretro environment;
def _ramstate_screen_change(ram):
    return _ramstate_is_screen_scrolling(ram) or _ramstate_is_cave_enter(ram)

def _ramstate_is_screen_scrolling(ram):
    SCROLL_GAME_MODES = {4, 6, 7}
    return ram[0x12] in SCROLL_GAME_MODES

def _ramstate_screen_scrolling_north(ram):
    return int(ram[132] == 61) * _ramstate_is_screen_scrolling(ram)

def _ramstate_screen_scrolling_south(ram):
    return int(ram[132] == 221) * _ramstate_is_screen_scrolling(ram)

def _ramstate_screen_scrolling_west(ram):
    return int(ram[112] == 0) * _ramstate_is_screen_scrolling(ram)

def _ramstate_screen_scrolling_east(ram):
    return int(ram[112] == 240) * _ramstate_is_screen_scrolling(ram)

def _ramstate_is_drawing_text(ram):
    return ram[0x0605] == 0x10

def _ramstate_is_cave_enter(ram):
    return ram[0x0606] == 0x08

def _ramstate_is_inventory_scroll(ram):
    return 65 < ram[0xfc]

def _ramstate_is_openning_scene(ram):
    return bool(ram[0x007c])

def _ramstate_link_killed(ram):
    return _ramstate_hearts(ram) <= 0

def _ramstate_link_alive(ram):
    return _ramstate_hearts(ram) > 0

def _ramstate_hearts(ram):
    link_full_hearts = 0x0f & ram[0x066f]
    link_partial_hearts = ram[0x0670] / 255
    return link_full_hearts + link_partial_hearts

def _ramstate_all_enemies_dead(ram):
    return all([ram[848+i] == 0 for i in range(6)])

def _ramstate_all_enemies_health(ram):
    healths = []
    for i in range(6):
        rawhealth = ram[1158+i]
        if rawhealth > 0 and (rawhealth < 16 or rawhealth >= 128):
            healths.append(0)
        else:
            healths.append(rawhealth//16)
    return sum(healths)

def _ramstate_total_enemies(ram):
    healths = []
    for i in range(6):
        rawhealth = ram[1158+i]
        if rawhealth > 0 and (rawhealth < 16 or rawhealth >= 128):
            healths.append(0)
        else:
            healths.append(rawhealth//16)
    return sum([1 for health in healths if health > 0])

def _ramstate_screen_change(ram):
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

        if kind == 'move':
            num_action_space = 4
        elif kind == 'moveA':
            num_action_space = 8
        elif kind == 'moveAB':
            num_action_space = 12
        else:
            num_action_space = 16

        assert isinstance(env.action_space, gymnasium.spaces.MultiBinary)

        self.button_count = env.action_space.n
        self.buttons = env.unwrapped.buttons

        self._decode_discrete_action = []
        for action in self.actions:
            arr = np.array([False] * self.button_count)

            for button in action:
                arr[self.buttons.index(button)] = True

            self._decode_discrete_action.append(arr)

        self.action_space = gymnasium.spaces.Discrete(num_action_space)

    def action(self, action):
        if isinstance(action, list) and len(action) and isinstance(action[0], str):
            arr = np.array([False] * self.button_count)
            for button in action:
                arr[self.buttons.index(button)] = True

            return arr

        return self._decode_discrete_action[action].copy()
