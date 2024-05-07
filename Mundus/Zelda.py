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


class ZeldaWrapper(gymnasium.Wrapper):
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
            ):

        # bookkeeping
        super().__init__(env)
        self.stdout_debug = stdout_debug
        self.use_full_subtiles = use_full_subtiles
        self.link_view_radius = link_view_radius
        self.center_xy = center_xy
        self.output_link_view = output_link_view

        # switch the terminal to the alternate screen
        if self.stdout_debug:
            print('\u001B[?1049h')

        # create a new observation space
        observations = self.generate_observations()
        low = -256
        high = 256
        shape = [len(observations)]
        dtype = np.float16
        self.observation_space = gymnasium.spaces.Box(low, high, shape, dtype, seed=0)
        logging.info(f'observations.shape={shape}')

    def close(self):
        # switch the terminal away from the alternate screen
        if self.stdout_debug:
            print('\u001B[?1049l')

        # cleanly close superclass
        super().close()

    def reset(self, **kwargs):
        self.episode_reward = 0
        obs, info = super().reset(**kwargs)
        return self.observation_space.sample(), info

    def step(self, action):
        new_observations = self.generate_observations()
        new_observations_array = np.array(list(new_observations.values()), dtype=np.float16)
        observation, reward, terminated, truncated, info = super().step(action)
        self.episode_reward += reward

        if self.stdout_debug:
            text = ''
            text += '\x1b[2J' # clear the screen
            text += _observations_to_str(new_observations)
            text += f'\ntotal_variables: {len(new_observations)}'
            text += f'\nepisode_reward = {self.episode_reward:0.4f}'
            print(text)

        return new_observations_array, reward, terminated, truncated, info

    def generate_observations(self):
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
        outputs = {}
        inputs = self.env.data.lookup_all()

        # the screen is divided into an 11x16 grid,
        # and each tile is divided into 2x2 subtiles;
        # therefore, the subtile grid is 22x32 (=0x2c0).
        # the leftmost, rightmost, and bottommost subtiles do not get displayed,
        # so only a 21x30 grid of subtiles is displayed
        subtiles = self.env.unwrapped.get_ram()[0x530+0x800:0x530+0x2c0+0x800].reshape([32,22]).T

        # NOTE:
        # accessing the memory through get_state() allows changing the state;
        # the code below is usefull for modifying the grid in a running game;
        # it does not update the pictures (which are drawn at the beginning of the screen),
        # but does update how link/enemies behave on the tiles
        #import code
        #code.interact(local=locals())
        # >>> tiles = self.env.unwrapped.em.get_state()[14657:14657+11*4*16]
        # >>> newstate = b'\x00'*88; state = self.env.unwrapped.em.get_state(); state = state[:14657]+newstate+state[14657+len(newstate):]; self.env.unwrapped.em.set_state(state)

        if self.use_full_subtiles:
            tiles = subtiles
        else:
            tiles = subtiles[::2,::2]

        link_tile_x = (inputs[f'link_char_x']+8)//16
        link_tile_y = (inputs[f'link_char_y']-56)//16
        if not self.use_full_subtiles:
            link_tile_y = (inputs[f'link_char_y']-48)//16

        padded_tiles = np.pad(tiles, self.link_view_radius, constant_values=0)
        link_view = padded_tiles[
                link_tile_y : link_tile_y + 2*self.link_view_radius+1,
                link_tile_x : link_tile_x + 2*self.link_view_radius+1,
                ]

        if self.output_link_view:
            for y in range(link_view.shape[0]):
                for x in range(link_view.shape[1]):
                    outputs[f'link_char_view_{x:2d}_{y:2d}'] = link_view[y,x]

        # link specific data
        if inputs['link_heart_partial'] == 0:
            outputs['link_char_health_simple'] = 0
        else:
            partial = 0.5
            if inputs['link_heart_partial'] > 127:
                partial = 0
            outputs['link_char_health_simple'] = inputs['link_heart_full'] - math.floor(inputs['link_heart_full'] / 16) * 16 - partial + 1
        outputs['link_char_x'] = inputs['link_char_x']
        outputs['link_char_y'] = inputs['link_char_y']
        
        # add appropriate data from the inputs to the outputs
        for k in inputs:

            # add data that needs no special modifications
            if any([v in k for v in ['state', 'drop', 'count', 'direction']]):
                outputs[k] = inputs[k]
           
            # add 1-hot encoded directions
            if 'direction' == k[-9:]:
                outputs[k+'_east'] = int(inputs[k] == 1)
                outputs[k+'_west'] = int(inputs[k] == 2)
                outputs[k+'_south'] = int(inputs[k] == 4)
                outputs[k+'_north'] = int(inputs[k] == 8)
                outputs[k+'_other'] = int(inputs[k] > 8)

            # process the monster health to a usable format
            if 'health' == k[-6:]:
                if inputs[k] > 0 and (inputs[k] < 16 or inputs[k] >= 128):
                    health_simple = 0
                    health_weird = 1
                else:
                    health_simple = inputs[k]//16
                    health_weird = 0
                outputs[k+'_simple'] = health_simple
                outputs[k+'_weird'] = health_weird

            # add relative positions
            if 'link_char' not in k:
                if k[-2:] == '_x':
                    outputs[k + 'rel'] = inputs[k] - inputs['link_char_x']
                if k[-2:] == '_y':
                    outputs[k + 'rel'] = inputs[k] - inputs['link_char_y']

            # add distance
            if 'link_char' not in k:
                if k[-2:] == '_x':
                    base = k[:-2]
                    xdist = abs(inputs[base + '_x'] - inputs['link_char_x'])
                    ydist = abs(inputs[base + '_y'] - inputs['link_char_y'])
                    outputs[base + '_dist'] = xdist + ydist

            # add similarity positions
            if 'link_char' not in k:
                if k[-2:] == '_x':
                    outputs[k + 'sim'] = 10/(abs(inputs[k] - inputs['link_char_x']) + 10)
                if k[-2:] == '_y':
                    outputs[k + 'sim'] = 10/(abs(inputs[k] - inputs['link_char_y']) + 10)

            # add centered absolute positions
            if False:
                if k[-1] == 'x':
                    outputs[k+'center'] = inputs[k] - 128
                if k[-1] == 'y':
                    outputs[k+'center'] = inputs[k] - 56 - 88

        # clean variables for dead entities
        def prefixnum_is_active(prefixnum):
            health = outputs.get(prefixnum + '_health_simple', 0)
            drop = outputs.get(prefixnum + '_drop', 0)
            state = outputs.get(prefixnum + '_state', 0)
            return health > 0 or drop > 0 or state > 0 # or 'projectile' in prefixnum


        if True:
            prefixnums = set(['_'.join(k.split('_')[:2]) for k in outputs])
            for prefixnum in prefixnums:
                if not prefixnum_is_active(prefixnum):
                    def reset(k, v):
                        if k in outputs:
                            outputs[k] = v
                    reset(prefixnum + '_x', 999)
                    reset(prefixnum + '_y', 999)
                    reset(prefixnum + '_xrel', 999)
                    reset(prefixnum + '_yrel', 999)
                    reset(prefixnum + '_xcenter', 999)
                    reset(prefixnum + '_ycenter', 999)
                    reset(prefixnum + '_dist', 999)
                    reset(prefixnum + '_xsim', 0.0)
                    reset(prefixnum + '_ysim', 0.0)
                    reset(prefixnum + '_direction', 0)

        # reorder the enemies/projectiles
        def reorder_dictionary(outputs0, prefix):
            def key(prefixnum):
                distance = abs(outputs0[prefixnum+'_xrel']) + abs(outputs0[prefixnum+'_yrel']) + 1000
                if prefixnum_is_active(prefixnum):
                    penalty = 0
                else:
                    penalty = 10000
                return distance + penalty
            attrs = set([k[len(prefix)+2:] for k in outputs0 if k.startswith(prefix) and k[len(prefix)+2] == '_'])
            prefixnums = list(set(k[:len(prefix)+2] for k in outputs0 if k.startswith(prefix) and k[len(prefix)+2] == '_'))
            prefixnums.sort(key=key)
            outputs1 = copy.copy(outputs0)
            for i, prefixnum in enumerate(prefixnums):
                for attr in attrs:
                    if prefixnum + attr in outputs0:
                        outputs1[f'{prefix}_{i+1}'+attr] = outputs0[prefixnum + attr]
            return outputs1
        #outputs = reorder_dictionary(outputs, prefix='projectile')
        #outputs = reorder_dictionary(outputs, prefix='enemy')

        # filter outputs
        #outputs = {k:v for k,v in outputs.items() if 'enemy_1' in k or 'link' in k}
        keeps = ['enemy', 'link_char', 'link_sword', 'projectile']
        #keeps = ['enemy_1', 'link_char', 'projectile_1']
        #keeps = ['enemy', 'link_char']
        outputs = {k:v for k,v in outputs.items() if any([keep in k for keep in keeps])}

        #keeps = ['_x', '_y', 'drop', 'count']
        #outputs = {k:v for k,v in outputs.items() if any([keep in k for keep in keeps])}

        # return the output observations
        return outputs


def _get_attrs(observations, prefix):
    attrs = set()
    for k in observations:
        if k.startswith(prefix):
            attrs.add('_'.join(k.split('_')[2:]))
    return attrs
    #return set([k[len(prefix)+3:] for k in observations if k.startswith(prefix) and k[len(prefix)+2: len(prefix)+3] == '_'])


def _get_prefixnums(observations):
    return set(['_'.join(k.split('_')[:2]) for k in observations])


def _get_prefixes(observations):
    return set(['_'.join(k.split('_')[:1]) for k in observations])


def _observations_to_str(outputs):
    '''
    '''
    lines = []
    prefixnums = _get_prefixnums(outputs)
    prefixnums_width = max([len(x) for x in prefixnums])
    prefixes = _get_prefixes(outputs)
    attrs = []
    for prefix in prefixes:
        attrs.extend(_get_attrs(outputs, prefix))

    def display_tiles(tiles):
        ret = ''
        for i in range(tiles.shape[0]):
            for j in range(tiles.shape[1]):
                if (j,i) == (link_tile_x, link_tile_y):
                    ret += ' # '
                else:
                    if tiles[i,j] & 32 == 32:
                        ret += '   '
                    else:
                        ret += f'{hex(tiles[i,j])[2:]:2} '
            ret += '\n'
        return ret
    #lines.append(display_tiles(tiles))
    #lines.append('====================')
    #lines.append(display_tiles(link_view))
    #lines.append('====================')

    def makeline(name):
        dispname = name[:7]
        if name[-2] == '_':
            dispname = name[:5] + name [-2:]
        textout = f'{name:{prefixnums_width}} '
        if 'x' in attrs:
            x = outputs.get(f'{name}_x', '-')
            y = outputs.get(f'{name}_y', '-')
            textout += f'x,y: {x:4},{y:4} '
        if 'xcenter' in attrs:
            xc = outputs.get(f'{name}_xcenter', '-')
            yc = outputs.get(f'{name}_ycenter', '-')
            textout += f'xc,yc: {xc:4},{yc:4} '
        if 'dist' in attrs:
            dist = outputs.get(f'{name}_dist', '-')
            textout += f'dist: {dist:4} '
        if 'xsim' in attrs:
            xc = outputs.get(f'{name}_xsim')
            yc = outputs.get(f'{name}_ysim')
            if xc is not None:
                textout += f'xsim,ysim: {xc:0.2f},{yc:0.2f} '
            else:
                textout += f'xsim,ysim:   - ,   - '
        if 'xrel' in attrs:
            xrel = outputs.get(f'{name}_xrel', '-')
            yrel = outputs.get(f'{name}_yrel', '-')
            textout += f'xrel,yrel: {xrel:4},{yrel:4} '
        d = outputs.get(f'{name}_direction', '-')
        d1 = outputs.get(f'{name}_direction_north', '-')
        d2 = outputs.get(f'{name}_direction_south', '-')
        d3 = outputs.get(f'{name}_direction_east', '-')
        d4 = outputs.get(f'{name}_direction_west', '-')
        d5 = outputs.get(f'{name}_direction_other', '-')
        a = outputs.get(f'{name}_state', '-')
        c = outputs.get(f'{name}_countdown', '-')
        drop = outputs.get(f'{name}_drop', '-')
        h = outputs.get(f'{name}_health_simple', '-')
        hw = outputs.get(f'{name}_health_weird', '-')
        textout += f'dir:{d:3} {d1}{d2}{d3}{d4}{d5} health: {h:3} {hw:1} state: {a:3} drop: {drop:3} count: {c:3}'
        return textout

    lines.append('--------------------')
    for prefixnum in sorted(prefixnums):
        lines.append(makeline(prefixnum))
    lines.append('--------------------')

    return '\n'.join(lines)
