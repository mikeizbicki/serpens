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
        self.supports_want_render = hasattr(env, "supports_want_render")

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
    def __init__(self, env, path):
        super().__init__(env)
        self.path = path

    def reset(self, **kwargs):
        results = super().reset(**kwargs)
        states = [os.path.basename(filepath).split('.')[0] for filepath in glob.glob(self.path + '/*.state')]
        newstate = random.choice(states)
        logging.info(f"newstate={newstate}")
        self.env.load_state(newstate, inttype=retro.data.Integrations.ALL)
        return results


class ZeldaWrapper(gymnasium.Wrapper):
    '''
    '''
    def __init__(self, env, stdout_debug=False, use_full_subtiles=False, link_view_radius=2):
        super().__init__(env)
        self.stdout_debug = stdout_debug
        self.use_full_subtiles = use_full_subtiles
        self.link_view_radius = link_view_radius
        self.reset()

    def reset(self, **kwargs):
        results = super().reset(**kwargs)

        # create the new variables with default values so that future
        link_view = np.zeros([1+2*self.link_view_radius, 1+2*self.link_view_radius])
        for y in range(link_view.shape[0]):
            for x in range(link_view.shape[1]):
                self.env.data.set_value(f'link_view_{x:2d}_{y:2d}', 0)
        values = self.env.data.lookup_all()
        for k in values:
            if 'direction' == k[-9:]:
                self.env.data.set_value(k+'_east', 0)
                self.env.data.set_value(k+'_west', 0)
                self.env.data.set_value(k+'_south', 0)
                self.env.data.set_value(k+'_north', 0)
                self.env.data.set_value(k+'_other', 0)
        return results

    def step(self, action):
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

        values = self.env.data.lookup_all()
        link_tile_x = (values[f'link_char_x']+8)//16
        link_tile_y = (values[f'link_char_y']-56)//16
        if not self.use_full_subtiles:
            link_tile_y = (values[f'link_char_y']-48)//16

        padded_tiles = np.pad(tiles, self.link_view_radius, constant_values=0)
        link_view = padded_tiles[
                link_tile_y : link_tile_y + 2*self.link_view_radius+1,
                link_tile_x : link_tile_x + 2*self.link_view_radius+1,
                ]

        for y in range(link_view.shape[0]):
            for x in range(link_view.shape[1]):
                self.env.data.set_value(f'link_view_{x:2d}_{y:2d}', link_view[y,x])
        
        for k in values:
            if 'direction' == k[-9:]:
                self.env.data.set_value(k+'_east', int(values[k] == 1))
                self.env.data.set_value(k+'_west', int(values[k] == 2))
                self.env.data.set_value(k+'_south', int(values[k] == 4))
                self.env.data.set_value(k+'_north', int(values[k] == 8))
                self.env.data.set_value(k+'_other', int(values[k] > 8))

        for k in values:
            if 'health' == k[-6:]:
                if values[k] > 0 and (values[k] < 16 or values[k] >= 128):
                    health_simple = 0
                    health_weird = 1
                else:
                    health_simple = values[k]/16
                    health_weird = 0
                self.env.data.set_value(k+'_simple', health_simple)
                self.env.data.set_value(k+'_weird', health_weird)

        if self.stdout_debug:
            values = self.env.data.lookup_all()
            def print_tiles(tiles):
                print('========================================')
                for i in range(tiles.shape[0]):
                    for j in range(tiles.shape[1]):
                        if (j,i) == (link_tile_x, link_tile_y):
                            print(' # ', end='')
                        else:
                            if tiles[i,j] & 32 == 32:
                                print('   ', end='')
                            else:
                                print(f'{hex(tiles[i,j])[2:]:2} ', end='')
                        #if subtiles[i,j] == 38:
                            #print(' ', end='')
                        #else:
                            #print('X', end='')
                    print()
            print_tiles(tiles)
            print_tiles(link_view)
            print('========================================')
            x = values[f'link_char_x']
            y = values[f'link_char_y']
            d = values[f'link_char_direction']
            d1 = values.get(f'link_char_direction_north', '-')
            d2 = values.get(f'link_char_direction_south', '-')
            d3 = values.get(f'link_char_direction_east', '-')
            d4 = values.get(f'link_char_direction_west', '-')
            d5 = values.get(f'link_char_direction_other', '-')
            a = values[f'link_char_animation']
            print(f'   link x,y: {x:3d},{y:3d}  dir:{d:3d} {d1}{d2}{d3}{d4}{d5} state:{a:3d}') 
            x = values[f'link_sword_melee_x']
            y = values[f'link_sword_melee_y']
            d = values[f'link_sword_melee_direction']
            d1 = values.get(f'link_sword_melee_direction_north', '-')
            d2 = values.get(f'link_sword_melee_direction_south', '-')
            d3 = values.get(f'link_sword_melee_direction_east', '-')
            d4 = values.get(f'link_sword_melee_direction_west', '-')
            d5 = values.get(f'link_sword_melee_direction_other', '-')
            a = values[f'link_sword_animation']
            print(f'sword_m x,y: {x:3d},{y:3d}  dir:{d:3d} {d1}{d2}{d3}{d4}{d5} state:{a:3d}') 
            x = values[f'link_sword_projectile_x']
            y = values[f'link_sword_projectile_y']
            d = values[f'link_sword_projectile_direction']
            d1 = values.get(f'link_sword_projectile_direction_north', '-')
            d2 = values.get(f'link_sword_projectile_direction_south', '-')
            d3 = values.get(f'link_sword_projectile_direction_east', '-')
            d4 = values.get(f'link_sword_projectile_direction_west', '-')
            d5 = values.get(f'link_sword_projectile_direction_other', '-')
            s = values[f'link_sword_projectile_state']
            print(f'sword_p x,y: {x:3d},{y:3d}  dir:{d:3d} {d1}{d2}{d3}{d4}{d5} state:{s:3d}') 
            for i in '123456':
                x = values[f'enemy_{i}_x']
                y = values[f'enemy_{i}_y']
                d = values[f'enemy_{i}_direction']
                t = values[f'enemy_{i}_type']
                h = values[f'enemy_{i}_health_simple']
                hw = values[f'enemy_{i}_health_weird']
                d1 = values.get(f'enemy_{i}_direction_north', '-')
                d2 = values.get(f'enemy_{i}_direction_south', '-')
                d3 = values.get(f'enemy_{i}_direction_east', '-')
                d4 = values.get(f'enemy_{i}_direction_west', '-')
                d5 = values.get(f'enemy_{i}_direction_other', '-')
                px = values.get(f'projectile_{i}_x', ' --')
                py = values.get(f'projectile_{i}_y', ' --')
                pd = values.get(f'projectile_{i}_direction', ' --')
                pd1 = values.get(f'projectile_{i}_direction_north', '-')
                pd2 = values.get(f'projectile_{i}_direction_south', '-')
                pd3 = values.get(f'projectile_{i}_direction_east', '-')
                pd4 = values.get(f'projectile_{i}_direction_west', '-')
                pd5 = values.get(f'projectile_{i}_direction_other', '-')
                drop = values[f'drop_enemy{i}']
                s = values.get(f'enemystate_{i}', ' --')
                c = values.get(f'countdown_enemy{i}')
                print(f'enemy {i} x,y: {x:3},{y:3}  dir:{d:3} {d1}{d2}{d3}{d4}{d5} state:{s:3} type: {t:2} health: {h:3} {hw:1} drop:{drop:3} count:{c:3} | proj x,y: {px:3},{py:3}, pd: {pd:3} {pd1}{pd2}{pd3}{pd4}{pd5} ') 
        return super().step(action)
