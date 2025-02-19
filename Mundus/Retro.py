import logging
import random
import re

import numpy as np
import gymnasium
import retro

# load font for writing to image
from PIL import Image, ImageDraw, ImageFont
pil_font = ImageFont.load_default()

from .util import *


from multiprocessing import Queue, Process
import queue


class ForkedRetroEnv(gymnasium.Wrapper):
    '''
    This class is used for parallelizing evaluation of the environment.
    It creates a child process that runs only the emulator
    so that the parent process can run non-emulator tasks.
    The class was designed for mobile hardware that has several slow CPUs.
    An individual CPU was not powerful enough to handle the entire play pipeline,
    so this class can be used to offload the emulation work to one CPU
    while another CPU handles the non-emulation work.

    NOTE:
    An unavoidable consequence of the parallelism is that
    the parent process will always be one frame behind the child process.
    So any agent (human/AI) will necessarily have a one frame lag to their inputs.
    Human players won't notice the lag,
    but overfit AIs might experience problems.

    NOTE:
    There is a lot of black magic in the implementation here,
    and the code is only lightly tested.
    There are probably subtle bugs.
    It should be fine for interactive use of the emulator,
    but might (or might not!) cause weird problems with training.

    FIXME:
    Currently, this Wrapper must be run directly on the RetroEnv instance.
    It should be possible to make it a bit more generic so that it can work
    on any wrapper.
    '''
    def __init__(self, env):
        super().__init__(env)
        
        # qin stores messages that go to the worker;
        # qout stores messages that come from the worker;
        # they are of size 1 to ensure syncronization between the worker and parent process;
        # if one process gets ahead of the other process,
        # it will block trying to add a second entry into the queue
        self.qin = Queue(maxsize=1)
        self.qout = Queue(maxsize=1)

        # start the emulator process
        def emulator_worker():
            # disable rendering in the worker environement
            # to ensure that multiple rendering windows do not get opened
            env.render_mode = 'rgb_array'

            # the worker will loop forever;
            # it blocks on the qin.get() line waiting for a new command;
            # then it runs the specified command and places the resulting emulator state in qout
            while True:
                message_in = self.qin.get()
                message_out = {}
                if message_in['method'] == 'step':
                    message_out['return'] = env.step(message_in['action'])
                elif message_in['method'] == 'reset':
                    message_out['return'] = env.reset()
                message_out['ram'] = env.get_ram()
                message_out['img'] = env.img
                message_out['audio'] = self.env.unwrapped.em.get_audio()
                self.qout.put(message_out)
        p = Process(target=emulator_worker)
        p.start()

        # NOTE:
        # the code below tries to provide a backwards-compatible interface
        # to the original environment and emulator;
        # FIXME
        # the code below does not support the full RetroEnv interface,
        # only those that I have needed for my downstream tasks
        oldenv = self.env
        
        # the ._internal_XXX properties store internal states of the emulator;
        # these will be updated by the ._update_state() method
        # when we poll the emulator worker for the current state
        self._internal_ram = oldenv.get_ram()
        self._internal_audio = np.array([])
        self._internal_audio_rate = oldenv.em.get_audio_rate()

        # we don't want the downstream users of our wrapper to have to interact
        # with the ._internal_XXX properties directly,
        # so we replicate the interface provided by RetroEnv here
        self.env = type('FakeRetroEnv', (), {})()
        self.__dict__['unwrapped'] = self.env
        self.env.__dict__['unwrapped'] = self.env
        self.env.action_space = oldenv.action_space
        self.env.buttons = oldenv.buttons
        self.env.em = type('FakeEmulator', (), {})()
        self.env.em.get_audio_rate = lambda: self._internal_audio_rate
        self.env.render_mode = oldenv.render_mode
        self.env.viewer = oldenv.viewer
        self.env.render = lambda: retro.RetroEnv.render(self.env)
        self.env.get_ram = lambda: self._internal_ram
        self.env.em.get_audio = lambda: self._internal_audio

        # delete the old environment;
        # this frees resources and ensures that we don't accidentally interact
        # with the original emulator instead of the worker process
        del oldenv


    def _update_state(self, message_out):
        self.env.img = message_out['img']
        self._internal_ram = message_out['ram']
        self._internal_audio = message_out['audio']

    def step(self, action):
        # update the main process with the previous state of the emulator
        message_out = self.qout.get()
        self._update_state(message_out)

        # ask the emulator to compute another step
        message_in = {
            'method': 'step',
            'action': action,
            }
        self.qin.put(message_in)
        return message_out['return']

    def reset(self, **kwargs):
        # empty the queues;
        # this ensures that the worker process is ready for new commands
        try:
            self.qin.get_nowait()
        except queue.Empty:
            pass

        try:
            self.qout.get_nowait()
        except queue.Empty:
            pass

        # send reset command to emulator_worker
        message_in = {
            'method': 'reset',
            'kwargs': kwargs,
            }
        self.qin.put(message_in)

        # wait for the emulator to complete the reset,
        # and store the resulting state
        message_out = self.qout.get()
        self._update_state(message_out)

        # send a step command to emulator_worker;
        # this initial step will not have any action taken;
        # the worker will run the step while the main process continues
        message_in = {
            'method': 'step',
            'action': self.env.action_space.sample() * 0,
            }
        self.qin.put(message_in)
        return message_out['return']


class RetroWithRam(gymnasium.Wrapper):
    '''
    The .get_ram() method is relatively expensive.
    The purpose of this wrapper is to:
    1. store the result of .get_ram() in an attribute to prevent multiple calls.
    2. track a history of the ram state
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
        self.ram = None
        self._update_ram()

    def _update_ram(self):
        self.ram2 = self.ram
        self.ram = self._IntLookup(self.unwrapped.get_ram())

    def step(self, action):
        self._update_ram()
        return super().step(action)

    def reset(self, **kwargs):
        self.ram = self.unwrapped.get_ram()
        obs, info = super().reset(**kwargs)
        self.ram2 = None
        self.ram = self.unwrapped.get_ram()
        return obs, info


class RetroKB(RetroWithRam):
    def __init__(
            self,
            env,
            task_regex='attack',
            stdout_debug=False,
            debug_assert=False,
            no_render_skipped_frames=True,
            skip_boring_frames=True,
            render_kb=True,
            seed=None,
            ):
        
        # bookkeeping
        super().__init__(env)
        self.seed = seed
        self.stdout_debug = stdout_debug
        self.debug_assert = debug_assert
        self.no_render_skipped_frames=no_render_skipped_frames
        self.skip_boring_frames = skip_boring_frames 
        self.render_kb = render_kb

        # self._valid_tasks contains all the tasks that the reset() function might select
        pattern = re.compile(task_regex)
        self.valid_tasks = sorted([task for task in self.tasks.keys() if pattern.match(task)])

        # we have two random number generators
        # self.random should be used for generating random numbers,
        # it can be used wherever
        # self._random_reset is used to generate seeds for self.random;
        # it ensures that the selection of environments is consistent
        # no matter how many random numbers are generated between calls to reset();
        # it should only be accessed within the reset() method
        self.random = None
        self._random_reset = random.Random(self.seed)

        # FIXME:
        # this stuff shouldn't be hardcoded
        import Mundus.Zelda
        self.game_module = Mundus.Zelda
        self.generate_knowledge_base = self.game_module.generate_knowledge_base

        # create a new observation space
        kb = self.generate_knowledge_base(self.ram, self.ram2)
        self.observation_space = kb.get_observation_space()
        self.observation_space['rewards'] = self.observation_space['events']
        logging.info(f'self.observation_space.shape={self.observation_space.shape}')
        logging.info(f"self.observation_space.keys()={self.observation_space.keys()}")
        for k in self.observation_space:
            logging.info(f"self.observation_space[k].shape={self.observation_space[k].shape}")

        self.mouse = None

    def reset(self, **kwargs):
        # reset the random number generator
        self.random = random.Random(self._random_reset.randint(0, 2**30))

        # reset all counters
        self.mouse = None
        self.stepcount = 0
        self.skipped_frames = 0
        self.episode_event_multiset = MultiSet()
        self.episode_reward_multiset = MultiSet()
        self.episode_pseudoreward_multiset = MultiSet()
        self.episode_reward = 0
        self.episode_pseudoreward = 0
        self.episode_task = self.random.choice(self.valid_tasks)
        return super().reset(**kwargs)

    def _step_silent(self, buttons=[], force_norender=False):
        '''
        Step the emulator, but don't register as a step in the environment.
        This is useful for skipping "boring" frames in the game
        where the agent has no available actions.

        NOTE:
        buttons is different from action in .step();
        it should be a list of strings, 
        where the entries are the names of the buttons to pass to the emulator;
        this is a universal format that doesn't depend on the action space
        '''
        asd
        self.skipped_frames += 1
        self._set_buttons(buttons)
        self.env.em.step()
        self.env.data.update_ram()
        self._update_ram()
        ob = self.env._update_obs()
        if (self.env.render_mode == "human" 
                and not self.no_render_skipped_frames
                and not force_norender
                ):
            self.env.render()

    def step(self, action):
        # step the emulator;
        # NOTE:
        # we do not render here, but render later if desired;
        # this allows us to write to the screen if we need to
        render_mode = self.env.render_mode
        self.unwrapped.render_mode = 'rgb_array'
        observation, reward, terminated, truncated, info = super().step(action)
        self.unwrapped.render_mode = render_mode

        # record step counts
        self.stepcount += 1
        info['misc_step'] = self.stepcount
        info['misc_skipped_frames'] = self.skipped_frames

        # compute the task information
        self.ram.mouse = self.mouse # FIXME: this should be in only one spot
        kb = self.generate_knowledge_base(self.ram, self.ram2)
        kb_obs = kb.to_observation()
        task = self.tasks[self.episode_task]

        kb_obs['rewards'] = np.array([task['reward'].get(k, 0) for k in sorted(kb.events)])

        terminated = any([getattr(self.game_module, fname)(self.ram) for fname in task['terminated']])
        info['is_success'] = any([getattr(self.game_module, fname)(self.ram) for fname in task['is_success']])
        if 'step' in task:
            task['step'](self, kb)

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

        info['episode_task'] = self.episode_task
        info['event__success'] = info['is_success']

        info['misc_episode_reward'] = self.episode_reward
        info['misc_episode_pseudoreward'] = self.episode_pseudoreward

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
                    self.unwrapped.img[ymin:ymax+1, xmin] = 255
                    self.unwrapped.img[ymin:ymax+1, xmax] = 255
                    self.unwrapped.img[ymin, xmin:xmax+1] = 255
                    self.unwrapped.img[ymax, xmin:xmax+1] = 255
                else:
                    xmin = max(0, min(239, v['x'] - 4))
                    xmax = max(0, min(239, v['x'] + 4))
                    ymin = max(0, min(223, v['y'] - 4))
                    ymax = max(0, min(223, v['y'] + 4))
                    self.unwrapped.img[ymin:ymax+1, xmin:xmax+1] = [255, 0, 255]

            '''
            # write the task to the screen
            # FIXME:
            # the code below is useful for debugging, but very slow
            # it takes about as much CPU as a single step of the emulator!
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
            '''

        if self.stdout_debug:
            text = ''
            text += kb.display()
            #text += kb.info['mapstr']
            text += f'\n{format_dict_pretty(self.episode_event_multiset)}'
            text += f'\n'
            text += f'\nepisode_reward = {self.episode_reward:0.4f}'
            text += f'\nepisode_pseudoreward = {self.episode_pseudoreward:0.4f}'
            text += f'\nterminated = {terminated}'
            text += f'\nmap_coord = {hex(self.ram[0xEB])}'
            print(text)

        if render_mode == 'human':
            self.env.render()

        return kb_obs, reward + pseudoreward, terminated, truncated, info

    ########################################
    # MARK: functions to be overwritten by derived classes
    ########################################

    def register_text(self, text):
        logging.info(f'register_text: {text}')

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

