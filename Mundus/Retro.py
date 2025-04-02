import atexit
import json
import logging
import multiprocessing
import random
import re
import os
import sys

import numpy as np
import gymnasium
import retro
import setproctitle
import xxhash

from Mundus.Object import *
import Mundus.Games.NES_Generic

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
            setproctitle.setproctitle(f'serpens: emulator_worker')

            # disable rendering in the worker environement
            # to ensure that multiple rendering windows do not get opened
            env.render_mode = 'rgb_array'

            # the worker will loop forever;
            # it blocks on the qin.get() line waiting for a new command;
            # then it runs the specified command and places the resulting emulator state in qout
            last_step_result = None
            while True:
                message_in = self.qin.get()
                message_out = {}
                if message_in['method'] == 'step':
                    last_step_result = env.step(message_in['action'])
                    message_out['return'] = last_step_result
                elif message_in['method'] == 'em.step':
                    a, b = message_in['button_mask']
                    env.em.set_button_mask(a, b)
                    message_out['return'] = last_step_result
                    env.em.step()
                    env.data.update_ram()
                    env.img = env.em.get_screen()
                elif message_in['method'] == 'reset':
                    message_out['return'] = env.reset()
                message_out['ram'] = env.get_ram()
                message_out['img'] = env.img
                message_out['audio'] = self.env.unwrapped.em.get_audio()
                self.qout.put(message_out)
        p = Process(name='serpens: emulator_worker', target=emulator_worker)
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
        self.env.data = type('FakeData', (), {})()
        self.env.em = type('FakeEmulator', (), {})()
        self.env.em.get_audio = lambda: self._internal_audio
        self.env.em.get_audio_rate = lambda: self._internal_audio_rate
        self.env.get_ram = lambda: self._internal_ram
        self.env.render = lambda: retro.RetroEnv.render(self.env)
        self.env.render_mode = oldenv.render_mode
        self.env.viewer = oldenv.viewer

        def _set_button_mask(a, b):
            self._internal_button_mask = (a, b)
        self.env.em.set_button_mask = _set_button_mask
        def _em_step():
            message_out = self.qout.get()
            self.qin.put({'method': 'em.step', 'button_mask': self._internal_button_mask})
            self._update_state(message_out)
        self.env.em.step = _em_step
        self.env.data.update_ram = lambda: None

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
            if isinstance(key, int):
                return int(self.ram[key])
            elif isinstance(key, slice):
                return self.ram[key].astype(np.int64)
            else:
                raise ValueError(f'IntegerLookup; type={type(key)}, value={key} ')

    def __init__(self, env):
        super().__init__(env)
        self.ram = None
        #self.screen = None
        self._update_ram()

    def _update_ram(self):
        self.ram2 = self.ram
        self.ram = self._IntLookup(self.unwrapped.get_ram())
        #self.screen2 = self.screen
        #self.screen = self.unwrapped.em.get_screen()
        self.ram.em_state = self.unwrapped.em.get_state()
        self.ram.screen = self.unwrapped.em.get_screen()

    def step(self, action):
        self._update_ram()
        return super().step(action)

    def reset(self, **kwargs):
        self.ram = self._IntLookup(self.unwrapped.get_ram())
        obs, info = super().reset(**kwargs)
        self.ram2 = None
        self.ram = self._IntLookup(self.unwrapped.get_ram())
        #self.screen2 = None
        #self.screen = self.unwrapped.em.get_screen()
        self.ram.em_state = self.unwrapped.em.get_state()
        self.ram.screen = self.unwrapped.em.get_screen()
        return obs, info


class RetroKB(RetroWithRam):

    tasks = {}
    tasks['attack'] = {
        'terminated': [
            ],
        'is_success': [
            ],
        'reward': {
            },
        'pseudoreward': {
            }
        }

    def __init__(
            self,
            env,
            game_module=None,
            task_regex='attack',
            alt_screen=False,
            debug_assert=False,
            no_render_skipped_frames=True,
            skip_boring_frames=True,
            render_kb=True,
            render_downsample=False,
            render_task=True,
            seed=None,
            lang='en',
            kb_kwargs={},
            **kwargs
            ):
        
        # bookkeeping
        super().__init__(env)
        self.seed = seed
        self.alt_screen = alt_screen
        self.debug_assert = debug_assert
        self.no_render_skipped_frames=no_render_skipped_frames
        self.skip_boring_frames = skip_boring_frames 
        self.render_kb = render_kb
        self.render_downsample = render_downsample
        self.render_task = render_task
        self.lang = lang
        self._text_callbacks = []
        self.game_module = game_module

        self.kb_kwargs = kb_kwargs
        self.kb_kwargs['game_module'] = self.game_module
        logging.debug(f"self.kb_kwargs={self.kb_kwargs}")

        # there is lots of important functionality in this class;
        # placing a reference to the RetroKB object in the base RetroEnv class
        # ensures that this functionality is easily findable for all downstream uses
        self.unwrapped.RetroKB = self

        # switch the terminal to the alternate screen
        if alt_screen:
            print('\u001B[?1049h')

            # cause the program to exit the alternate screen when the program exits;
            # the use of the atexit library ensures that we switch
            # even when the program exits abnormally
            def exit_alternate_screen():
                print('\u001B[?1049l')
            atexit.register(exit_alternate_screen)

            # FIXME:
            # the code above doesn't cause print the traceback to the main screen;
            # the code below does, but it also immediately causes the program to crash for some reason;
            # this didn't used to be the case before moving the code into this file.
            # if python crashes, we want the exception printed to the main screen
            # and not the alternate screen
            def crash_handler(type, value, tb):
                print("\033[?1049l")  # exit alternate screen
                if type == KeyboardInterrupt:
                    sys.exit(0)
                else:
                    sys.__excepthook__(type, value, tb)
            sys.excepthook = crash_handler

        # load *_to_text json
        with open(f'data/events.{self.lang}.json') as fin:
            self.event_to_text = json.load(fin)
        with open(f'data/tasks.{self.lang}.json') as fin:
            self.task_to_text = json.load(fin)

        # self._valid_tasks contains all the tasks that the reset() function might select
        pattern = re.compile(task_regex)
        self.valid_tasks = sorted([task for task in self.tasks.keys() if pattern.match(task) and 'interactive' not in task])

        # we have two random number generators
        # self.random should be used for generating random numbers,
        # it can be used wherever
        # self._random_reset is used to generate seeds for self.random;
        # it ensures that the selection of environments is consistent
        # no matter how many random numbers are generated between calls to reset();
        # it should only be accessed within the reset() method
        self.random = None
        self._random_reset = random.Random(self.seed)

        # create a new observation space
        kb = generate_knowledge_base(self.ram, self.ram2, **self.kb_kwargs)
        self.observation_space = kb.get_observation_space()
        self.observation_space['rewards'] = self.observation_space['events']
        logging.debug(f'self.observation_space.shape={self.observation_space.shape}')
        for k in self.observation_space:
            logging.debug(f"self.observation_space[{k}].shape={self.observation_space[k].shape}")

        self.mouse = None

    def _get_valid_tasks(self):
        task_dict = {
            task : {
                'is_valid': self.tasks[task].get('is_valid', lambda ram: True)(self.ram),
                'not terminated': not self.compute_for_task(task, 'terminated'),
                'not is_success': not self.compute_for_task(task, 'is_success'),
                }
            for task in self.tasks
            }
        valid_tasks = [task for task in task_dict if all(task_dict[task].values())]
        logging.debug(f"_get_valid_tasks: valid_tasks={valid_tasks}")
        if len(valid_tasks) == 0:
            logging.warning(f"_get_valid_tasks: valid_tasks=[]; task_dict={task_dict}")
        return valid_tasks

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
        self._text_infos = []

        obs, info = super().reset(**kwargs)
        kb_obs = self._generate_kb_obs()
        return kb_obs, info

    def _generate_kb_obs(self):
        kb = generate_knowledge_base(self.ram, self.ram2, **self.kb_kwargs)
        kb_obs = kb.to_observation()
        task = self.tasks[self.episode_task]
        kb_obs['rewards'] = np.array([task['reward'].get(k, 0) for k in sorted(kb.events)])
        return kb_obs

    def _set_episode_task(self, task):
        if task not in self.tasks:
            logging.warning(f'set_episode_task({task}); task not in tasks')
        elif task != self.episode_task:
            self.episode_task = task
            text = self.random.choice(self.task_to_text[task])
            self.register_text(text, speaker='Navi')

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
        self.skipped_frames += 1
        self._set_buttons(buttons)
        self.unwrapped.em.step()
        self.unwrapped.data.update_ram()
        self._update_ram()
        #ob = self.env._update_obs()
        if (self.env.render_mode == "human" 
                and not self.no_render_skipped_frames
                and not force_norender
                ):
            self.env.render()

    def compute_for_task(self, task, key):
        task_dict = self.tasks[task]
        return any([getattr(self.game_module, fname)(self.ram) for fname in task_dict[key]])

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
        kb = generate_knowledge_base(self.ram, self.ram2, **self.kb_kwargs)
        pseudoreward = 0
        kb_obs = kb.to_observation()
        task = self.tasks[self.episode_task]
        kb_obs['rewards'] = np.array([task['reward'].get(k, 0) for k in sorted(kb.events)])

        # register events
        for event in kb.events:
            if kb.events[event] != 0:
                texts = self.event_to_text[event]
                if len(texts) > 0:
                    self.register_text(texts[0])

        # run any task-specific operations
        if 'step' in task:
            task['step'](self, kb)

        # FIXME:
        if True:
            terminated = self.compute_for_task(self.episode_task, 'terminated')
            info['is_success'] = self.compute_for_task(self.episode_task, 'is_success')

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

        # move mouse
        if self.mouse is not None:
            if 'mode' not in self.mouse:
                logging.warning('"mode" not in self.mouse')

            elif self.mouse['mode'] in ['click', 'sprite']:
                # if a sprite is within this many pixels of a mouse click,
                # then self.mouse will automatically track the monster
                min_l1dist0 = 16

                # extract the list of sprites
                sprites = []
                for k in kb.items:
                    if 'mouse' not in k:
                        sprites.append(kb.items[k])

                # find the sprite closest to the mouse position,
                # and move self.mouse on top of the sprite
                oldmouse = kb.items['mouse']
                if 'x' not in oldmouse: oldmouse['x'] = 0
                if 'y' not in oldmouse: oldmouse['y'] = 0
                min_l1dist = min_l1dist0
                newmouse = None
                for sprite in sprites:
                    sprite_l1dist = abs(sprite['x'] - oldmouse['x']) + abs(sprite['y'] - oldmouse['y'])
                    if sprite_l1dist < min_l1dist:
                        min_l1dist = sprite_l1dist
                        newmouse = {
                            'x': sprite['x'],
                            'y': sprite['y'],
                            }
                if newmouse is not None:
                    newmouse['mode'] = 'sprite'
                    self.mouse = newmouse
                    self.ram.mouse = newmouse
                else:
                    self.mouse['mode'] = 'background'
                #logging.info(f"len(sprites)={len(sprites)}, min_l1dist={min_l1dist}, newmouse={newmouse}")

            elif self.mouse['mode'] == 'background':
                crop_size = 4
                cropped_screen2 = self.ram2.screen[crop_size:-crop_size,crop_size:-crop_size,:]
                scroll_amounts = [(i, j) for i in [2, 1, 0, -1, 2] for j in [2, 1, 0, -1, -2]]
                scores = []
                for i, j in scroll_amounts:
                    cropped_screen1 = self.ram.screen[crop_size+i:-crop_size+i, crop_size+j:-crop_size+j, :]
                    l1_dist = np.sum(np.abs(cropped_screen1 - cropped_screen2))
                    scores.append((l1_dist, (i, j)))
                scores.sort()
                max_score = max([score for score, coord in scores])
                scores = [(score/(max_score + 0.01), coord) for score, coord in scores]
                mouse_adjust = scores[0][1]
                #logging.debug(f"mouse_adjust={mouse_adjust}")
                if self.mouse is not None:
                    self.mouse['x'] += mouse_adjust[1]
                    self.mouse['y'] += mouse_adjust[0]
                #logging.debug(f"scores={' ; '.join([ f'{score:0.4f} {coord}; ' for score, coord in scores])}")
                #logging.debug(f"self.mouse, self.ram.mouse={self.mouse, self.ram.mouse}")
                if scores[0][0] > 0.25:
                    logging.warning('background mouse slippage likely')

        # render the environment
        if self.render_kb:

            if self.render_downsample:
                # set the screen's datatype to np.uint8 for rendering;
                # it is stored in kb as a float for training efficiency
                screen = kb.screen
                screen = screen.astype(self.unwrapped.img.dtype)

                # rescale the downsampled image back to its original size;
                # this is important so that the subsequent rendering code
                # places everything in the correct location
                screen_downsample = self.kb_kwargs['screen_downsample']
                screen = np.repeat(screen, screen_downsample, axis=0)
                screen = np.repeat(screen, screen_downsample, axis=1)
                self.unwrapped.img = screen

            # draw bounding box around objects
            for k, v in kb.items.items():
                # NOTE: we apply min/max to all values;
                # some objects can be located off screen,
                # and trying to render them directly crashes without clipping
                if 'sprite' in k:
                    # sprites are 8x16 or 8x8 pixels
                    xmin = max(0, min(239, v['x'] - v['width']//2))
                    xmax = max(0, min(239, v['x'] + v['width']//2))
                    ymin = max(0, min(223, v['y'] - v['height']//2))
                    ymax = max(0, min(223, v['y'] + v['height']//2))
                    self.unwrapped.img[ymin:ymax+1, xmin] = [255, 0, 0]
                    self.unwrapped.img[ymin:ymax+1, xmax] = [255, 0, 0]
                    self.unwrapped.img[ymin, xmin:xmax+1] = [255, 0, 0]
                    self.unwrapped.img[ymax, xmin:xmax+1] = [255, 0, 0]
                elif k != 'mouse':
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

        if self.render_task:
            # write the task to the screen
            # FIXME:
            # the code below is useful for debugging, but very slow
            # it takes about as much CPU as a single step of the emulator!
            img_pil = Image.fromarray(self.unwrapped.img)
            draw = ImageDraw.Draw(img_pil)
            draw.text(
                [0,0],
                self.episode_task,
                font=pil_font,
                fill=(255, 0, 255),
                antialiasing=False,
                )
            self.unwrapped.img = np.array(img_pil)

        if self.alt_screen:
            text = '\x1b[2J' # clears the screen
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

        self._run_register_text_callbacks()

        return kb_obs, reward + pseudoreward, terminated, truncated, info

    ########################################
    # MARK: functions for mouse manipulation
    ########################################

    def set_mouse(self, x, y):
        self.mouse = {'x': x, 'y': y, 'mode': 'click'}
        self.ram.mouse = self.mouse

    ########################################
    # MARK: functions for registering text callbacks
    ########################################

    def register_text(self, text, speaker='Link'):
        text_info = {
            'step': self.stepcount,
            'text': text,
            'lang': self.lang,
            'speaker': speaker,
            }
        #logging.info(f'register_text: {text_info}')
        self._text_infos.append(text_info)
        # FIXME:
        # we don't directly run the callbacks here
        # because register_text can be called from within a thread,
        # and this causes the tkinter callbacks to raise an exception;
        # the workaround is to store the text_info object
        # to be later added by the main thread when the step() function is called;
        # the current code isn't thread safe,
        # but it's not clear if thread safety is worth the cognitive + runtime overhead

    def _run_register_text_callbacks(self):
        for f in self._text_callbacks:
            for text_info in self._text_infos:
                f(text_info)
        self._text_infos = []

    def add_text_callback(self, f):
        self._text_callbacks.append(f)

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
            self.ram.ram[k] = v
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

    ########################################
    # MARK: save gamestate
    ########################################

    def save_state(self, statename):
        '''
        Create a file that stores the emulator's state.
        The file is gzip compressed so that it is compatible with being loaded directly in gym-retro.
        '''
        counter = 0
        while True:
            filename = f'custom_integrations/{self.env.gamename}/{statename}{str(counter).zfill(4)}.state'
            if not os.path.exists(filename):
                break
            counter += 1
        import gzip
        with gzip.open(filename, 'wb') as f:
            f.write(self.unwrapped.em.get_state())
        logging.info(f'save_state({statename}); filename={filename}')

    def load_state(self, statename):
        '''
        Load a state file.
        This is useful for debuging purposes to load in the middle of a game run.
        '''
        filename = f'custom_integrations/{self.env.gamename}/{statename}.state'
        import gzip
        with gzip.open(filename, 'rb') as f:
            self.unwrapped.em.set_state(f.read())
        logging.info(f'load_state({statename}); filename={filename}')


def parse_state(state):
    '''
    Takes the FCEUX save state as input (given from env.em.get_state()

    The output is all of the raw subsection chunk data.
    See <https://fceux.com/web/help/fcs.html> for details.
    '''
    bytes_to_int = lambda x: int.from_bytes(x, byteorder='little')

    # process 16 byte header
    assert state[:3] == b'FCS'
    assert state[3] == 0xff
    total_size = bytes_to_int(state[4:8])
    version = bytes_to_int(state[8:12])

    state = state[16:]
    assert len(state) == total_size

    ret = {}
    chunks = []
    while len(state) > 0:
        section = state[0]
        size = bytes_to_int(state[1:5])
        offset = size + 5
        chunk = state[:offset]
        chunk_data = chunk[5:]
        chunks.append(chunk)
        state = state[offset:]
        while len(chunk_data) > 0:
            subsection_name = chunk_data[:4].decode('utf-8').replace('\x00', '')
            subsection_size = bytes_to_int(chunk_data[4:8])
            subsection_data = chunk_data[8:8 + subsection_size]
            chunk_data = chunk_data[8 + subsection_size:]
            ret[subsection_name] = subsection_data

    return ret


def generate_knowledge_base(ram, ram2, **kwargs):
    kb = KnowledgeBase(observation_format={
        'objects_discrete': ['id'],
        'objects_discrete_id': ['id'],
        'objects_string_chunk': ['chunk_id'],
        'objects_continuous': ['relx', 'rely'],
        })

    # add the mouse object
    if ram.mouse is not None:
        item = {}
        item['x'] = ram.mouse['x']
        item['y'] = ram.mouse['y']
        item['id'] = 0
        item['chunk_id'] = 'common'
        kb['mouse'] = item

    # add a (potentially downsampled) screen
    w, h, c = ram.screen.shape
    screen_downsample = kwargs['screen_downsample']
    assert w % screen_downsample == 0
    assert h % screen_downsample == 0
    new_w = w // screen_downsample
    new_h = h // screen_downsample
    screen = ram.screen.reshape(new_w, screen_downsample, new_h, screen_downsample, c)
    screen = np.mean(screen, axis=(1, 3))
    kb.screen = screen

    # NOTE:
    # The NES uses the OAM region of memory $0200-$02FF for storing sprites.
    # The code below extracts sprites stored in this memory.
    # This serves as a generic object detector that works for all games.
    # Some "objects", however, are not stored as sprites but as background tiles.
    # These objects will not get detected here.
    #
    # See the following links for details on NES rendering/hardware:
    # <https://austinmorlan.com/posts/nes_rendering_overview/>
    # <https://www.copetti.org/writings/consoles/nes/>

    # the code below determines whether the NES is using 8 or 16 pixel tall sprites;
    # see <https://www.nesdev.org/wiki/PPU_registers#PPUCTRL> for details
    state = parse_state(ram.em_state)
    PPUCTRL = state['PPUR'][0]
    PPUSCROLL = state['XOFF'][0]
    if PPUCTRL & 32 == 0:
        sprite_height = 8
        sprite_width = 8
        sprite_yoffset = -4
    else:
        sprite_width = 8
        sprite_height = 16
        sprite_yoffset = 0

    # CHRR (character ram) is the region of ram that stores the sprites' graphics;
    # item['id'] stores the index into CHRR that is being used to draw the sprite,
    # item['chunk_id'] uniquely identifies the CHRR table being used;
    # in some games (e.g. Zelda), the CHRR is stored in the emulator state,
    # but in other games (e.g. Mario) it is not;
    # it's not clear to me why this is the case,
    # and the code below will have "hash conflicts" for games like Mario
    chrr = state.get('CHRR')
    if chrr is None:
        chrr_hash = '_NONE'
        if not hasattr(generate_knowledge_base, 'chrr_warning'):
            generate_knowledge_base.chrr_warning = True
            logging.warning('chrr is None')
    else:
        chrr_hash = format(xxhash.xxh32(chrr).intdigest() & 0xFFFFFFFF, '08x')
    chunk_id = kwargs['game_name'] + '_' + chrr_hash
    if not hasattr(generate_knowledge_base, 'chunk_ids'):
        generate_knowledge_base.chunk_ids = set()
    if chunk_id not in generate_knowledge_base.chunk_ids:
        logging.debug(f"generate_knowledge_base: chunk_id={chunk_id}")
        generate_knowledge_base.chunk_ids.add(chunk_id)

    # render the CHRR
    # (only if it has changed since the last frame)
    if kwargs.get('display_chrr'):
        if ram2 is not None:
            state2 = parse_state(ram2.em_state)
            if chrr is not None and chrr != state2['CHRR']:
                render_nes_pattern_table(chrr)
        else:
            if chrr is not None:
                render_nes_pattern_table(chrr)

    # loop over each sprite in the OAM
    for i in range(64):
        # The following link describes the details of the information being extracted here
        # <https://www.nesdev.org/wiki/PPU_OAM>
        base_addr = 0x0200 + 4*i
        item = {}
        item['chunk_id'] = chunk_id
        item['id'] = ram[base_addr + 1]
        item['x'] = ram[base_addr + 3] - sprite_width // 2
        item['y'] = ram[base_addr + 0] + sprite_yoffset
        item['width'] = sprite_width
        item['height'] = sprite_height

        byte2 = ram[base_addr + 2]
        item['pal4'] = int(byte2 & 0x3 == 0)
        item['pal5'] = int(byte2 & 0x3 == 1)
        item['pal6'] = int(byte2 & 0x3 == 2)
        item['pal7'] = int(byte2 & 0x3 == 3)
        item['priority']= int(byte2 & 32  >  0)
        item['flip_hori'] = int(byte2 & 64  >  0)
        item['flip_vert'] = int(byte2 & 128 >  0)

        # the NES has no flag for whether a sprite is visible or not;
        # sprites are made non-visible by moving the off the edge of the screen;
        # we only want to add sprites to the item list if they are visible
        add_sprite = False
        if item['y'] < 240 and item['x'] < 240:
            add_sprite = True

        # if the sprite is adjacent to another sprite in the image,
        # optionally choose to link the two sprites into a single object;
        # this can mildly improve inference runtime of downstream models
        # and make debugging output slightly cleaner;
        # in most games, the sprite that is being removed will be uniquely determined
        # by the sprite that it is adjacent to;
        # if this is the case, then the transformation should have no effect
        # on the models' ability to represent game states;
        # but if these adjacent sprites are not uniquely determined by each other,
        # than this transformation can cause different game states to be represented
        # with the same sprite information
        # FIXME:
        # this code joins some sprites but not all sprites;
        # in particular I have verified that the vertical code will not join
        # a chain of sprites together (in e.g. SuperMario),
        # and I suspect the horizontal code will not as well
        if add_sprite and kwargs.get('link_sprites'):

            # join sprites horizontally
            for olditem in kb.items.values():
                if item['x'] == olditem['x'] + olditem.get('width', -1000) and item['y'] == olditem['y'] and item.get('height') == olditem.get('height'):
                    olditem['width'] += item['width']
                    olditem['x'] += sprite_width // 2
                    add_sprite = False
                    break
                if item['x'] == olditem['x'] - olditem.get('width', -1000) and item['y'] == olditem['y'] and item.get('height') == olditem.get('height'):
                    olditem['width'] += item['width']
                    olditem['x'] -= sprite_width // 2
                    add_sprite = False
                    break

            # we have already linked a sprite horizontally;
            # now try to link it vertically
            if not add_sprite:
                for sprite_name, olditem2 in list(kb.items.items()):
                    if olditem['x'] == olditem2['x'] and olditem['width'] == olditem2['width'] and olditem['y'] == olditem2['y'] + olditem2.get('height', -1000):
                        olditem['height'] += olditem2['height']
                        olditem['y'] -= olditem2['height'] // 2
                        del kb.items[sprite_name]
                    if olditem['x'] == olditem2['x'] and olditem['width'] == olditem2['width'] and olditem['y'] == olditem2['y'] - olditem2.get('height', -1000):
                        olditem['height'] += olditem2['height']
                        olditem['y'] += olditem2['height'] // 2
                        del kb.items[sprite_name]

        # finally add the sprite
        if add_sprite:
            kb[f'sprite_{i:02}'] = item

    # generate background
    if kwargs.get('background_items'):
        assert kwargs['game_module']
        background_items, info = kwargs['game_module'].generate_knowledge_base_background_items(ram)
        kb.items |= background_items

    # center and normalize all items
    kb.columns.add('relx')
    kb.columns.add('rely')

    if kwargs.get('center_player'):
        assert kwargs['game_module']
        center_x = kwargs['game_module']._ramstate_player_x(ram)
        center_y = kwargs['game_module']._ramstate_player_y(ram)
    else:
        center_x = 120
        center_y = 120
    for item, val in kb.items.items():
        kb.items[item]['relx'] = kb.items[item]['x'] - center_x
        kb.items[item]['rely'] = kb.items[item]['y'] - center_y

        # normalize
        kb.items[item]['relx'] /= 120
        kb.items[item]['rely'] /= 120

    # add events
    kb.events = get_events(ram, ram2)
    if kwargs['game_module']:
        kb.events |= kwargs['game_module'].get_events(ram, ram2)

    return kb


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


################################################################################
# MARK: debug functions below
################################################################################

import matplotlib.pyplot as plt
import numpy as np

# Global variables to track state
_pattern_table_fig = None
_pattern_table_img = None

def render_nes_pattern_table(chrr, title="NES Pattern Table", use_8x16=True):
    """
    Render an NES pattern table to a window based on official NES encoding.

    Args:
        chrr: Byte array containing the NES pattern table data
        title: Title for the window
        use_8x16: If True, render as 8x16 sprites instead of 8x8 sprites
    """
    global _pattern_table_fig, _pattern_table_img

    # Calculate dimensions based on the data
    tiles_per_row = 16  # Standard NES pattern table layout is 16x16
    chrr_size = len(chrr)

    if use_8x16:
        tile_height = 16
        bytes_per_tile = 32  # 16 bytes per 8x8 tile × 2 tiles
    else:
        tile_height = 8
        bytes_per_tile = 16  # 16 bytes per 8x8 tile

    num_complete_tiles = chrr_size // bytes_per_tile
    rows = (num_complete_tiles + tiles_per_row - 1) // tiles_per_row  # Ceiling division

    # Create image data
    img_width = tiles_per_row * 8
    img_height = rows * tile_height
    img_data = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    # NES default colors for visualization
    nes_colors = [
        [0, 0, 0],          # Background/transparent
        [85, 85, 85],       # Color index 1
        [170, 170, 170],    # Color index 2
        [255, 255, 255]     # Color index 3
    ]

    # Process each tile
    for tile_idx in range(min(num_complete_tiles, tiles_per_row * rows)):
        if tile_idx * bytes_per_tile + bytes_per_tile - 1 >= chrr_size:
            break  # Safety check

        # Calculate tile position in the grid
        tile_x = tile_idx % tiles_per_row
        tile_y = tile_idx // tiles_per_row

        tile_offset = tile_idx * bytes_per_tile

        # Handle 8x8 or 8x16 tiles
        for sub_tile in range(1 if not use_8x16 else 2):
            sub_offset = sub_tile * 16  # 0 for top half, 16 for bottom half in 8x16 mode

            # Draw the 8x8 tile (or sub-tile for 8x16 mode)
            for y in range(8):
                # Get the two bit planes for this row
                plane0_byte = chrr[tile_offset + sub_offset + y]
                plane1_byte = chrr[tile_offset + sub_offset + y + 8]

                for x in range(8):
                    bit_pos = 7 - x
                    bit0 = (plane0_byte >> bit_pos) & 1
                    bit1 = (plane1_byte >> bit_pos) & 1
                    color_idx = (bit1 << 1) | bit0

                    # Set the pixel color
                    pixel_y = tile_y * tile_height + y + (sub_tile * 8)
                    pixel_x = tile_x * 8 + x

                    if pixel_y < img_height and pixel_x < img_width:
                        img_data[pixel_y, pixel_x] = nes_colors[color_idx]

    # The rest of the function remains the same
    if _pattern_table_fig is None or not plt.fignum_exists(_pattern_table_fig.number):
        plt.ion()
        _pattern_table_fig, ax = plt.subplots(figsize=(6, 6))
        _pattern_table_fig.canvas.manager.set_window_title(title)
        ax.set_title(title)
        ax.axis('off')
        _pattern_table_img = ax.imshow(img_data, interpolation='nearest')
        plt.tight_layout()
        plt.show(block=False)
    else:
        if _pattern_table_img.get_array().shape != img_data.shape:
            ax = _pattern_table_fig.axes[0]
            ax.clear()
            ax.axis('off')
            _pattern_table_img = ax.imshow(img_data, interpolation='nearest')
        else:
            _pattern_table_img.set_data(img_data)

        _pattern_table_fig.canvas.draw_idle()
        _pattern_table_fig.canvas.flush_events()
