#!/usr/bin/env python3

# set logging level
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d [pid=%(process)d] %(levelname)s: %(message)s',
    #format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# use only 1 CPU for model inference
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMPY_NUM_THREADS'] = '1'

import setproctitle
setproctitle.setproctitle(f'serpens')

logging.debug('import stdlib')
import multiprocessing
import os
import pprint
import queue
import random
import sys
import signal
import time

logging.debug('import numpy')
import numpy as np

logging.debug('import pyglet')
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pyglet
logging.debug('import pygame')
import pygame
logging.debug('import gymnasium')
import gymnasium
logging.debug('import retro')
import retro

logging.debug('import .Audio')
from Mundus.Audio import *
logging.debug('import .Wrappers')
from Mundus.Wrappers import *
logging.debug('import .Zelda')
from Mundus.Zelda import *


# pyglet doesn't seem to have a map built-in for converting keycodes to names;
# this is my hack to make such a map
from pyglet.window import key as keycodes
from collections import defaultdict
keycode_map = defaultdict(lambda: [])
for name in dir(keycodes):
    keycode = getattr(keycodes, name)
    try:
        keycode_map[keycode].append(name)
    except TypeError:
        pass


logging.debug('import OpenGL')
from OpenGL import GL
'''
from pyopengltk import OpenGLFrame
class GLFrame(OpenGLFrame):
    def initgl(self):
        self.texture_id = GL.glGenTextures(1)
        GL.glEnable(GL.GL_TEXTURE_2D)

    def redraw(self, array=None):
        if array is not None:
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB, array.shape[1], array.shape[0],
                           0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, array)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)

        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glLoadIdentity()
        self._draw_texture()
        self.tkSwapBuffers()

    def _draw_texture(self):
        GL.glBegin(GL.GL_QUADS)
        GL.glTexCoord2f(0, 1); GL.glVertex2f(-1, -1)
        GL.glTexCoord2f(1, 1); GL.glVertex2f(1, -1)
        GL.glTexCoord2f(1, 0); GL.glVertex2f(1, 1)
        GL.glTexCoord2f(0, 0); GL.glVertex2f(-1, 1)
        GL.glEnd()
'''

import pygame
import tkinter as tk
from tkinter import Frame

class GLFrameOG(Frame):
    def __init__(self, master, width, height):
        super().__init__(master, width=width, height=height)
        os.environ['SDL_WINDOWID'] = '0'
        pygame.init()
        pygame.display.init()
        self.width = width
        self.height = height
        self.bind('<Map>', self._on_map)
        self.bind('<Map>', self._on_map)

        # Pre-create surface and array
        self.scaled_size = (width, height)
        self.screen = None
        self.scaled_surface = pygame.Surface(self.scaled_size)
        #pygame.display.set_hwsurface(True)
        #pygame.display.set_doublebuf(True)

    def _on_configure(self, event):
        # Update dimensions when frame is resized
        self.width = event.width
        self.height = event.height
        self.scaled_size = (self.width, self.height)
        if self.screen:
            self.screen = pygame.display.set_mode(self.scaled_size, pygame.HWACCEL | pygame.DOUBLEBUF)
            self.scaled_surface = pygame.Surface(self.scaled_size)

    def _on_map(self, event):
        os.environ['SDL_WINDOWID'] = str(self.winfo_id())
        self.screen = pygame.display.set_mode(self.scaled_size, pygame.HWACCEL | pygame.DOUBLEBUF)

    def redraw(self, array=None):
        if array is not None and self.screen:
            # Convert array directly to surface using buffer protocol
            surface = pygame.surfarray.make_surface(array.swapaxes(0,1))
            pygame.transform.scale(surface, self.scaled_size, self.screen)
            pygame.display.flip()


class GLFrame(Frame):
    def __init__(self, master, width, height):
        super().__init__(master, width=width, height=height)
        os.environ['SDL_WINDOWID'] = '0'
        pygame.init()
        pygame.display.init()
        self.width = width
        self.height = height
        self.bind('<Map>', self._on_map)
        self.bind('<Configure>', self._on_configure)
        self.bind('<Visibility>', self._on_visibility)

        self.scaled_size = (width, height)
        self.screen = None
        self.scaled_surface = pygame.Surface(self.scaled_size)
        array_size = (240, 224)
        self.array_surface = pygame.Surface(array_size, pygame.HWSURFACE)

    def _on_visibility(self, event):
        if self.screen:
            # Force realignment when window becomes visible
            self.update_pygame_window()

    def update_pygame_window(self):
        os.environ['SDL_WINDOWID'] = str(self.winfo_id())
        self.screen = pygame.display.set_mode(self.scaled_size, pygame.HWACCEL | pygame.DOUBLEBUF | pygame.SCALED)
        self.scaled_surface = pygame.Surface(self.scaled_size)

    def _on_configure(self, event):
        self.width = event.width
        self.height = event.height
        self.scaled_size = (self.width, self.height)
        if self.screen:
            self.update_pygame_window()

    def _on_map(self, event):
        self.update_pygame_window()

    def redraw(self, array=None):
        if array is not None and self.screen:
            pygame.surfarray.blit_array(self.array_surface, array.swapaxes(0,1))
            pygame.transform.scale(self.array_surface, self.scaled_size, self.screen)
            pygame.display.flip()


import tkinter as tk
from PIL import Image, ImageTk
class ImageViewer:
    def __init__(self):
        self.width = 1080
        self.height = 600
        #self.width = 400
        #self.height = 240
        self._create_window()

    def _create_window(self):
        self.window = tk.Tk()
        self.window.title('Zelda')

        # NOTE:
        # the RetroEnv class expects this variable to exist,
        # but we don't actually use it at all
        self.isopen = True

        # Prevent window resizing
        self.window.resizable(False, False)

        # Add window close handler
        def on_closing():
            self.window.destroy()
            sys.exit(0)
            #os._exit(0)
        self.window.protocol("WM_DELETE_WINDOW", on_closing)

        # Center the window and set exact size
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        x = (screen_width - self.width) // 2
        y = (screen_height - self.height) // 2
        self.window.geometry(f"{self.width}x{self.height}+{x}+{y}")

        # Calculate new image dimensions maintaining aspect ratio
        original_ratio = 256 / 240  # Original NES aspect ratio
        self.image_height = self.height
        self.image_width = int(self.image_height * original_ratio)

        # Create and configure frames
        self.image_frame = GLFrame(self.window, width=self.image_width, height=self.image_height)
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH)
        self.image_frame.pack_propagate(False)

        textbox_frame = tk.Frame(self.window)
        textbox_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create and configure textbox
        self.textbox = tk.Text(textbox_frame)
        self.scrollbar = tk.Scrollbar(textbox_frame)
        self.textbox.config(yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.textbox.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.textbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.textbox.tag_config("black", foreground="black")
        self.textbox.tag_config("blue", foreground="blue")
        self.textbox.tag_config("red", foreground="red")
        self.textbox.config(state="disabled")

        def on_resize(event):
            self.width = self.window.winfo_width()
            self.height = self.window.winfo_height()
            self.image_height = self.height
            original_ratio = 256 / 240  # Original NES aspect ratio
            self.image_width = int(self.image_height * original_ratio)
            self.image_frame.config(width=self.image_width, height=self.image_height)
        self.window.bind('<Configure>', on_resize)
        #self.window.attributes("-fullscreen", True)
        self.window.update_idletasks()

        self.framecount = 0

    def imshow(self, arr):
        self.framecount += 1
        # FIXME:
        # this is a dirty hack to speed up rendering by rendering
        # only some frames
        if self.framecount %3 == 0:
            self.image_frame.redraw(arr)
            self.image_frame.update()
            self.window.update()

    def register_text(self, text_info, color="blue"):
        # calculate elapsed time
        if not hasattr(self, 'first_text_time'):
            self.first_text_time = time.time()
        elapsed_time = time.time() - self.first_text_time

        # format elapsed time
        minutes, seconds = divmod(elapsed_time, 60)
        hours, minutes = divmod(minutes, 60)
        timestamp = f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"

        # insert text
        self.textbox.config(state="normal")
        self.textbox.insert(tk.END, timestamp + " ", 'black')
        self.textbox.insert(tk.END, text_info['text'] + "\n", color)
        if not self.textbox.tag_config(color):
            self.textbox.tag_config(color, foreground=color)
        self.textbox.yview(tk.END)
        self.textbox.config(state="disabled")
        self.window.update()


class Interactive(gymnasium.Wrapper):
    def __init__(self, env, maxfps=60):
        super().__init__(env)

        self.action_override = False
        self.maxfps0 = maxfps
        self.maxfps_multiplier = 0
        self.log_every = 1.0

        # initialize pygame joysticks
        # NOTE:
        # the only thing we use pygame for is access to the joystick;
        # stable-retro only uses pyglet, but gymnasium uses pygame;
        # since we're using the window created by stable-retro,
        # all of the non-joystick code is based on pyglet;
        # it might a bit nicer to move the joystick code over to pyglet as well
        pygame.init()
        self.joysticks = {}

        # NOTE:
        # The RetroEnv class comes with a built-in object for rendering images.
        # It is the SimpleImageViewer class and created automatically.
        # Here, we overwrite that with our own ImageViewer defined above.
        # (To allow for inputs and other outputs besides just the screen.)
        self.unwrapped.viewer = ImageViewer()
        self.unwrapped.RetroKB.add_text_callback(lambda text: self.unwrapped.viewer.register_text(text))

        # setup the key handler
        self._key_previous_states = {}
        self._key_handler = pyglet.window.key.KeyStateHandler()
        #self.unwrapped.viewer.window.push_handlers(self._key_handler)

        # NOTE:
        # the mouse handler requires direct access to ZeldaWrapper
        # (and not e.g. a self.env that has additional Wrappers applied later)
        # so we first extract the zelda_env, then register the mouse handler
        zelda_env = self.env
        while not isinstance(zelda_env, ZeldaWrapper):
            assert zelda_env.env is not None
            zelda_env = zelda_env.env

        def on_mouse_press(event):
            x = event.x
            y = event.y
            edge_size = 16

            # NOTE:
            # the values here are hardcoded for zelda;
            # 240 is the y resolution,
            # and 60 is the height of the black bar
            newx = int(x / self.unwrapped.viewer.image_width * 240)
            newy = 224 - int((self.unwrapped.viewer.image_height - y) / self.unwrapped.viewer.image_height * 224)
            zelda_env.mouse = {}
            zelda_env.mouse['x'] = newx
            zelda_env.mouse['y'] = newy
            if zelda_env.mouse['x'] < edge_size:
                zelda_env.mouse['x'] = -edge_size
                zelda_env._set_episode_task('screen_west')
            elif zelda_env.mouse['x'] > 240 - edge_size:
                zelda_env.mouse['x'] = 240 + edge_size
                zelda_env._set_episode_task('screen_east')
            elif zelda_env.mouse['y'] < 60 + edge_size - 16:
                zelda_env.mouse['y'] = 60 - edge_size
                zelda_env._set_episode_task('screen_north')
            elif zelda_env.mouse['y'] > 224 - edge_size:
                zelda_env.mouse['y'] = 224 + edge_size
                zelda_env._set_episode_task('screen_south')
            else:
                zelda_env._set_episode_task('interactive_onmouse')

        self.unwrapped.viewer.image_frame.bind("<Button-1>", on_mouse_press)

    def reset(self, **kwargs):
        self.frames_since_log = 0
        self.last_log_time = time.time()
        self.last_log_reward = 0
        self.this_log_sleeptime = 0
        self.this_log_frames = 0
        self.laststep_time = time.time()
        return super().reset(**kwargs)

    def get_maxfps(self):
        return self.maxfps0 * 2**(self.maxfps_multiplier)
        
    def step(self, action):
        
        """
        # register any pygame events
        for event in pygame.event.get():
            if event.type == pygame.JOYDEVICEADDED:
                joy = pygame.joystick.Joystick(event.device_index)
                self.joysticks[joy.get_instance_id()] = joy
                logging.info(f"Joystick {joy.get_instance_id()} connencted")
            if event.type == pygame.JOYDEVICEREMOVED:
                del self.joysticks[event.instance_id]
                logging.info(f"Joystick {event.instance_id} disconnected")

        # convert joystick actions into equivalent keyboard actions
        joystick_keycodes = {}
        for joystick in self.joysticks.values():
            if joystick.get_axis(0) > 0.1:
                joystick_keycodes[keycodes.MOTION_RIGHT] = True
            if joystick.get_axis(0) < -0.1:
                joystick_keycodes[keycodes.MOTION_LEFT] = True
            if joystick.get_axis(1) > 0.1:
                joystick_keycodes[keycodes.MOTION_DOWN] = True
            if joystick.get_axis(1) < -0.1:
                joystick_keycodes[keycodes.MOTION_UP] = True
            button_map = {
                    0: keycodes.X, # X
                    1: keycodes.Z, # A
                    2: keycodes.Z, # B
                    3: keycodes.X, # Y
                    4: keycodes.Q, # L
                    5: keycodes.Q, # R
                    8: keycodes.TAB, # SELECT
                    9: keycodes.ENTER, # START
                    }
            for button, keycode in button_map.items():
                if joystick.get_button(button):
                    joystick_keycodes[keycode] = True
            #for i in range(joystick.get_numbuttons()):
                #print(f"i, joystick.get_button(i)={i, joystick.get_button(i)}")

        # compute which keys are pressed
        keys_pressed = []
        keys_pressed_this_frame = []
        prev_action_override = self.action_override
        for keycode, pressed in (self._key_handler | joystick_keycodes).items():
            if pressed:
                self.action_override = True
                keynames = keycode_map[keycode]
                #print(f"keycode, keynames={keycode, keynames}")
                keys_pressed.extend(keycode_map[keycode])
                if not self._key_previous_states.get(keycode):
                    keys_pressed_this_frame.extend(keynames)
            self._key_previous_states[keycode] = pressed

        # handle meta-actions that affect the emulator environment
        manual_reward = 0
        lang_input = None
        for name in keys_pressed_this_frame:
            if name == 'BRACKETLEFT':
                self.maxfps_multiplier += 1
                logging.debug(f"self.get_maxfps()={self.get_maxfps()}")

            if name == 'BRACKETRIGHT':
                self.maxfps_multiplier -= 1
                logging.debug(f"self.get_maxfps()={self.get_maxfps()}")

            if name == 'SPACE':
                # NOTE:
                # See the "MARK: debug code" section of ZeldaWrapper
                # for useful functions to run once in the repl.
                import readline
                import code
                code.interact(local=locals())
                self.action_override = prev_action_override

        for name in keys_pressed:
            if name == 'MINUS':
                manual_reward -= 1
                self.action_override = prev_action_override
                logging.debug(f"manual_reward={manual_reward}")

            if name == 'EQUAL':
                manual_reward += 1
                self.action_override = prev_action_override
                logging.debug(f"manual_reward={manual_reward}")

        # perform the action
        # FIXME:
        # this code only works for the ALL action space
        if self._key_handler.get(keycodes.ESCAPE):
            self.action_override = False
        if self.action_override:
            action = self.keys_to_act(keys_pressed)
        """
        observation, reward, terminated, truncated, info = super().step(action)
        results = observation, reward, terminated, truncated, info

        # sleep to adjust the fps
        steptime = time.time() - self.laststep_time
        sleeptime = max(0, 1.0/self.get_maxfps() - steptime)
        self.this_log_sleeptime += sleeptime
        time.sleep(sleeptime)
        self.laststep_time = time.time()

        # report debugging info
        self.frames_since_log += 1
        time_since_log = time.time() - self.last_log_time
        if time_since_log >= self.log_every:
            logging.debug(f'fps={self.frames_since_log/self.log_every:0.1f} sleep_fraction={self.this_log_sleeptime/self.log_every:0.4f}')
            self.last_log_time = time.time()
            self.frames_since_log = 0
            self.this_log_sleeptime = 0

        return results

    def keys_to_act(self, keys):
        inputs = {
            None: False,
            "BUTTON": "Z" in keys,
            "A": "Z" in keys,
            "B": "X" in keys,
            "C": "C" in keys,
            "X": "A" in keys,
            "Y": "S" in keys,
            "Z": "D" in keys,
            "L": "Q" in keys,
            "R": "W" in keys,
            "UP": "UP" in keys,
            "DOWN": "DOWN" in keys,
            "LEFT": "LEFT" in keys,
            "RIGHT": "RIGHT" in keys,
            "MODE": "TAB" in keys,
            "SELECT": "TAB" in keys,
            "RESET": "ENTER" in keys,
            "START": "ENTER" in keys,
            "PAUSE": "ENTER" in keys,
        }
        pushed_buttons = [b for b in self.unwrapped.buttons if inputs[b]]
        #print(f"pushed_buttons={pushed_buttons}")
        return [inputs[b] for b in self.unwrapped.buttons]


def main():
    # parse command line args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', default='es')
    parser.add_argument('--task_regex', default='attack')
    parser.add_argument('--state', default='overworld_07')
    parser.add_argument('--model', default='models/model.zip')
    parser.add_argument('--logfile', default='.play.log')
    parser.add_argument('--action_space', default='zelda-all')

    emulator_settings = parser.add_argument_group('emulator settings')
    emulator_settings.add_argument('--no_render_skipped_frames', action='store_true')
    emulator_settings.add_argument('--allframes', action='store_true')
    emulator_settings.add_argument('--alt_screen', action='store_true')
    emulator_settings.add_argument('--noaudio', action='store_true')
    emulator_settings.add_argument('--doresets', action='store_true')

    args = parser.parse_args()

    # convert warnings to errors
    import warnings
    warnings.filterwarnings('always', append=True)
    warnings.simplefilter('error')
    # FIXME:
    # there is a tempfile getting created somewhere that doesn't get deleted;
    # this filter cleans up the warning
    warnings.filterwarnings('ignore', category=ResourceWarning)

    # create the environment
    logging.info('creating environment')
    env = make_zelda_env(
            # FIXME:
            # the action_space should be loaded automatically from the model
            action_space=args.action_space,
            alt_screen=args.alt_screen,
            debug_assert=True,
            state=args.state,
            no_render_skipped_frames=args.no_render_skipped_frames,
            skip_boring_frames=not args.allframes,
            task_regex=args.task_regex,
            reset_method='None',
            fork_emulator=True,
            lang=args.lang,
            )

    if not args.noaudio:
        env = PlayAudio_pyaudio(env)
        env = PlayAudio_ElevenLabs(env)
    env = Interactive(env)


    logging.info('begin main loop')
    env.reset()

    # qin stores messages that go to the worker;
    # qout stores messages that come from the worker;
    # they are of size 1 to ensure syncronization between the worker and parent process;
    # if one process gets ahead of the other process,
    # it will block trying to add a second entry into the queue
    qin = multiprocessing.Queue(maxsize=1)
    qout = multiprocessing.Queue(maxsize=1)
    qout_lock = multiprocessing.Lock()

    # start the emulator process
    def model_worker():
        current_name = multiprocessing.current_process().name
        setproctitle.setproctitle(f'{current_name}: model_worker')

        # NOTE:
        # loading the models can cause a large number of warnings;
        # this with block prevents those warnings from being displayed
        logging.info('loading model')
        model = None
        if args.model:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from stable_baselines3 import PPO
                custom_objects = {
                    'observation_space': env.observation_space,
                    'action_space': env.action_space,
                    }
                model = PPO.load(args.model, custom_objects=custom_objects)

        # set the first action to be empty;
        # we must do this after the model is loaded
        # so that the main loop doesn't start playing before the model is loaded
        action = env.action_space.sample() * 0
        qout.put(action)

        # set this process to have the lowest priority;
        # the user won't notice if this process lags,
        # but they will notice if this process interrupts the emulator or main processes
        os.nice(19)

        # the worker will loop forever;
        # it blocks on the qin.get() line waiting for a new observation;
        # then it puts the appropriate action in the queue
        logging.info('entering model_worker main loop')
        while True:
            observation = qin.get()
            if model is not None:
                action, _states = model.predict(observation, deterministic=True)
            else:
                action = env.action_space.sample() * 0
            with qout_lock:
                qout.get()
                qout.put(action)
            # wait before selecting another action;
            # in the meantime, the same action will remain in qout,
            # and so the main loop will just repeat the action;
            # this should have minimal adverse effects on model performance;
            # the purpose of waiting is to reduce CPU load for low-resource devices;
            time.sleep(1/60)
    p = Process(name='serpens: model_worker', target=model_worker)
    p.start()

    while True:
        # get the next action from the model
        # (which is the action from the previous iteration's observation)
        with qout_lock:
            action = qout.get()
            qout.put(action)

        # step the environment
        observation, reward, terminated, truncated, info = env.step(action)
        if args.doresets and (terminated or truncated):
            env.reset()

        # send the observation to the model_worker process
        # if there is already an observation,
        # that means the model is running slowly and falling behind the main process;
        # this happens when the model is larger than the hardware can run at 60fps;
        # we overwrite the existing observation with the latest observation
        # so that the model will always make decision on the latest observation
        # even if it has to skip frames
        try:
            qin.get_nowait()
        except queue.Empty:
            pass
        qin.put(observation)


if __name__ == "__main__":
    main()
