#!/usr/bin/env python3

# set logging level
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # log level for current project
logging = logger

logging.debug('import stdlib')
import atexit
import numpy as np
import os
import pprint
import random
import sys
import time

logging.debug('import pyglet')
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pyglet
logging.debug('import pygame')
import pygame
logging.debug('import gymnasium')
import gymnasium
from  gymnasium.wrappers import *
logging.debug('import retro')
import retro

logging.debug('import .*')
from Mundus.Audio import *
from Mundus.Wrappers import *
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

import tkinter as tk
from PIL import Image, ImageTk
class ImageViewer:
    def __init__(self):
        self.width = 1080
        self.height = 600
        self._create_window()

    def _create_window(self):
        self.isopen = True
        self.window = tk.Tk()
        self.window.title('Zelda')
        
        # Prevent window resizing
        self.window.resizable(False, False)

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
        self.textbox.pack(fill=tk.BOTH, expand=True)
        self.textbox.config(state="disabled")

        self.textbox.config(state="normal")
        self.textbox.insert(tk.END, "This is blue text\n", "blue")
        self.textbox.tag_config("blue", foreground="blue")
        self.textbox.insert(tk.END, "This is red text\n", "red")
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

    def imshow(self, arr):
        self.image_frame.redraw(arr)
        self.image_frame.update()
        self.window.update()

    def register_text(self, text, color="blue"):
        tag_name = color
        self.textbox.config(state="normal")
        self.textbox.insert(tk.END, text + "\n", tag_name)
        if not self.textbox.tag_config(tag_name):
            self.textbox.tag_config(tag_name, foreground=color)
        self.textbox.config(state="disabled")
        self.window.update()

    def close(self):
        sys.exit(0)
        #self.window.destroy()
        self.isopen = False

    def __del__(self):
        self.close()


class Interactive(gymnasium.Wrapper):
    def __init__(self, env, maxfps=60):
        super().__init__(env)

        self.action_override = False
        self.maxfps0 = maxfps
        self.maxfps_multiplier = 0
        self.laststep = time.time()
        self.frames_since_log = 0
        self.last_log_time = time.time()
        self.last_log_reward = 0
        self.this_log_reward = 0
        self.log_every = 1.0
        self.total_reward = 0

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
        # self.render() is needed to create the self.unwrapped.viewer object;
        # we need this to interact with the pyglet window
        # and register the handlers
        self.unwrapped.viewer = ImageViewer()

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
            text = random.choice(['I will go there', 'Hi Evan', 'Attack!!!', 'Kill the monsters', 'I will do whatever you say', 'Your wish is my command'])
            self.register_text(text)

            x = event.x
            y = event.y
            if 'step' in zelda_env.tasks['onmouse_enemy']:
                del zelda_env.tasks['onmouse_enemy']['step']
            edge_size = 16

            # NOTE:
            # the values here are hardcoded for zelda;
            # 240 is the y resolution,
            # and 60 is the height of the black bar
            newx = int(x / self.unwrapped.viewer.image_width * 240)
            newy = 224 - int((self.unwrapped.viewer.image_height - y) / self.unwrapped.viewer.image_height * 224)
            if (zelda_env.ram.mouse is not None and
                'x' in zelda_env.ram.mouse and
                -edge_size <= zelda_env.ram.mouse['x'] - newx <= edge_size and
                -edge_size <= zelda_env.ram.mouse['y'] - newy <= edge_size ):
                    zelda_env.episode_task = 'attack'
                    zelda_env.mouse = None
            else:
                zelda_env.episode_task = 'onmouse_enemy'

                zelda_env.mouse = {}
                zelda_env.mouse['x'] = newx
                zelda_env.mouse['y'] = newy
                if zelda_env.mouse['x'] < edge_size:
                    zelda_env.mouse['x'] = -edge_size
                    zelda_env.episode_task = 'screen_west'
                if zelda_env.mouse['x'] > 240 - edge_size:
                    zelda_env.mouse['x'] = 240 + edge_size
                    zelda_env.episode_task = 'screen_east'
                if zelda_env.mouse['y'] < 60 + edge_size:
                    zelda_env.mouse['y'] = 60 - edge_size
                    zelda_env.episode_task = 'screen_north'
                if zelda_env.mouse['y'] > 224 - edge_size:
                    zelda_env.mouse['y'] = 224 + edge_size
                    zelda_env.episode_task = 'screen_south'
        self.unwrapped.viewer.window.bind("<Button-1>", on_mouse_press)

    def register_text(self, text):
        self.unwrapped.viewer.register_text(text)
        return self._recursive_register_text(self.env, text)

    def _recursive_register_text(self, env, text):
        if hasattr(env, 'register_text'):
            return env.register_text(text)
        if isinstance(env, gymnasium.Wrapper):
            return self._recursive_register_text(env.env, text)

    def reset(self, **kwargs):
        self.frames_since_log = 0
        self.last_log_time = time.time()
        self.last_log_reward = 0
        self.this_log_reward = 0
        self.this_log_sleeptime = 0
        self.log_every = 1.0
        self.total_reward = 0
        return super().reset(**kwargs)

    def get_maxfps(self):
        return self.maxfps0 * 2**(self.maxfps_multiplier)
        
    def step(self, action):
        
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
        observation, reward, terminated, truncated, info = super().step(action)
        results = observation, reward + manual_reward, terminated, truncated, info
        self.this_log_reward += reward
        self.total_reward += reward

        # sleep to adjust the fps
        steptime = time.time() - self.laststep
        sleeptime = 1.0/self.get_maxfps() - steptime
        self.this_log_sleeptime += sleeptime
        time.sleep(max(0, sleeptime))
        self.laststep = time.time()

        # report debugging info
        time_diff = time.time() - self.last_log_time
        self.frames_since_log += 1
        if time_diff >= self.log_every:
            logging.debug(f'fps={self.frames_since_log} reward_diff={self.this_log_reward:+0.4f} total_reward={self.total_reward:+0.4f} sleep_percent={self.this_log_sleeptime/self.log_every:0.4f}')
            self.last_log_time = time.time()
            self.frames_since_log = 0
            self.this_log_reward = 0
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
    parser.add_argument('--task_regex', default='attack')
    parser.add_argument('--state', default='spiders_lowhealth_01*.state')
    parser.add_argument('--model')
    parser.add_argument('--logfile', default='.play.log')
    parser.add_argument('--action_space', default='ALL')

    emulator_settings = parser.add_argument_group('emulator settings')
    emulator_settings.add_argument('--no_render_skipped_frames', action='store_true')
    emulator_settings.add_argument('--allframes', action='store_true')
    emulator_settings.add_argument('--no_alternate_screen', action='store_true')
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
            stdout_debug=not args.no_alternate_screen,
            debug_assert=True,
            no_render_skipped_frames=args.no_render_skipped_frames,
            skip_boring_frames=not args.allframes,
            task_regex=args.task_regex,
            reset_method='None',
            )

    if not args.noaudio:
        env = PlayAudio_pyaudio(env)
        env = PlayAudio_ElevenLabs(env)
    env = Interactive(env)

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

    # set the default action
    action = env.action_space.sample() * 0

    # switch the terminal to the alternate screen
    if not args.no_alternate_screen:
        print('\u001B[?1049h')

        # cause the program to exit the alternate screen when the program exits;
        # the use of the atexit library ensures that we switch
        # even when the program exits abnormally
        def exit_alternate_screen():
            print('\u001B[?1049l')
        atexit.register(exit_alternate_screen)

        # if python crashes, we want the exception printed to the main screen
        # and not the alternate screen
        def crash_handler(type, value, tb):
            env.close()
            print("\033[?1049l")  # exit alternate screen
            if type == KeyboardInterrupt:
                sys.exit(0)
            else:
                sys.__excepthook__(type, value, tb)
        sys.excepthook = crash_handler

    logging.info('begin main loop')
    env.reset()

    while True:
        # clear the screen
        if not args.no_alternate_screen:
            print('\x1b[2J', end='')

        # step the environment
        observation, reward, terminated, truncated, info = env.step(action)
        if args.doresets and (terminated or truncated):
            env.reset()

        # select the next action;
        if model is not None:
            action, _states = model.predict(observation, deterministic=True)
            
            # ensure that the model does not press start/select
            #action[2] = 0
            #action[3] = 0


if __name__ == "__main__":
    main()
