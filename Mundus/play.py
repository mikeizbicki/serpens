#!/usr/bin/env python3

# set logging level
import logging

COLORS = {
    'DEBUG': '\033[90m',  # grey
    #'INFO': '\033[36m',  # cyan
    'WARNING': '\033[91m',  # red
    'ERROR': '\033[91m\033[43m',  # red on yellow
    'CRITICAL': '\033[91m\033[43m'  # red on yellow
}

class DuplicateFilter:
    def __init__(self):
        self.msg_dict = {}

    def filter(self, record):
        import time
        current_time = time.time()
        if record.msg in self.msg_dict:
            if current_time - self.msg_dict[record.msg] < 5:
                return False
        self.msg_dict[record.msg] = current_time
        return True

class CustomFormatter(logging.Formatter):
    def format(self, record):
        levelname = record.levelname
        if levelname in COLORS:
            return COLORS[levelname] + super().format(record) + '\033[0m'
        else:
            return super().format(record)

logger = logging.getLogger()
logger.addFilter(DuplicateFilter())
handler = logging.StreamHandler()
handler.setFormatter(CustomFormatter(
    '%(asctime)s.%(msecs)03d %(name)-10s [pid=%(process)d] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

logging.getLogger('groq').setLevel(logging.WARNING)
logging.getLogger('httpcore.connection').setLevel(logging.WARNING)
logging.getLogger('httpcore.core').setLevel(logging.WARNING)
logging.getLogger('httpcore.http11').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

logger_fps = logging.getLogger('root/fps')
logger_fps.setLevel(logging.INFO)

# use only 1 CPU for model inference
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMPY_NUM_THREADS'] = '1'

import setproctitle
setproctitle.setproctitle(f'serpens')

logging.debug('import stdlib')
import os
import pprint
import random
import sys
import signal
import time
import warnings

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
logging.debug('import tkinter')
import tkinter as tk

logging.debug('import Mundus.Wrappers')
from Mundus.Wrappers import *
logging.debug('import Mundus.Zelda')
from Mundus.Games import *
from Mundus.Agent.Zelda import *
logging.debug('import Mundus.Play.Audio')
from Mundus.Play.Audio import *
logging.debug('import Mundus.Play.ModelHandler')
from Mundus.Play.ModelHandler import *

class Pygame_frame(tk.Frame):
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


class ImageViewer:
    def __init__(self, env, fullscreen=True):
        self.width = 1080
        self.height = 600
        self._create_window(fullscreen=fullscreen)

        self.env = env
        self.Agent = find_Wrapper(self, Agent)

    def _create_window(self, fullscreen):
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
        self.image_frame = Pygame_frame(self.window, width=self.image_width, height=self.image_height)
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH)
        self.image_frame.pack_propagate(False)

        textbox_frame = tk.Frame(self.window)
        textbox_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Add buttons above textbox
        button_frame = tk.Frame(textbox_frame)
        button_frame.pack(side=tk.TOP)

        self.button_task = tk.Button(button_frame, text="New Task", command=self.button_task_callback)
        self.button_task.pack(side=tk.LEFT, pady=5)

        self.button_objective = tk.Button(button_frame, text="New Objective", command=self.button_objective_callback)
        self.button_objective.pack(side=tk.LEFT, pady=5)

        # Create and configure textbox
        self.textbox = tk.Text(textbox_frame)
        self.scrollbar = tk.Scrollbar(textbox_frame)
        self.textbox.config(yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.textbox.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.textbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.textbox.tag_config("black", foreground="black")
        self.textbox.tag_config("gray", foreground="gray")
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
        if fullscreen:
            self.window.attributes("-fullscreen", True)
        self.window.update_idletasks()

        self.framecount = 0

    def button_objective_callback(self):
        if self.Agent:
            self.Agent.generate_objective()
        else:
            logger.debug('no self.Agent')

    def button_task_callback(self):
        if self.Agent:
            self.Agent.generate_newtask()
        else:
            logger.debug('no self.Agent')

    def imshow(self, arr):
        self.framecount += 1
        # FIXME:
        # this is a dirty hack to speed up rendering by rendering
        # only some frames
        if self.framecount %3 == 0:
            self.image_frame.redraw(arr)
            self.image_frame.update()
            self.window.update()

    def register_text(self, text_info):
        # format elapsed time
        elapsed_time = text_info['step'] / 60
        minutes, seconds = divmod(elapsed_time, 60)
        hours, minutes = divmod(minutes, 60)
        timestamp = f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"

        # insert text
        self.textbox.config(state="normal")
        self.textbox.insert(tk.END, timestamp + " ", 'black')
        if text_info['speaker'] == 'Link':
            color = 'blue'
        elif text_info['speaker'] == 'Navi':
            color = 'red'
        else:
            color = 'gray'
        self.textbox.insert(tk.END, f"({text_info['speaker']}) {text_info['text']}\n", color)
        if not self.textbox.tag_config(color):
            self.textbox.tag_config(color, foreground=color)
        self.textbox.yview(tk.END)
        self.textbox.config(state="disabled")
        self.window.update()


class Interactive(gymnasium.Wrapper):
    def __init__(self, env, windowed, maxfps=60):
        super().__init__(env)

        self.action_override = False
        self.maxfps0 = maxfps
        self.maxfps_multiplier = 0
        self.log_every = 1.0

        # NOTE:
        # The RetroEnv class comes with a built-in object for rendering images.
        # It is the SimpleImageViewer class and created automatically.
        # Here, we overwrite that with our own ImageViewer defined above.
        # (To allow for inputs and other outputs besides just the screen.)
        self.unwrapped.viewer = ImageViewer(env=self, fullscreen=not windowed)
        self.unwrapped.RetroKB.add_text_callback(lambda text: self.unwrapped.viewer.register_text(text))

        # NOTE:
        # the mouse handler requires direct access to RetroKB
        # (and not e.g. a self.env that has additional Wrappers applied later)
        # so we first extract the retrokb, then register the mouse handler
        retrokb = find_Wrapper(self, RetroKB)

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
            #retrokb.mouse = {}
            #retrokb.mouse['x'] = newx
            #retrokb.mouse['y'] = newy
            if newx < edge_size:
                newx = -edge_size
                retrokb._set_episode_task('screen_west')
            elif newx > 240 - edge_size:
                newx = 240 + edge_size
                retrokb._set_episode_task('screen_east')
            elif newy < 60 + edge_size - 16:
                newy = 60 - edge_size
                retrokb._set_episode_task('screen_north')
            elif newy > 224 - edge_size:
                newy = 224 + edge_size
                retrokb._set_episode_task('screen_south')
            else:
                retrokb._set_episode_task('interactive_onmouse')
            retrokb.set_mouse(newx, newy)

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
            logger_fps.debug(f'fps={self.frames_since_log/self.log_every:0.1f} sleep_fraction={self.this_log_sleeptime/self.log_every:0.4f}')
            self.last_log_time = time.time()
            self.frames_since_log = 0
            self.this_log_sleeptime = 0

        return results


class Keyboard(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.logger = logging.getLogger('Keyboard')
        self.logger.setLevel(logging.DEBUG)
        #self.logger.setLevel(logging.INFO)

        self.logger.info("Keyboard wrapper initialized")
        self.logger.debug(f"self.unwrapped.buttons={self.unwrapped.buttons}")

        self.image_frame = self.unwrapped.viewer.image_frame
        self.window = self.unwrapped.viewer.window

        # NOTE:
        # the code below handles the keyboard events;
        # the attribute self.controller_buttons should always contain the set of buttons currently pressed down on the controller
        self.controller_buttons = set()
        self.keyboard_to_controller = {
            'Left': 'LEFT',
            'Right': 'RIGHT',
            'Up': 'UP',
            'Down': 'DOWN',
            'z': 'A',
            'x': 'B',
            'a': 'START',
            's': 'SELECT',
            }

        def _on_key_press(event):
            controller_button = self.keyboard_to_controller.get(event.keysym)
            self.controller_buttons.add(controller_button)
            self.logger.debug(f"Key pressed tkinter: keyboard={event.keysym}, controller={controller_button}")

            if event.keysym == 'space':
                retrokb = find_Wrapper(self, RetroKB)
                import code; code.interact(local=locals())

            if event.keysym == 'Escape':
                retrokb = find_Wrapper(self, RetroKB)
                retrokb.save_state('manualsave')

        def _on_key_release(event):
            controller_button = self.keyboard_to_controller.get(event.keysym)
            try:
                self.controller_buttons.remove(controller_button)
            except KeyError:
                self.logger.warning(f'KeyError: self.controller_buttons.remove({event.keysym})')
            self.logger.debug(f"Key released tkinter: keyboard={event.keysym}, controller={controller_button}")

        self.image_frame.bind("<KeyPress>", _on_key_press)
        self.image_frame.bind("<KeyRelease>", _on_key_release)

        # NOTE:
        # the code below ensures that the self.image_frame always maintains proper focus
        # so that the keyboard events get received
        def _on_focus_in(event):
            self.logger.debug("Window gained focus")
            self.image_frame.focus_set()
            self.image_frame.focus_force()

        def _on_focus_out(event):
            self.logger.debug("Window lost focus")

        self.window.bind("<FocusIn>", _on_focus_in)
        self.window.bind("<FocusOut>", _on_focus_out)
        self.image_frame.focus_set()
        self.image_frame.focus_force()

    def step(self, action):

        # FIXME:
        # this formula for computing the action only works for the ALL action space
        action = [button in self.controller_buttons for button in self.unwrapped.buttons]
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, truncated, info


def main():
    # parse command line args
    import argparse
    parser = argparse.ArgumentParser()

    def boolean_flag(value):
        if value.lower() == 'true':
            return True
        elif value.lower() == 'false':
            return False
        else:
            raise argparse.ArgumentTypeError('Invalid boolean value')

    group = parser.add_argument_group('debug options')
    group.add_argument('--logfile', default='.play.log')
    group.add_argument('--fork_emulator', type=boolean_flag, default=True)
    group.add_argument('--fork_model', type=boolean_flag, default=True)
    group.add_argument('--fork_model_sleep', type=float, default=0.0)

    group = parser.add_argument_group('settings: interface')
    group.add_argument('--lang', default='es')
    group.add_argument('--windowed', action='store_true')

    group = parser.add_argument_group('settings: emulator')
    group.add_argument('--game', default='Zelda')
    group.add_argument('--no_render_skipped_frames', action='store_true')
    group.add_argument('--allframes', action='store_true')
    group.add_argument('--alt_screen', action='store_true')
    group.add_argument('--noaudio', action='store_true')
    group.add_argument('--doresets', action='store_true')

    # FIXME: these should all be loaded from the model file
    group = parser.add_argument_group('hyperparameters: architecture')
    group.add_argument('--action_space', default='zelda-all')
    group.add_argument('--model', default='models/model.zip')

    group = parser.add_argument_group('hyperparameters: environment')
    group.add_argument('--reset_method', default='None')
    group.add_argument('--reset_state', default=None, type=str)
    group.add_argument('--task_regex', default='attack')

    args = parser.parse_args()

    # convert warnings to errors
    import warnings
    warnings.filterwarnings('always', append=True)
    warnings.simplefilter('error')
    # FIXME:
    # there is a tempfile getting created somewhere that doesn't get deleted sometimes;
    # this filter cleans up the warning
    warnings.filterwarnings('ignore', category=ResourceWarning)

    # create the environment
    logging.info('creating environment')
    env = make_game_env(
            # FIXME:
            # the action_space should be loaded automatically from the model
            game=args.game,
            action_space=args.action_space,
            alt_screen=args.alt_screen,
            debug_assert=True,
            no_render_skipped_frames=args.no_render_skipped_frames,
            skip_boring_frames=not args.allframes,
            task_regex=args.task_regex,
            reset_method=args.reset_method,
            reset_state=args.reset_state,
            fork_emulator=args.fork_emulator,
            interactive=True,
            lang=args.lang,
            )

    if not args.noaudio:
        env = PlayAudio_pyaudio(env)
        env = PlayAudio_ElevenLabs(env)
    # FIXME:
    # in order to go into an interactive code session after creating the Interactive environment,
    # we need to have been in an interactive code session beforehand
    #import code; code.interact(local=locals())
    env = Interactive(env, args.windowed or args.alt_screen)
    if args.action_space in ['all']:
        env = Keyboard(env)
    observation, info = env.reset()

    # FIXME:
    # creating the model_handler is slow because it needs to load the model from disk;
    # in principle, this could be parallelized with the make_zelda_env command,
    # but I need to find a way to remove the env dependency,
    # which seems possible but inconvenient
    if args.fork_model:
        model_handler = ModelHandlerForked(args.model, env, args.fork_model_sleep)
    else:
        model_handler = ModelHandler(args.model, env)

    logging.info('begin main loop')
    while True:
        action = model_handler.get_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        if args.doresets and (terminated or truncated):
            env.reset()

if __name__ == "__main__":
    main()
