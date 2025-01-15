import time
import numpy as np
import random
import os
import logging
import sys
import pprint
from queue import Queue

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pyglet
import pygame
import gymnasium
from  gymnasium.wrappers import *
import retro

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
        self.render() 

        # setup the key handler
        self._key_previous_states = {}
        self._key_handler = pyglet.window.key.KeyStateHandler()
        self.unwrapped.viewer.window.push_handlers(self._key_handler)

        # setup the mouse handler
        # FIXME:
        # this is a janky setup to set the mouse coordinates in the Zelda environment;
        # it assumes that this Wrapper is called directly on the Zelda
        def on_mouse_motion(x, y, dx, dy):
            self.env.mouse = {}
            self.env.mouse['x'] = int(x / self.unwrapped.viewer.window.width * 240)
            self.env.mouse['y'] = int((self.unwrapped.viewer.window.height - y) / self.unwrapped.viewer.window.height * 224)
            
            # NOTE:
            # the values above are hardcoded for zelda;
            # 240 is the y resolution,
            # and 60 is the height of the black bar
        self.unwrapped.viewer.window.push_handlers(on_mouse_motion)

        def on_mouse_leave(x, y):
            self.unwrapped.mouse = None
        self.unwrapped.viewer.window.push_handlers(on_mouse_leave)

    def reset(self, **kwargs):
        self.frames_since_log = 0
        self.last_log_time = time.time()
        self.last_log_reward = 0
        self.this_log_reward = 0
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
        if self._key_handler.get(keycodes.ESCAPE):
            self.action_override = False
        if self.action_override:
            action = self.keys_to_act(keys_pressed)
        #print(f"action={action}")
        observation, reward, terminated, truncated, info = super().step(action)
        #results = (observation, lang_input), reward + manual_reward, terminated, truncated, info
        results = observation, reward + manual_reward, terminated, truncated, info
        self.this_log_reward += reward
        self.total_reward += reward

        # sleep to adjust the fps
        steptime = time.time() - self.laststep
        sleeptime = 1.0/self.get_maxfps() - steptime
        time.sleep(max(0, sleeptime))
        self.laststep = time.time()

        # report debugging info
        time_diff = time.time() - self.last_log_time
        self.frames_since_log += 1
        if time_diff >= self.log_every:
            logging.debug(f'fps={self.frames_since_log} reward_diff={self.this_log_reward:+0.4f} total_reward={self.total_reward:+0.4f}')
            self.last_log_time = time.time()
            self.frames_since_log = 0
            self.this_log_reward = 0


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
    parser.add_argument('--task', default='attack')
    parser.add_argument('--state', default='spiders_lowhealth_01*.state')
    parser.add_argument('--model', default='models/simple_attack.zip')
    parser.add_argument('--logfile', default='.play.log')
    parser.add_argument('--action_space', default='DISCRETE')

    emulator_settings = parser.add_argument_group('emulator settings')
    emulator_settings.add_argument('--no_render_skipped_frames', action='store_true')
    emulator_settings.add_argument('--allframes', action='store_true')
    emulator_settings.add_argument('--no_alternate_screen', action='store_true')
    emulator_settings.add_argument('--noaudio', action='store_true')
    emulator_settings.add_argument('--doresets', action='store_true')

    args = parser.parse_args()

    # set logging level
    import logging
    logging.basicConfig(
        filename=args.logfile,
        level=logging.DEBUG,
        format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # convert warnings to errors
    import warnings
    warnings.filterwarnings('always', append=True)
    warnings.simplefilter('error')

    # create the environment
    logging.info('creating environment')
    env = make_zelda_env(
            # FIXME:
            # the action_space should be loaded automatically from the model
            action_space=args.action_space,
            stdout_debug=not args.no_alternate_screen,
            no_render_skipped_frames=args.no_render_skipped_frames,
            skip_boring_frames=not args.allframes,
            task=args.task,
            )
    env = Interactive(env)
    if not args.noaudio:
        env = PlayAudio(env)
    #env = StochasticFrameSkip(env, 4, 0.25)
    env = RandomStateReset(env, path='custom_integrations/Zelda-Nes', globstr=args.state)
    #env = Whisper(env)
    #env = ConsoleWrapper(env)

    # NOTE:
    # loading the models can cause a large number of warnings;
    # this with block prevents those warnings from being displayed
    logging.info('loading model')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from stable_baselines3 import PPO
        custom_objects = {
            'observation_space': env.observation_space,
            'action_space': env.action_space,
            }
        model = None
        #model = PPO.load(args.model, custom_objects=custom_objects)

    logging.info('begin main loop')
    env.reset()
    env.reset()
    try:
        # set the default action
        action = env.action_space.sample() * 0

        # switch the terminal to the alternate screen
        if not args.no_alternate_screen:
            print('\u001B[?1049h')

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
                action[2] = 0
                action[3] = 0

            #import code
            #code.interact(local=locals())
            # >>> model.policy.value_net.weight
            # >>> list(zip(env.observations_keys, model.policy.value_net.weight[0]))
            # for x in list(zip(env.observations_keys, model.policy.value_net.weight[0])): print(x)
            # for x in list(zip(env.observations_keys, model.policy.action_net.weight.T)):
    except KeyboardInterrupt:
        pass

    # clean all resources
    env.close()

    # switch the terminal away from the alternate screen
    if not args.no_alternate_screen:
        print('\u001B[?1049l')


if __name__ == "__main__":
    main()
