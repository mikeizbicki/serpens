import retro
import gymnasium
import pyglet
import pyaudio
import pygame
import time
import numpy as np
import random
import os
import logging
import sys
import pprint

from Mundus.Wrappers import *
from Mundus.Zelda import *


# FIXME:
import warnings
warnings.filterwarnings("ignore")


import threading
class KeyboardThread(threading.Thread):
    '''
    Taken from <https://stackoverflow.com/a/57387909>.
    '''

    def __init__(self, prompt=None, input_cbk=None, name='keyboard-input-thread'):
        self.input_cbk = input_cbk
        self.prompt = prompt
        super(KeyboardThread, self).__init__(name=name, daemon=True)
        self.start()

    def run(self):
        while True:
            self.input_cbk(input(self.prompt)) #waits to get input + Return


class ConsoleWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        # create multithreaded keyboard input
        self.current_command = None
        def set_current_command(text):
            self.current_command = text
        self.input_thread = KeyboardThread('input: ', set_current_command)


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
        pygame.init()
        self.joysticks = {}

        # setup the key handler
        self._key_previous_states = {}
        self._key_handler = pyglet.window.key.KeyStateHandler()

        # NOTE:
        # self.render() is needed to create the self.viewer object
        self.render() 
        self.viewer.window.push_handlers(self._key_handler)

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

        # handle meta-actions
        manual_reward = 0
        lang_input = None
        for name in keys_pressed_this_frame:
            if name == 'BRACKETLEFT':
                self.maxfps_multiplier += 1
                print(f"self.get_maxfps()={self.get_maxfps()}")

            if name == 'BRACKETRIGHT':
                self.maxfps_multiplier -= 1
                print(f"self.get_maxfps()={self.get_maxfps()}")

            if name == 'SPACE':
                import code
                code.interact(local=locals())
                self.action_override = prev_action_override
                #newstate = b'\x00'*88; state = self.env.unwrapped.em.get_state(); state = state[:14657]+newstate+state[14657+len(newstate):]; self.env.unwrapped.em.set_state(state)

        for name in keys_pressed:
            if name == 'MINUS':
                manual_reward -= 1
                self.action_override = prev_action_override
                print(f"manual_reward={manual_reward}")

            if name == 'EQUAL':
                manual_reward += 1
                self.action_override = prev_action_override
                print(f"manual_reward={manual_reward}")

        # perform the action
        if self._key_handler.get(keycodes.ESCAPE):
            self.action_override = False
        if self.action_override:
            action = self.keys_to_act(keys_pressed)
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
            logging.info(f'fps={self.frames_since_log} reward_diff={self.this_log_reward:+0.4f} total_reward={self.total_reward:+0.4f}')
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
        pushed_buttons = [b for b in self.env.buttons if inputs[b]]
        #print(f"pushed_buttons={pushed_buttons}")
        return [inputs[b] for b in self.env.buttons]


class PlayAudio(gymnasium.Wrapper):
    def __init__(self, env, ignore_ALSA_warnings=True):
        super().__init__(env)
        self.chunks = []
        self.frame_index = 0
    
        def buffer_size():
            return sum([len(chunk) for chunk in self.chunks]) - self.frame_index

        # ALSA prints a lot of debugging information to stderr by default;
        # this is annoying, and the following code disables these messages;
        # see: <https://stackoverflow.com/a/13453192/1241368>
        if ignore_ALSA_warnings:
            from ctypes import CFUNCTYPE, cdll, c_char_p, c_int
            ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
            def py_error_handler(filename, line, function, err, fmt):
                return
            c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
            asound = cdll.LoadLibrary('libasound.so')
            asound.snd_lib_error_set_handler(c_error_handler)

        # create the audio stream
        def playing_callback(in_data, frame_count, time_info, status):
            frame_count *= 2
            if len(self.chunks) == 0:
                data = np.array([0]*frame_count).astype('int16')
            else:
                chunk0 = self.chunks[0]
                chunk0_len = self.chunks[0].shape[0]
                data = chunk0[self.frame_index:self.frame_index + frame_count]
                self.frame_index += frame_count
                if len(data) < frame_count:
                    self.chunks = self.chunks[1:]
                    self.frame_index = 0
                    frame_count_diff = frame_count - len(data)
                    if len(self.chunks) == 0:
                        data = np.concatenate([data, [0]*frame_count_diff])
                    else:
                        data = np.concatenate([data, self.chunks[0][:frame_count_diff]])
                        self.frame_index = frame_count_diff
            return (data, pyaudio.paContinue)

        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=2,
                rate=int(env.em.get_audio_rate()/2),
                output=True,
                stream_callback=playing_callback,
                )

        # FIXME:
        # commenting this line out will remove the underrun warnings;
        # but the error handler defined above will then have problems
        asound.snd_lib_error_set_handler(None)

    def step(self, actions):
        data = self.env.em.get_audio().flatten().astype('int16')
        self.chunks.append(data)
        if len(self.chunks) > 10:
            logging.warning('audiobuffer overflowing, truncating self.chunks')
            self.chunks = self.chunks[-1:]
        return super().step(actions)

    def close(self):
        self.stream.close()
        self.p.terminate()
        super().close()
    

def main():
    # parse command line args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="Zelda-Nes")
    parser.add_argument("--state", default=retro.State.DEFAULT)
    parser.add_argument("--scenario", default=None)
    args = parser.parse_args()

    # set logging level
    import logging
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)

    # create the environment
    custom_path = os.path.join(os.getcwd(), 'custom_integrations')
    retro.data.Integrations.add_custom_path(custom_path)
    env = retro.make(
            game=args.game,
            inttype=retro.data.Integrations.ALL
            )
    #env = StochasticFrameSkip(env, 4, 0.25)
    env = ObserveVariables(env)
    env = RandomStateReset(env, path='custom_integrations/'+args.game)
    env = ZeldaWrapper(env, stdout_debug=True)
    env = Interactive(env)
    #env = PlayAudio(env)
    #env = ConsoleWrapper(env)
    env.reset()

    # main game loop
    try:
        while True:
            # set no action as the default
            action = env.action_space.sample() * 0
            observation, reward, terminated, truncated, info = env.step(action)
            env.render()
            data = env.unwrapped.get_ram()[0xba2+0x800:0xba2+0x800+4]
            data = env.unwrapped.get_ram()[0x12ba+0x800:0x12ba+0x800+4]
    except KeyboardInterrupt:
        pass

    # clean all resources
    env.close()


if __name__ == "__main__":
    main()
