import retro
import gymnasium
import pyglet
import pyaudio
import time
import numpy as np
import random
import os
import logging

from Mundus.Wrappers import *


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
    def __init__(self, env, maxfps=30):
        super().__init__(env)

        self.action_override = False
        self.maxfps0 = maxfps
        self.maxfps_multiplier = 0
        self.laststep = time.time()

        # setup the key handler
        self._key_previous_states = {}
        self._key_handler = pyglet.window.key.KeyStateHandler()

        # NOTE:
        # self.render() is needed to create the self.viewer object
        self.render() 
        self.viewer.window.push_handlers(self._key_handler)

    def get_maxfps(self):
        return self.maxfps0 * 2**(self.maxfps_multiplier)
        
    def step(self, action):

        # compute which keys are pressed
        keys_pressed = []
        keys_pressed_this_frame = []
        prev_action_override = self.action_override
        for keycode, pressed in self._key_handler.items():
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
                #lang_input = input('Language input: ')
                import code
                code.interact(local=locals())
                self.action_override = prev_action_override

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

        # sleep to adjust the fps
        steptime = time.time() - self.laststep
        sleeptime = 1.0/self.get_maxfps() - steptime
        time.sleep(max(0, sleeptime))
        self.laststep = time.time()

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
    def __init__(self, env):
        super().__init__(env)
        self.chunks = []
        self.frame_index = 0
    
        def buffer_size():
            return sum([len(chunk) for chunk in self.chunks]) - self.frame_index

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

        p = pyaudio.PyAudio()
        self.stream = p.open(
                format=pyaudio.paInt16,
                channels=2,
                rate=int(env.em.get_audio_rate()/2),
                output=True,
                stream_callback=playing_callback,
                )

    def step(self, actions):
        data = self.env.em.get_audio().flatten().astype('int16')
        self.chunks.append(data)
        if len(self.chunks) > 10:
            logging.warning('audiobuffer overflowing, truncating self.chunks')
            self.chunks = self.chunks[-1:]
        return super().step(actions)

    def close(self):
        self.stream.close()
        p.terminate()
        super().close()
    

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="Zelda-Nes")
    parser.add_argument("--state", default=retro.State.DEFAULT)
    parser.add_argument("--scenario", default=None)
    args = parser.parse_args()

    # create the environment
    custom_path = os.path.join(os.getcwd(), 'custom_integrations')
    retro.data.Integrations.add_custom_path(custom_path)
    env = retro.make(
            game=args.game,
            inttype=retro.data.Integrations.ALL
            )
    env = StochasticFrameSkip(env, 4, 0.25)
    env = ObserveVariables(env)
    env = RandomStateReset(env, path='custom_integrations/'+args.game)
    env = Interactive(env)
    env = PlayAudio(env)
    env.reset()

    # main game loop
    while True:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            env.reset()

    env.close()


if __name__ == "__main__":
    main()
