import retro
import gymnasium
import pyglet
import time


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
    def __init__(self, env, maxfps=60000):
        super().__init__(env)

        self.action_override = False
        self.maxfps0 = maxfps
        self.maxfps_multiplier = 0
        self.laststep = time.time()

        # setup the key handler
        self._key_previous_states = {}
        self._key_handler = pyglet.window.key.KeyStateHandler()
        self.render() # create the self.viewer object
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
                #state = self.em.get_state()
                #import gzip
                #with gzip.open('snapshot.state', mode='wb') as f:
                    #f.write(state)
                #print(f"state={state}")
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

def main():
    import os
    custom_path = os.path.join(os.getcwd(), 'custom_integrations')
    retro.data.Integrations.add_custom_path(custom_path)

    env = retro.make(
            #game="GauntletII-Nes",
            game="Zelda-Nes",
            #game="MyMario-Nes",
            #game="Gauntlet-Sms",
            #game="Gauntlet-Genesis",
            #render_mode="rgb_array",
            inttype=retro.data.Integrations.ALL
            )
    env = Interactive(env)
    env.reset()

    #print(f"env.statename={env.statename}")


    '''
    from stable_baselines3 import PPO
    model = PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=lambda f: f * 2.5e-4,
        n_steps=128,
        batch_size=32,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        verbose=1,
    )
    model.learn(
        total_timesteps=100_000_000,
        log_interval=1,
    )
    '''

    start_time = time.time()
    i = 0
    total_printed = 0
    frames_since_print = 0
    while True:
        i += 1
        frames_since_print += 1
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            env.reset()
        elapsed = time.time() - start_time
        if elapsed > total_printed:
            total_printed += 1
            print(f' total_printed={total_printed} elapsed={elapsed} fps={frames_since_print}')
            frames_since_print = 0

    env.close()


if __name__ == "__main__":
    main()

