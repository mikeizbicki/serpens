import time

import gymnasium
from  gymnasium.wrappers import *
import retro

from Mundus.Audio import *
from Mundus.Wrappers import *
from Mundus.Zelda import *

def main():
    # parse command line args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="Zelda-Nes")
    parser.add_argument('--render_mode', choices=['human', 'rgb_array'], default='rgb_array')
    parser.add_argument('--ZeldaWrapper', action='store_true')
    args = parser.parse_args()

    # set logging level
    import logging
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    # create the environment
    logging.info('creating environment')
    custom_path = os.path.join(os.getcwd(), 'custom_integrations')
    retro.data.Integrations.add_custom_path(custom_path)
    env = retro.make(
            game=args.game,
            inttype=retro.data.Integrations.ALL,
            use_restricted_actions=retro.Actions.ALL,
            render_mode='rgb_array',
            )
    if args.ZeldaWrapper:
        env = ZeldaWrapper(
                env,
                skip_boring_frames=False,
                #reorder_outputs=False,
                #clean_outputs=False,
                )
    #env = RandomStateReset(env, path='custom_integrations/'+args.game, globstr='spiders_lowhealth_01*.state')
    env.reset()

    # set the default action
    action = env.action_space.sample() * 0

    last_time = time.time()
    frames_since_log = 0
    while True:
        # step environment
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            env.reset()

        # report fps
        time_diff = time.time() - last_time
        frames_since_log += 1
        if time_diff >= 1:
            logging.info(f'fps={frames_since_log}')
            last_time = time.time()
            frames_since_log = 0

    # clean all resources
    env.close()


if __name__ == "__main__":
    main()

