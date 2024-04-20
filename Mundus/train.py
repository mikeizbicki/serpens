import retro
import time
import os

from Mundus.Wrappers import *

def main():
    import os
    custom_path = os.path.join(os.getcwd(), 'custom_integrations')
    retro.data.Integrations.add_custom_path(custom_path)

    def make_env():
        env = retro.make(
                #game="GauntletII-Nes",
                game="Zelda-Nes",
                #game="MyMario-Nes",
                #game="Gauntlet-Sms",
                #game="Gauntlet-Genesis",
                #render_mode="rgb_array",
                inttype=retro.data.Integrations.ALL
                )
        #env = RenderSkip(env)
        env = StochasticFrameSkip(env, 4, 0.25)
        env = ObserveVariables(env)
        return env


    import stable_baselines3
    from stable_baselines3.common.vec_env import (
        SubprocVecEnv,
        VecFrameStack,
        VecTransposeImage,
    )
    #venv = VecTransposeImage(VecFrameStack(SubprocVecEnv([make_env] * 4), n_stack=2))
    env = stable_baselines3.common.vec_env.DummyVecEnv([make_env])
    #print(f"env.observation_space={env.observation_space}")
    env.reset()
    #print(f"env.observation_space={env.observation_space}")
    #asd
    model = stable_baselines3.PPO(
        policy="MlpPolicy",
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
    #model = stable_baselines3.DQN('MlpPolicy', env, verbose=1)
    model.learn(
        total_timesteps=100_000_000,
        log_interval=1,
    )

    env.close()


if __name__ == "__main__":
    main()

