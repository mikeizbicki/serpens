import retro
import time
import os
from  gymnasium.wrappers import *
import stable_baselines3
from stable_baselines3 import TD3
from stable_baselines3.common import results_plotter
from stable_baselines3.common.vec_env import *
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from Mundus.Wrappers import *
import torch


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), "timesteps")
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose >= 1:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="Zelda-Nes")
    parser.add_argument("--state", default=retro.State.DEFAULT)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--log_dir", default='log')
    parser.add_argument("--nproc", type=int, default=4)
    args = parser.parse_args()

    # create the environment
    def make_env():
        custom_path = os.path.join(os.getcwd(), 'custom_integrations')
        retro.data.Integrations.add_custom_path(custom_path)
        env = retro.make(
                game=args.game,
                inttype=retro.data.Integrations.ALL,
                #render_mode='rgb_array',
                )
        env = TimeLimit(env, max_episode_steps=30*60*5)
        #env = StochasticFrameSkip(env, 4, 0.25)
        env = ObserveVariables(env)
        env = FrameStack(env, 4)
        env = RandomStateReset(env, path='custom_integrations/'+args.game)
        return env
    env = SubprocVecEnv([make_env] * args.nproc)
    env = VecMonitor(env, args.log_dir)
    #env = DummyVecEnv([make_env])
    #env = Monitor(env, args.log_dir)
    #env = make_env()
    env.reset()

    model = stable_baselines3.PPO(
        policy="MlpPolicy",
        env=env,
        #learning_rate=lambda f: f * 2.5e-4,
        learning_rate=lambda f: f * 2.5e-3,
        n_steps=128,
        batch_size=32,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        verbose=1,
        #policy_kwargs={'net_arch': [96, 96]}
    )
    #model = stable_baselines3.PPO('MlpPolicy', env, verbose=1)
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=args.log_dir)
    model.learn(
        total_timesteps=100_000_000,
        log_interval=1,
        callback = callback,
    )

    # cleanly close resources
    env.close()


if __name__ == "__main__":
    main()

