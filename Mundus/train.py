import retro
import time
import os
from  gymnasium.wrappers import *
import stable_baselines3
from stable_baselines3 import TD3
from stable_baselines3.common import results_plotter
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import *
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam, Video
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from Mundus.Wrappers import *
from Mundus.Zelda import *
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
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1, min_episodes=20):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf
        self.min_episodes = min_episodes

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), "timesteps")
          if len(x) > self.min_episodes:
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


from typing import Any, Dict
class VideoRecorderCallback(BaseCallback):
    def __init__(self, eval_env: gymnasium.Env, render_freq: int, n_eval_episodes: int = 1, deterministic: bool = False):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to TensorBoard

        :param eval_env: A gym environment from which the trajectory is recorded
        :param render_freq: Render the agent's trajectory every eval_freq call of the callback.
        :param n_eval_episodes: Number of episodes to render
        :param deterministic: Whether to use deterministic or stochastic policy
        """
        super().__init__()
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic

    def _on_step(self) -> bool:
        if self.n_calls % self._render_freq == 0:
            screens = []

            def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                """
                Renders the environment in its current state, recording the screen in the captured `screens` list

                :param _locals: A dictionary containing all local variables of the callback's scope
                :param _globals: A dictionary containing all global variables of the callback's scope
                """
                # We expect `render()` to return a uint8 array with values in [0, 255] or a float array
                # with values in [0, 1], as described in
                # https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_video
                #screen = self._eval_env.render(mode="rgb_array")
                screen = self._eval_env.get_wrapper_attr('get_screen')()
                # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
                screens.append(screen.transpose(2, 0, 1))

            mean, stddev = evaluate_policy(
                self.model,
                self._eval_env,
                callback=grab_screens,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
            )
            self.logger.record(
                'evaluate/mean_reward',
                mean,
                exclude=("stdout", "log", "json", "csv"),
            )
            self.logger.record(
                "trajectory/video",
                Video(torch.from_numpy(np.asarray([screens])), fps=60),
                exclude=("stdout", "log", "json", "csv"),
            )
        return True


class HParamCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    """

    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning rate": self.model.learning_rate,
            "gamma": self.model.gamma,
        }
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "rollout/ep_len_mean": 0,
            "train/value_loss": 0.0,
            'evaluate/mean_reward': 0.0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="Zelda-Nes")
    parser.add_argument("--state", default=retro.State.DEFAULT)
    parser.add_argument("--render_mode", default=None)
    parser.add_argument("--log_monitor", default='log_monitor')
    parser.add_argument("--log_tensorboard", default='log_tensorboard')
    parser.add_argument("--nproc", type=int, default=3)
    args = parser.parse_args()

    # create the environment
    def make_env(render_mode):
        custom_path = os.path.join(os.getcwd(), 'custom_integrations')
        retro.data.Integrations.add_custom_path(custom_path)
        env = retro.make(
                game=args.game,
                inttype=retro.data.Integrations.ALL,
                state='overworld_07',
                render_mode=render_mode,
                #render_mode='rgb_array',
                )
        env = ZeldaWrapper(
                env,
                skip_boring_frames=False,
                )
        env = TimeLimit(env, max_episode_steps=30*60*5)
        env = StochasticFrameSkip(env, 4, 0.25)
        #env = ObserveVariables(env)
        #env = FrameStack(env, 1)
        #env = RandomStateReset(env, path='custom_integrations/'+args.game)
        env = RandomStateReset(env, path='custom_integrations/'+args.game, globstr='spiders_lowhealth_01*.state')
        #env = RandomStateReset(env, path='custom_integrations/'+args.game, globstr='overworld_07.state')
        return env
    #eval_env = make_env(render_mode='rgb_array')
    #eval_env = Monitor(eval_env, args.log_monitor + '/eval')

    train_env = SubprocVecEnv([lambda: make_env(render_mode=args.render_mode)] * args.nproc)
    train_env = VecMonitor(train_env, args.log_monitor)
    train_env.reset()

    model = stable_baselines3.PPO(
        policy="MlpPolicy",
        env=train_env,
        #learning_rate=2.5e-4,
        learning_rate=lambda f: f * 2.5e-4,
        #n_steps=128,
        n_steps=1024,
        batch_size=32,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=args.log_tensorboard,
        #policy_kwargs={'net_arch': [1024]}
        #policy_kwargs={'net_arch': [1024], 'activation_fn': torch.nn.LeakyReLU}
        #policy_kwargs={'net_arch': [4024], 'activation_fn': torch.nn.ReLU}
        #policy_kwargs={'net_arch': [1024], 'activation_fn': torch.nn.Tanh}
        #policy_kwargs={'net_arch': [96, 96]}
        #policy_kwargs={'net_arch': [256, 256]}
        #policy_kwargs={'net_arch': [256, 256], 'activation_fn': torch.nn.ReLU}
    )
    model.learn(
        total_timesteps=100_000_000,
        log_interval=1,
        callback = [
            SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=args.log_monitor),
            #VideoRecorderCallback(eval_env, 1024*4),
            #HParamCallback(),
            ],
    )

    # cleanly close resources
    train_env.close()
    #eval_env.close()


if __name__ == "__main__":
    main()

