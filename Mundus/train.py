import retro
import time
import os
from collections import Counter
from  gymnasium.wrappers import *
import stable_baselines3
from stable_baselines3 import TD3
from stable_baselines3.common import results_plotter
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import *
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam, Video
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import *
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
    def __init__(
            self,
            check_freq: int=1_000,
            log_dir: str=None,
            verbose: int=1,
            min_episodes=100,
            ):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.best_mean_reward = -np.inf
        self.min_episodes = min_episodes

    def _init_callback(self) -> None:
        if self.log_dir is None:
            self.log_dir = self.logger.get_dir()
        self.save_path = os.path.join(self.log_dir, 'model')

    def _on_step(self) -> bool:
        #import code
        #code.interact(local=locals())
        #print('self.n_calls=', self.n_calls)
        if self.n_calls % self.check_freq == 0:
            logging.info(f'saving model to {self.save_path}')
            self.model.save(self.save_path)

        '''
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
        '''

        return True


class HParamCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    """

    def _on_training_start(self) -> None:
        def _wrap_value(x):
            if hasattr(x, '__call__'):
                return x(1)
            else:
                return x

        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning rate": _wrap_value(self.model.learning_rate),
            "gamma": self.model.gamma,
        }
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorboard will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "episode/is_success": 0,
            "rollout/ep_len_mean": 0,
            "rollout/ep_rew_mean": 0,
            "rollout/is_success": 0,
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


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(
            self,
            verbose=0,
            max_video_length=60*30,
            record_every=1,
            ):
        super().__init__(verbose)

        # store settings
        self.record_every = record_every
        self.max_video_length = max_video_length

        # these variables are internally created/updated by ._on_step()
        self.num_episodes = None
        self.num_scenarios = Counter()
        if self.record_every is None:
            self.screens = None
        else:
            self.screens = []

    def _on_step(self) -> bool:

        # on the first iteration,
        # we set num_episodes to a list whose length is the number of vectorized environments;
        # each entry will count the total number of episodes for that environment
        if self.num_episodes is None:
            self.num_episodes = [0] * len(self.locals['dones'])

        # if self.screens is a list,
        # then we are currently recording the screens of the first environment
        if self.screens is not None and len(self.screens) < self.max_video_length:
            screen = self.training_env.get_images()[0]
            self.screens.append(screen.transpose(2, 0, 1))
        
        # for each environment, we record our info if it is done
        for i, done in enumerate(self.locals['dones']):
            if done:
                self.num_episodes[i] += 1

                # special screen recording processing for the first environment only
                if self.record_every is not None and i == 0:

                    # record the screen
                    if type(self.screens) == list:
                        self.logger.record(
                            "trajectory/video",
                            Video(torch.from_numpy(np.asarray([self.screens])), fps=60),
                            exclude=("stdout", "log", "json", "csv"),
                        )
                        self.screens = None

                    # begin processing a new screen only occasionally
                    if self.num_episodes[i] % self.record_every == 0:
                        self.screens = []

                # record all of the "summary_*" keys within the info dict
                def record_prefix(prefix):
                    keys = [key for key in self.locals['infos'][i].keys() if key.startswith(prefix)]
                    prefixmod = prefix.strip('_')
                    for key in keys:
                        keymod = key[len(prefix):]
                        self.logger.record_mean(prefixmod+'/'+keymod, self.locals['infos'][i][key], exclude=('stdout',))

                record_prefix('summary_')
                record_prefix('reward_')
                record_prefix('scenario_success_')
                #record_prefix('scenario_count_')

                prefix = 'scenario_count_'
                keys = [key for key in self.locals['infos'][i].keys() if key.startswith(prefix)]
                self.logger.record('episode/num_episodes', sum(self.num_episodes))
                prefixmod = prefix.strip('_')
                for key in keys:
                    keymod = key[len(prefix):]
                    self.num_scenarios[keymod] += 1
                    self.logger.record(prefixmod+'/'+keymod, self.num_scenarios[keymod], exclude=('stdout',))

                #keys = [key for key in self.locals['infos'][i].keys() if key.startswith('summary_')]
                #for key in keys:
                    #self.logger.record_mean('summary/'+key, self.locals['infos'][i][key])
                #keys = [key for key in self.locals['infos'][i].keys() if key.startswith('reward_')]
                #self.logger.record('episode/num_episodes', sum(self.num_episodes))
                #for key in keys:
                    #self.logger.record_mean('reward/'+key, self.locals['infos'][i][key])
#
                #scenario_keys = [key for key in self.locals['infos'][i].keys() if key.startswith('scenario_success_')]
                #for key in scenario_keys:
                    #self.logger.record_mean('scenario_success/'+key, self.locals['infos'][i][key])
        return True


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="Zelda-Nes")
    parser.add_argument("--log_tensorboard", default='log_tensorboard')

    debug = parser.add_argument_group('debug')
    debug.add_argument("--log_dir", default='log')
    debug.add_argument("--disable_video", action='store_true')
    debug.add_argument("--render_mode", default='rgb_array')
    debug.add_argument("--comment")
    debug.add_argument("--total_timesteps", default=1_000_000_000_000, type=int)

    hyperparameters = parser.add_argument_group('hyperparameters')
    hyperparameters.add_argument('--policy', choices=['MlpPolicy', 'CnnPolicy', 'ObjectCnn'], default=['ObjectCnn'])
    hyperparameters.add_argument('--pooling', choices=['lstm', 'mean', 'max'], default='mean')
    #hyperparameters.add_argument('--scenario', default='attack')
    hyperparameters.add_argument('--scenario', default=None)
    hyperparameters.add_argument('--state', default='spiders_lowhealth_01*.state')
    hyperparameters.add_argument('--net_arch', type=int, nargs='*', default=[])
    hyperparameters.add_argument('--features_dim', type=int, default=64)
    hyperparameters.add_argument('--lr', type=float, default=3e-4)
    hyperparameters.add_argument('--gamma', type=float, default=0.99)
    hyperparameters.add_argument('--n_env', type=int, default=3)
    hyperparameters.add_argument('--n_steps', type=int, default=128)
    hyperparameters.add_argument('--batch_size', type=int, default=32)
    hyperparameters.add_argument('--seed', type=int, default=None)
    hyperparameters.add_argument('--warmstart', default=None)

    args = parser.parse_args()

    # set seed
    if args.seed is None:
        args.seed = int(time.time()*10_000_000) % 2**32

    # set logging level
    import logging
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    # experiment name
    import datetime
    starttime = datetime.datetime.now().isoformat()

    arch_string = '-'.join([str(i) for i in args.net_arch])
    policy_params = ''
    if args.policy == 'ObjectCnn':
        policy_params = f',pooling={args.pooling}'

    experiment_name = ''
    if args.comment is not None:
        experiment_name = args.comment + '--'
    experiment_name += f'scenario={args.scenario},state={args.state},policy={args.policy}{policy_params},net_arch={arch_string},{args.features_dim},lr={args.lr},gamma={args.gamma},n_env={args.n_env},n_steps={args.n_steps},batch_size={args.batch_size}'
    logging.info(f'experiment_name: [{experiment_name}]')

    # create the environment
    def make_env(render_mode):
        custom_path = os.path.join(os.getcwd(), 'custom_integrations')
        retro.data.Integrations.add_custom_path(custom_path)
        env = retro.make(
                game=args.game,
                inttype=retro.data.Integrations.ALL,
                state='overworld_07',
                render_mode=render_mode,
                )
        env = ZeldaWrapper(
                env,
                skip_boring_frames=False,
                scenario=args.scenario,
                )
        env = TimeLimit(env, max_episode_steps=30*60*5)
        env = StochasticFrameSkip(env, 4, 0.25)
        env = RandomStateReset(env, path='custom_integrations/'+args.game, globstr=args.state)
        return env

    train_env = SubprocVecEnv([lambda: make_env(render_mode=args.render_mode)] * args.n_env)
    train_env = VecMonitor(train_env)
    train_env.reset()

    policy_kwargs = {
        'net_arch': args.net_arch,
        }
    if args.policy == 'ObjectCnn':
        policy_kwargs['features_extractor_class'] = ObjectCnn
        policy_kwargs['features_extractor_kwargs'] = {
            'pooling': args.pooling,
            'features_dim': args.features_dim,
            }
    elif args.policy == 'RewardsExtractor':
        policy_kwargs['features_extractor_class'] = RewardExtractor
        policy_kwargs['features_extractor_kwargs'] = {
            'pooling': args.pooling,
            'features_dim': args.features_dim,
            }
    if 'Cnn' in args.policy:
        policy = ActorCriticCnnPolicy
    else:
        policy = 'MlpPolicy'
    model = stable_baselines3.PPO(
        policy=policy,
        env=train_env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=4,
        gamma=args.gamma,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=args.log_dir,
        policy_kwargs=policy_kwargs,
        seed=args.seed,
    )
    reset_num_timesteps = True
    if args.warmstart:
        reset_num_timesteps = False
        warmstart = stable_baselines3.PPO.load(args.warmstart)
        model.policy = warmstart.policy
        model.num_timesteps = warmstart.num_timesteps
        #import code
        #code.interact(local=locals())
    model.learn(
        total_timesteps=args.total_timesteps,
        log_interval=1,
        tb_log_name=experiment_name,
        reset_num_timesteps=reset_num_timesteps,
        callback = [
            SaveOnBestTrainingRewardCallback(),
            #HParamCallback(),
            TensorboardCallback(record_every=None if args.disable_video or args.render_mode=='human' else 1),
            ],
    )

    # cleanly close resources
    train_env.close()


if __name__ == "__main__":
    main()

