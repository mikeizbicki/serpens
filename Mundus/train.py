from collections import Counter
import hashlib
import os
import re
import retro
import time

from gymnasium.wrappers import *
from stable_baselines3 import TD3
from stable_baselines3.common import results_plotter
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import HParam, Video
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import *
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.vec_env import *
from torch.utils.tensorboard import SummaryWriter
import stable_baselines3
import torch

from Mundus.Wrappers import *
from Mundus.Games import *


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
        self.num_tasks = Counter()
        if self.record_every is None:
            self.screens = None
        else:
            self.screens = []

    def _init_callback(self) -> None:
        # self.logger already contains a SummaryWriter,
        # but it's hidden away where we can't access it directly;
        # creating a new one here allows us to create plots with
        # different x-axes
        self.my_writer = SummaryWriter(log_dir=self.logger.get_dir())

    def _on_step(self) -> bool:
        # on the first iteration,
        # we set num_episodes to a list whose length is the number of vectorized environments;
        # each entry will count the total number of episodes for that environment
        if self.num_episodes is None:
            self.num_episodes = [0] * len(self.locals['dones'])
            self.task_counters = [Counter() for env in self.locals['dones']]

        # if self.screens is a list,
        # then we are currently recording the screens of the first environment
        if self.screens is not None and len(self.screens) < self.max_video_length:
            screen = self.training_env.get_images()[0]
            self.screens.append(screen.transpose(2, 0, 1))
        
        # for each environment, we record our info if it is done
        for i, done in enumerate(self.locals['dones']):
            if done:
                self.num_episodes[i] += 1
                sum_num_episodes = sum(self.num_episodes)

                episode_task = self.locals['infos'][i]['episode_task']
                self.task_counters[i][episode_task] += 1
                num_tasks = sum(self.task_counters, start=Counter())
                num_episodes_with_current_task = num_tasks[episode_task]

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

                # record the info dict to tensorboard
                def record_prefix(prefix):
                    keys = [key for key in self.locals['infos'][i].keys() if key.startswith(prefix)]
                    prefixmod = prefix.strip('_')
                    for key in keys:
                        keymod = key[len(prefix):]
                        self.logger.record_mean('step_' + prefixmod+'/'+keymod, self.locals['infos'][i][key], exclude=('stdout',))
                        self.my_writer.add_scalar('episode_' + prefixmod + '_' +episode_task + '/'+keymod, self.locals['infos'][i][key], num_episodes_with_current_task)

                record_prefix('event_')
                record_prefix('misc_')

                self.my_writer.add_scalar('_rewards/'+episode_task, self.locals['infos'][i]['misc_episode_reward'], num_episodes_with_current_task)
                self.my_writer.add_scalar('_pseudorewards/'+episode_task, self.locals['infos'][i]['misc_episode_pseudoreward'], num_episodes_with_current_task)
                self.my_writer.add_scalar('_success/'+episode_task, self.locals['infos'][i]['event__success'], num_episodes_with_current_task)
                self.my_writer.add_scalar('_step/'+episode_task, self.locals['infos'][i]['misc_step'], num_episodes_with_current_task)

                self.logger.record('step_misc/num_episodes', sum(self.num_episodes))
        return True


def main():
    import argparse
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group('debug options')
    group.add_argument("--log_dir", default='log')
    group.add_argument("--disable_video", action='store_true')
    group.add_argument("--render_mode", default='rgb_array')
    group.add_argument("--comment")
    group.add_argument("--total_timesteps", default=1_000_000_000_000, type=int)

    group = parser.add_argument_group('hyperparameters: architecture')
    group.add_argument('--action_space', default='zelda-all')
    group.add_argument('--policy', default='ExtractorManager')
    group.add_argument('--objects_extractor', default='ObjectEmbedding')
    group.add_argument('--revents_extractor', default='Revents_Linear')
    group.add_argument('--pooling', choices=['lstm', 'mean', 'max'], default='mean')
    group.add_argument('--net_arch', type=int, nargs='*', default=[])
    group.add_argument('--features_dim', type=int, default=256)
    group.add_argument('--difference_dim', type=int, default=8)
    group.add_argument('--events_dim', type=int, default=32)
    group.add_argument('--rewards_dim', type=int, default=32)

    group = parser.add_argument_group('hyperparameters: training algorithm')
    group.add_argument('--alg', choices=['ppo', 'dqn'], default='ppo')
    group.add_argument('--lr', type=float, default=3e-4)
    group.add_argument('--gamma', type=float, default=0.9)
    group.add_argument('--n_env', type=int, default=128)
    group.add_argument('--n_steps', type=int, default=128)
    group.add_argument('--batch_size', type=int, default=256)
    group.add_argument('--warmstart', default=None)

    group = parser.add_argument_group('hyperparameters: environment')
    group.add_argument('--game', default='Zelda')
    group.add_argument('--center_player', action='store_true')
    group.add_argument('--reset_method', default='map link enemy', type=str)
    group.add_argument('--reset_state', default=None, type=str)
    group.add_argument('--task_regex', default='^attack$')
    group.add_argument('--seed', type=int, default=42)
    group.add_argument('--frames_without_attack_threshold', default=None, type=int)

    args = parser.parse_args()

    # set logging level
    import logging
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)

    # set experiment name
    nondefault_params = [f'comment={args.comment}']
    for arg, value in sorted(vars(args).items()):
        if arg == 'comment':
            continue
        default = parser.get_default(arg)
        if value != default:
            nondefault_params.append(f'{arg}={value}')
    experiment_name = ','.join(nondefault_params)
    experiment_id = hashlib.md5(experiment_name.encode()).hexdigest()[:8]
    experiment_name += f'--experiment_id={experiment_id}'
    logging.info(f'experiment_name: "{experiment_name}"')

    # create the environment
    def make_env(seed):
        logging.debug(f"seed={seed}")
        env = make_game_env(
                game=args.game,
                action_space=args.action_space,
                render_mode=args.render_mode,
                skip_boring_frames=True,
                task_regex=args.task_regex,
                seed=seed,
                frames_without_attack_threshold=args.frames_without_attack_threshold,
                fast_termination=True,
                center_player=args.center_player,
                reset_method=args.reset_method,
                reset_state=args.reset_state,
                )
        env = TerminatedTimeLimit(env, max_episode_steps=30*60*5)
        env = StochasticFrameSkip(env, 4, 0.25, seed=seed)
        return env

    # NOTE:
    # we use a simple LCG to generate non-sequential seeds for each environment;
    # this ensures that changing args.seed=0 to args.seed=1 doesn't result in many
    # environments with duplicate seeds
    train_env = SubprocVecEnv([lambda seed=seed: make_env((seed*134775813+1)%(2**31)) for seed in range(args.n_env)])
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
    elif args.policy == 'ChunkedObjectCnn':
        policy_kwargs['features_extractor_class'] = ChunkedObjectCnn
        policy_kwargs['features_extractor_kwargs'] = {
            'pooling': args.pooling,
            'features_dim': args.features_dim,
            }
    elif args.policy == 'ObjectEmbedding':
        policy_kwargs['features_extractor_class'] = ObjectEmbedding
        policy_kwargs['features_extractor_kwargs'] = {
            'pooling': args.pooling,
            'features_dim': args.features_dim,
            }
    elif args.policy == 'ObjectEmbeddingWithDiff':
        policy_kwargs['features_extractor_class'] = ObjectEmbeddingWithDiff
        policy_kwargs['features_extractor_kwargs'] = {
            'pooling': args.pooling,
            'features_dim': args.features_dim,
            }
    elif args.policy == 'EventExtractor':
        policy_kwargs['features_extractor_class'] = EventExtractor
        policy_kwargs['features_extractor_kwargs'] = {
            'ObjectCNN_kwargs': {
                'pooling': args.pooling,
                'features_dim': args.features_dim,
                },
            }
    elif args.policy == 'ContinuousEventExtractor':
        policy_kwargs['features_extractor_class'] = ContinuousEventExtractor
        policy_kwargs['features_extractor_kwargs'] = {
            'ObjectCNN_kwargs': {
                'pooling': args.pooling,
                'features_dim': args.features_dim,
                },
            'events_dim': args.events_dim,
            'rewards_dim': args.rewards_dim,
            }
    elif args.policy == 'ExtractorManager':
        objects_extractor = globals()[args.objects_extractor]
        revents_extractor = globals()[args.revents_extractor]
        policy_kwargs['features_extractor_class'] = ExtractorManager
        policy_kwargs['features_extractor_kwargs'] = {
            'objects_extractor': objects_extractor,
            'objects_kwargs': {
                'pooling': args.pooling,
                'features_dim': args.features_dim,
                'difference_dim': args.difference_dim,
                },
            'revents_extractor': revents_extractor,
            'revents_kwargs': {
                'revents_dim': args.events_dim,
                },
            }
        assert args.events_dim == args.rewards_dim
    else:
        assert False, "bad args.policy"


    if args.alg == 'ppo':
        model = stable_baselines3.PPO(
            policy=ActorCriticCnnPolicy,
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
    elif args.alg == 'dqn':
        model = stable_baselines3.DQN(
            policy=policy,
            env=train_env,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            gamma=args.gamma,
            verbose=1,
            tensorboard_log=args.log_dir,
            policy_kwargs=policy_kwargs,
            seed=args.seed,
        )

    reset_num_timesteps = True
    if args.warmstart:
        warmstart_path = args.warmstart

        # if warmstart is an 8 char hash value,
        # then assume it is an experiment_id;
        # then search the log folder for this experiment_id
        # and adjust warmstart_path to the matching path
        pattern = r'^[0-9a-f]{8}$'
        match = re.match(pattern, args.warmstart)
        if match:
            globstr = args.log_dir + '/*experiment_id=' + match.group() + '*'
            logging.debug(f"globstr={globstr}")
            filepath = None
            for filepath in glob.glob(globstr):
                logging.debug(f"filepath={filepath}")
                warmstart_path = filepath + '/model.zip'

        logging.info(f"warmstart_path={warmstart_path}")
        assert os.path.isfile(warmstart_path)
        reset_num_timesteps = False
        warmstart = stable_baselines3.PPO.load(warmstart_path)
        model.policy = warmstart.policy
        model.num_timesteps = warmstart.num_timesteps

    # train the model
    model.learn(
        total_timesteps=args.total_timesteps,
        log_interval=1,
        tb_log_name=experiment_name,
        reset_num_timesteps=reset_num_timesteps,
        callback = [
            SaveOnBestTrainingRewardCallback(),
            HParamCallback(),
            TensorboardCallback(record_every=None if args.disable_video or args.render_mode=='human' else 1),
            ],
    )

    # cleanly close resources
    train_env.close()


if __name__ == "__main__":
    main()

