import pprint
import logging

# set logging level
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

import gymnasium
from gymnasium.wrappers import *
import ray
from ray.tune.logger import TBXLoggerCallback
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch.nn as nn

context = ray.init()
logging.info(f"context.dashboard_url={context.dashboard_url}")

from Mundus.Wrappers import *
from Mundus.Zelda import *

#import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning)


import logging
from typing import TYPE_CHECKING, Dict

import numpy as np

from ray.air.constants import TRAINING_ITERATION
from ray.tune.logger.logger import _LOGGER_DEPRECATION_WARNING, Logger, LoggerCallback
from ray.tune.result import TIME_TOTAL_S, TIMESTEPS_TOTAL
from ray.tune.utils import flatten_dict
from ray.util.annotations import Deprecated, PublicAPI
from ray.util.debug import log_once

VALID_SUMMARY_TYPES = [int, float, np.float32, np.float64, np.int32, np.int64]

class CustomTBXLogger(Logger):
    """TensorBoardX Logger.

    Note that hparams will be written only after a trial has terminated.
    This logger automatically flattens nested dicts to show on TensorBoard:

        {"a": {"b": 1, "c": 2}} -> {"a/b": 1, "a/c": 2}
    """

    VALID_HPARAMS = (str, bool, int, float, list, type(None))
    VALID_NP_HPARAMS = (np.bool_, np.float32, np.float64, np.int32, np.int64)

    def _init(self):
        try:
            from tensorboardX import SummaryWriter
        except ImportError:
            if log_once("tbx-install"):
                logger.info('pip install "ray[tune]" to see TensorBoard files.')
            raise
        self._file_writer = SummaryWriter(self.logdir, flush_secs=30)
        self.last_result = None

    def on_result(self, result: Dict):
        step = result.get(TIMESTEPS_TOTAL) or result[TRAINING_ITERATION]
        print(f"step={step}")

        tmp = result.copy()
        for k in ["config", "pid", "timestamp", TIME_TOTAL_S, TRAINING_ITERATION]:
            if k in tmp:
                del tmp[k]  # not useful to log these

        flat_result = flatten_dict(tmp, delimiter="/")
        valid_result = {}

        for attr, value in flat_result.items():
            if 'custom_metrics/' in attr:
                full_attr = attr.split('custom_metrics/')[-1]
            else:
                full_attr = 'ray/' + attr
                # FIXME: there are probably interesting stats in here to pull out
            if isinstance(value, tuple(VALID_SUMMARY_TYPES)) and not np.isnan(value):
                valid_result[full_attr] = value
                self._file_writer.add_scalar(full_attr, value, global_step=step)
            elif (isinstance(value, list) and len(value) > 0) or (
                isinstance(value, np.ndarray) and value.size > 0
            ):
                valid_result[full_attr] = value

                # Must be a single image.
                if isinstance(value, np.ndarray) and value.ndim == 3:
                    self._file_writer.add_image(
                        full_attr,
                        value,
                        global_step=step,
                    )
                    continue

                # Must be a batch of images.
                if isinstance(value, np.ndarray) and value.ndim == 4:
                    self._file_writer.add_images(
                        full_attr,
                        value,
                        global_step=step,
                    )
                    continue

                # Must be video
                if isinstance(value, np.ndarray) and value.ndim == 5:
                    self._file_writer.add_video(
                        full_attr, value, global_step=step, fps=20
                    )
                    continue

                try:
                    self._file_writer.add_histogram(full_attr, value, global_step=step)
                # In case TensorboardX still doesn't think it's a valid value
                # (e.g. `[[]]`), warn and move on.
                except (ValueError, TypeError):
                    if log_once("invalid_tbx_value"):
                        logger.warning(
                            "You are trying to log an invalid value ({}={}) "
                            "via {}!".format(full_attr, value, type(self).__name__)
                        )

        self.last_result = valid_result
        self._file_writer.flush()

    def flush(self):
        if self._file_writer is not None:
            self._file_writer.flush()

    def close(self):
        if self._file_writer is not None:
            if self.trial and self.trial.evaluated_params and self.last_result:
                flat_result = flatten_dict(self.last_result, delimiter="/")
                scrubbed_result = {
                    k: value
                    for k, value in flat_result.items()
                    if isinstance(value, tuple(VALID_SUMMARY_TYPES))
                }
                self._try_log_hparams(scrubbed_result)
            self._file_writer.close()

    def _try_log_hparams(self, result):
        # TBX currently errors if the hparams value is None.
        flat_params = flatten_dict(self.trial.evaluated_params)
        scrubbed_params = {
            k: v for k, v in flat_params.items() if isinstance(v, self.VALID_HPARAMS)
        }

        np_params = {
            k: v.tolist()
            for k, v in flat_params.items()
            if isinstance(v, self.VALID_NP_HPARAMS)
        }

        scrubbed_params.update(np_params)

        removed = {
            k: v
            for k, v in flat_params.items()
            if not isinstance(v, self.VALID_HPARAMS + self.VALID_NP_HPARAMS)
        }
        if removed:
            logger.info(
                "Removed the following hyperparameter values when "
                "logging to tensorboard: %s",
                str(removed),
            )

        from tensorboardX.summary import hparams

        try:
            experiment_tag, session_start_tag, session_end_tag = hparams(
                hparam_dict=scrubbed_params, metric_dict=result
            )
            self._file_writer.file_writer.add_summary(experiment_tag)
            self._file_writer.file_writer.add_summary(session_start_tag)
            self._file_writer.file_writer.add_summary(session_end_tag)
        except Exception:
            logger.exception(
                "TensorboardX failed to log hparams. "
                "This may be due to an unsupported type "
                "in the hyperparameter values."
            )

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from collections import Counter
from typing import Dict
import numpy as np

class CustomMetricsCallback(DefaultCallbacks):
    def __init__(self, legacy_callbacks_dict: dict = None):
        #super().__init__(legacy_callbacks_dict)
        super().__init__()
        self.num_tasks = Counter()

    def on_episode_start(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: EpisodeV2,
        env_index: int,
        **kwargs,
    ):
        # Initialize any episode-specific variables here
        episode.user_data["episode_steps"] = 0

    def on_episode_step(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        episode: EpisodeV2,
        env_index: int,
        **kwargs,
    ):
        episode.user_data["episode_steps"] += 1

        # If you want to capture screens (similar to your video recording logic)
        # Note: You'll need to modify this based on your environment's specifics
        if hasattr(base_env, "get_images"):
            if "screens" not in episode.user_data:
                episode.user_data["screens"] = []
            screen = base_env.get_images()[0]
            episode.user_data["screens"].append(screen.transpose(2, 0, 1))

    def on_episode_end(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: EpisodeV2,
        env_index: int,
        **kwargs,
    ):
        info_dict = episode.last_info_for()
        if info_dict is None:
            return

        # Record all summary metrics
        for key, value in info_dict.items():
            if key.startswith('summary_'):
                metric_name = f"summary/{key[8:]}"  # Remove 'summary_' prefix
                episode.custom_metrics[metric_name] = value

            elif key.startswith('reward_'):
                metric_name = f"reward/{key[7:]}"  # Remove 'reward_' prefix
                episode.custom_metrics[metric_name] = value

            elif key.startswith('task_success_'):
                metric_name = f"task_success/{key[12:]}"  # Remove 'task_success_' prefix
                episode.custom_metrics[metric_name] = value

            elif key.startswith('task_count_'):
                task_name = key[11:]  # Remove 'task_count_' prefix
                self.num_tasks[task_name] += 1
                episode.custom_metrics[f"task_count/{task_name}"] = self.num_tasks[task_name]

        # Add episode count metric
        episode.custom_metrics["episode/num_episodes"] = episode.episode_id

        # Handle video recording if needed
        if "screens" in episode.user_data and len(episode.user_data["screens"]) > 0:
            screens = np.array(episode.user_data["screens"])
            # Note: RLlib doesn't have built-in video logging like SB3
            # You might want to use wandb or custom video logging here
            # Or save videos to disk using something like imageio

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    debug = parser.add_argument_group('debug')
    debug.add_argument("--log_dir", default='log')
    debug.add_argument("--disable_video", action='store_true')
    debug.add_argument("--render_mode", default='rgb_array')
    debug.add_argument("--comment")
    debug.add_argument("--total_timesteps", default=1_000_000_000_000, type=int)

    hyperparameters = parser.add_argument_group('hyperparameters')
    hyperparameters.add_argument('--policy', choices=['MlpPolicy', 'CnnPolicy', 'ObjectCnn'], default='ObjectCnn')
    hyperparameters.add_argument('--pooling', choices=['lstm', 'mean', 'max'], default='mean')
    hyperparameters.add_argument('--alg', choices=['ppo', 'dqn'], default='ppo')
    hyperparameters.add_argument('--task', default='attack')
    #hyperparameters.add_argument('--task', default=None)
    hyperparameters.add_argument('--state', default='spiders_lowhealth_01*.state')
    hyperparameters.add_argument('--net_arch', type=int, nargs='*', default=[])
    hyperparameters.add_argument('--features_dim', type=int, default=64)
    hyperparameters.add_argument('--lr', type=float, default=3e-4)
    hyperparameters.add_argument('--gamma', type=float, default=0.99)
    hyperparameters.add_argument('--n_env', type=int, default=3)
    hyperparameters.add_argument('--n_steps', type=int, default=128)
    hyperparameters.add_argument('--batch_size', type=int, default=32)
    hyperparameters.add_argument('--seed', type=int, default=0)
    hyperparameters.add_argument('--warmstart', default=None)
    hyperparameters.add_argument('--action_space', default='DISCRETE')

    args = parser.parse_args()

    # set experiment name
    import datetime
    starttime = datetime.datetime.now().isoformat()

    arch_string = '-'.join([str(i) for i in args.net_arch])
    policy_params = ''
    if args.policy == 'ObjectCnn':
        policy_params = f',pooling={args.pooling}'

    experiment_name = ''
    if args.comment is not None:
        experiment_name = args.comment + '--'
    experiment_name += f'task={args.task},action_space={args.action_space},state={args.state},policy={args.policy}{policy_params},net_arch={arch_string},{args.features_dim},alg={args.alg},lr={args.lr},gamma={args.gamma},n_env={args.n_env},n_steps={args.n_steps},batch_size={args.batch_size},seed={args.seed}'
    logging.info(f'experiment_name: "{experiment_name}"')

    # register the environment with ray
    def make_env(config):
        seed = (config.worker_index*134775813+1)%(2**31)
        env = make_zelda_env(
                action_space=args.action_space,
                render_mode=args.render_mode,
                skip_boring_frames=True,
                task=args.task,
                seed=seed,
                )
        env = TimeLimit(env, max_episode_steps=30*60*5)
        env = StochasticFrameSkip(env, 4, 0.25, seed=seed)
        env = RandomStateReset(env, path='custom_integrations/Zelda-Nes', globstr=args.state, seed=seed)
        return env
    from ray import tune
    tune.register_env('myzelda', make_env)

    # configure the training run
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment('myzelda')
        .callbacks(CustomMetricsCallback)
        .env_runners(num_env_runners=args.n_env)
        #.training(model={"custom_model": "custom_model"})
        )
    config["logger_config"] = {
        #"type": "ray.tune.logger.TBXLogger",
        "type": CustomTBXLogger,
        "logdir": args.log_dir + '/' + experiment_name,
        #"custom_metrics_reporter": custom_metrics_reporter,
        }

    algo = config.build()

    # setup logging
    #tbx_logger = TBXLoggerCallback(
        #log_dir=args.log_dir + '/' + experiment_name,
    #)
    #algo.add_callback(tbx_logger)

    # training loop
    for i in range(1000_000_000_000):
        logging.info(f"i={i}")
        result = algo.train()
        result.pop('config')
        pprint.pprint(result)

        if i % 5 == 0:
            checkpoint_dir = algo.save()
            print(f"Checkpoint saved in directory {checkpoint_dir}")
