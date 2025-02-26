from collections import OrderedDict, defaultdict, Counter
import logging
import numpy as np
import gymnasium
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import TensorDict


class LSTMPool(torch.nn.LSTM):
    def forward(self, x):
        x = x.transpose(1,2)
        out = super().forward(x)
        out = out[1][0]
        out = torch.concat([out[0, :, :], out[1, :, :]], dim=1)
        return out

class MaxPool(torch.nn.Module):
    def forward(self, x):
        out = torch.max(x, 2)[0]
        return out

class MeanPool(torch.nn.Module):
    def forward(self, x):
        out = torch.mean(x, 2)
        return out

class ObjectCnn(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gymnasium.Space,
        features_dim: int = 64,
        embedding_dim: int = 64,
        pooling: str = 'mean',
    ) -> None:
        super().__init__(observation_space, features_dim)

        self.num_continuous = observation_space['objects_continuous'].shape[1]
        self.num_discrete = observation_space['objects_discrete'].shape[1]
        self.discrete_max = int(np.max(observation_space['objects_discrete'].high))
        self.embeddings = nn.ModuleList()
        for size in range(self.num_discrete):
            self.embeddings.append(nn.Embedding(
                num_embeddings = self.discrete_max+1, # +1 to account for 0 index
                embedding_dim = embedding_dim,
                ))

        n_features = self.num_discrete*embedding_dim + self.num_continuous

        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, features_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(features_dim, features_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(features_dim, features_dim, kernel_size=1),
            nn.ReLU(),
        )
        
        if pooling == 'lstm':
            self.pool = LSTMPool(features_dim, features_dim, batch_first=True, bidirectional=True)
        elif pooling == 'max':
            self.pool = MaxPool()
        elif pooling == 'mean':
            self.pool = MeanPool()
        else:
            raise ValueError(f'pooling type "{pooling}" not supported')

        # Compute shape by doing one forward pass
        with torch.no_grad():
            out = observation_space.sample()
            for k in out:
                out[k] = torch.as_tensor(out[k], dtype=torch.float16)[None]
            out = self._embed_observations(out)
            out = self.cnn(out)
            out = self.pool(out)
            n_flatten = out.shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
            )

    def _embed_observations(self, observations):
        embeds = []
        for i, embedding in enumerate(self.embeddings):
            idxs = observations['objects_discrete'][:, :, i].int()
            embeddings = embedding(idxs)
            embeds.append(embeddings)
        ret = torch.concat([observations['objects_continuous']] + embeds, dim=2)
        ret = ret.transpose(1,2)
        return ret

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        ret = self._embed_observations(observations)
        ret = self.cnn(ret)
        ret = self.pool(ret)
        ret = self.linear(ret)
        return ret


class EventExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        **kwargs,
    ) -> None:
        # NOTE:
        # (this trick is copied from the SB3 MultiPolicy implementation.)
        # we do not know features_dim here before going over all the items,
        # so put something there temporarily;
        # update features_dim later
        super().__init__(observation_space, features_dim=1)

        object_space = spaces.Dict({
            'objects_discrete': observation_space['objects_discrete'],
            'objects_continuous': observation_space['objects_continuous'],
            })
        objcnn = ObjectCnn(object_space, **kwargs['ObjectCNN_kwargs'])
        extractors = {
            'objects': objcnn,
            'events': nn.Flatten(),
            'rewards': nn.Flatten(),
            }
        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        from stable_baselines3.common.preprocessing import get_flattened_obs_dim
        dim_events = get_flattened_obs_dim(observation_space['events'])
        dim_rewards = get_flattened_obs_dim(observation_space['rewards'])
        self._features_dim = objcnn.features_dim + dim_events + dim_rewards

    def forward(self, observations: TensorDict) -> torch.Tensor:
        encoded_tensor_list = [
                self.extractors['objects']({
                    'objects_discrete': observations['objects_discrete'],
                    'objects_continuous': observations['objects_continuous'],
                    }),
                self.extractors['events'](observations['events']),
                self.extractors['rewards'](observations['rewards']),
            ]
        return torch.cat(encoded_tensor_list, dim=1)


class ContinuousEventExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        events_dim=32,
        rewards_dim=32,
        **kwargs,
    ) -> None:
        # NOTE:
        # (this trick is copied from the SB3 MultiPolicy implementation.)
        # we do not know features_dim here before going over all the items,
        # so put something there temporarily;
        # update features_dim later
        super().__init__(observation_space, features_dim=1)

        object_space = spaces.Dict({
            'objects_discrete': observation_space['objects_discrete'],
            'objects_continuous': observation_space['objects_continuous'],
            })
        objcnn = ObjectCnn(object_space, **kwargs['ObjectCNN_kwargs'])
        extractors = {
            'objects': objcnn,
            'events': nn.Linear(observation_space['events'].shape[0], events_dim, bias=False),
            'rewards': nn.Linear(observation_space['rewards'].shape[0], rewards_dim, bias=False),
            }
        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = objcnn.features_dim + events_dim + rewards_dim

    def forward(self, observations: TensorDict) -> torch.Tensor:
        encoded_tensor_list = [
                self.extractors['objects']({
                    'objects_discrete': observations['objects_discrete'],
                    'objects_continuous': observations['objects_continuous'],
                    }),
                self.extractors['events'](observations['events']),
                self.extractors['rewards'](observations['rewards']),
            ]
        return torch.cat(encoded_tensor_list, dim=1)


class History:
    def __init__(self):
        self.event_counts = Counter()

    def register_KnowledgeBase(self, step, kb):
        self.event_counts += kb.events


class KnowledgeBase:
    def __init__(self, keys, max_objects=24):
        self.items = defaultdict(lambda: {})
        self.events = {}
        self.columns = set()
        self.max_objects = max_objects
        self.dtype_discrete = np.uint8
        self.dtype_continuous = np.float16
        self.keys = keys
        self.max_discrete = 256
        self.info = {}

    def __setitem__(self, name, val):
        if len(self.items) >= self.max_objects:
            logging.warning(f'len(self.items) >= self.max_objects; ignoring [{name}]={val}')
        else:
            self.items[name] = val
            for k, v in val.items():
                self.columns.add(k)
                # NOTE: we convert all integer values away from numpy types to python int
                # because the numpy types are unsigned and small;
                # this can result in hard-to-debug integer overflow problems
                if not isinstance(v, (np.floating, float)):
                    val[k] = int(v)

    def to_observation(self):

        observations = []
        for item, val in self.items.items():
            observation = {
                'objects_discrete': [val[k]%self.max_discrete for k in self.keys['objects_discrete']],
                'objects_continuous': [val[k] for k in self.keys['objects_continuous']],
                }
            observations.append(observation)
        # FIXME:
        # currently we pad the observation space because SB3
        # doesn't like variable sized observations
        pad = {
            'objects_discrete': [0] * len(self.keys['objects_discrete']),
            'objects_continuous': [0] * len(self.keys['objects_continuous']),
            }
        observations += [pad] * (self.max_objects - len(observations))

        events_array = np.array([v for k, v in sorted(self.events.items())])
        events_array = np.clip(events_array, -1, 1)

        ret = {
            'objects_continuous': np.array([x['objects_continuous'] for x in observations], dtype=self.dtype_continuous),
            'objects_discrete': np.array([x['objects_discrete'] for x in observations], dtype=self.dtype_discrete),
            'events': events_array,
            }
        return ret

    def get_observation_space(self):
        observation = self.to_observation()
        space = gymnasium.spaces.Dict({
            'objects_continuous': gymnasium.spaces.Box(-1, 1, observation['objects_continuous'].shape, self.dtype_continuous),
            'objects_discrete': gymnasium.spaces.Box(0, 255, observation['objects_discrete'].shape, self.dtype_discrete),
            'events': gymnasium.spaces.Box(-1, 1, observation['events'].shape, self.dtype_continuous)
            })
        return space

    def display(self):
        columns = sorted(self.columns)
        columns_lens = [max(len(column), 5) for column in columns]
        item_len = 12 #max([len(item) for item in self.items])
        ret = ''
        ret += ' '*item_len
        for column, column_len in zip(columns, columns_lens):
            ret += f' {column:>{column_len}}'
        ret += '\n'
        for item,val in self.items.items():
            ret += f'{item:>{item_len}}'
            for column, column_len in zip(columns, columns_lens):
                s = '-'
                if column in val:
                    s = val[column]
                if isinstance(s, (np.floating, float)):
                    ret += f' {s:>+0.{column_len-3}f}'
                else:
                    ret += f' {s:>{column_len}}'
            ret += '\n'
        return ret
