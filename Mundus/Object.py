from collections import OrderedDict, defaultdict, Counter
import hashlib
import logging
import numpy as np
import gymnasium
import struct
import time
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
    def __init__(self, observation_format, max_objects=65):
        self.items = defaultdict(lambda: {})
        self.events = {}
        self.info = {}
        self.columns = set()
        self.max_objects = max_objects
        self.dtype_discrete = np.uint8
        self.dtype_continuous = np.float16
        self.observation_format = observation_format
        self.max_discrete = 256

    def __setitem__(self, name, val):
        self.items[name] = val
        for k, v in val.items():
            self.columns.add(k)
            # NOTE: we convert all integer values away from numpy types to python int
            # because the numpy types are unsigned and small;
            # this can result in hard-to-debug integer overflow problems
            if isinstance(v, str):
                pass
            elif not isinstance(v, (np.floating, float)):
                val[k] = int(v)

    def to_observation(self):
        item_list = list(self.items.values())

        # if we have exceeded the maximum number of objects,
        # then truncate the list
        if len(item_list) > self.max_objects:
            item_list = item_list[:self.max_objects]
            logging.warning(f'len(self.items) [={len(self.items)}] >= self.max_objects [={self.max_objects}]; truncating')

        # observations have a weird structure due to:
        # (1) backwards compatibility with previous code,
        # (2) limitations in SB3 library;
        # the following code converts the item_list into the appropriate structure

        # group all of the item's keys into appropriate observation keys;
        items_formatted = []
        for val in item_list:
            observation = {}
            for observation_key in self.observation_format:
                # FIXME:
                # the % in the 'discrete' case exists only for backwards compatibility
                if 'discrete' in observation_key:
                    arr = [val[k]%self.max_discrete for k in self.observation_format[observation_key]]
                else:
                    arr = [val[k] for k in self.observation_format[observation_key]]
                observation[observation_key] = arr
            items_formatted.append(observation)

        # pad the observation space
        # (SB3 does not like variable sized observations)
        empty_item = {k: ['' if 'string' in k else 0] * len(self.observation_format[k]) for k in self.observation_format}
        items_formatted += [empty_item] * (self.max_objects - len(item_list))
        assert len(items_formatted) == self.max_objects

        # restructure the items formatting from a list of dicts to an dict of lists
        observation = {}
        for observation_key in self.observation_format:
            if 'discrete' in observation_key:
                arr = np.array([x[observation_key] for x in items_formatted], dtype=np.uint8)
            elif 'string' in observation_key:
                # FIXME:
                # str_hash_as_float is a janky work around to the fact that SB3 doesn't like Text spaces
                arr = np.array([[str_hash_as_float(s) for s in x[observation_key]] for x in items_formatted], dtype=np.float32)
            else:
                arr = np.array([x[observation_key] for x in items_formatted], dtype=self.dtype_continuous)
            observation[observation_key] = arr

        # compute event information
        events_array = np.array([v for k, v in sorted(self.events.items())])
        events_array = np.clip(events_array, -1, 1)
        observation['events'] = events_array

        # compute screen information
        observation['screen'] = self.screen
        #observation['screen'] /= 255 # normalize

        # emulator stores images as HxWxC,
        # but pytorch will process the image as CxHxW
        observation['screen'] = np.moveaxis(observation['screen'], -1, 0)

        return observation

    def get_observation_space(self):
        observation = self.to_observation()
        d = {}
        for k in observation:
            if 'discrete' in k:
                d[k] = gymnasium.spaces.Box(0, 255, observation[k].shape, self.dtype_discrete)
            elif 'string' in k:
                d[k] = gymnasium.spaces.Box(0, 255, observation[k].shape, np.uint64)
            elif 'screen' in k:
                d[k] = gymnasium.spaces.Box(0, 255, observation[k].shape, np.uint8)
            else:
                d[k] = gymnasium.spaces.Box(-1, 1, observation[k].shape, self.dtype_continuous)
        space = gymnasium.spaces.Dict(d)
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


################################################################################
# MARK: New Code for Sprite Generic_NES
################################################################################

class ChunkedObjectCnn(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gymnasium.Space,
        features_dim: int = 64,
        embedding_dim: int = 64,
        pooling: str = 'mean',
    ) -> None:
        super().__init__(observation_space, features_dim)

        self.num_continuous = observation_space['objects_continuous'].shape[1]
        self.num_discrete = observation_space['objects_discrete_id'].shape[1]
        self.discrete_max = int(np.max(observation_space['objects_discrete'].high))
        self.embeddings = nn.ModuleList()
        for size in range(self.num_discrete):
            self.embeddings.append(ChunkedEmbedding(
                embedding_dim = embedding_dim,
                chunk_size = self.discrete_max+1, # +1 to account for 0 index
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
            out['objects_string_chunk'] *= 0
            for k in out:
                #print(f"out[k].dtype={out[k].dtype}")
                #out[k] = torch.as_tensor(out[k], dtype=torch.float16)[None]
                out[k] = torch.as_tensor(out[k])[None]
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
            idxs = observations['objects_discrete_id'][:, :, i].int()
            #print(f"idxs.shape={idxs.shape}")
            #print(f"observations={observations}")
            #print(f"observations['objects_string_chunk'].dtype={observations['objects_string_chunk'].dtype}")
            chunk_idxs = observations['objects_string_chunk'][:, :, i]
            #print(f"chunk_idxs={chunk_idxs}")
            #print(f"chunk_idxs.shape={chunk_idxs.shape}")
            #print(f"chunk_idxs={chunk_idxs}")
            #asd
            embeddings = embedding(idxs, chunk_idxs)
            embeds.append(embeddings)
        ret = torch.concat([observations['objects_continuous']] + embeds, dim=2)
        ret = ret.transpose(1,2)
        return ret

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        #print(f"observations={observations}")
        ret = self._embed_observations(observations)
        ret = self.cnn(ret)
        ret = self.pool(ret)
        ret = self.linear(ret)
        return ret


class ObjectEmbedding(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gymnasium.Space,
        embedding_dim: int = 64,
        features_dim: int = 64,
        pooling: str = 'mean',
        **kwargs
    ) -> None:
        super().__init__(observation_space, features_dim)

        num_continuous = observation_space['objects_continuous'].shape[1]
        n_features = num_continuous + embedding_dim

        num_ids = int(np.max(observation_space['objects_discrete'].high)) + 1 # +1 to include 0 id
        self.embedding = ChunkedEmbedding(embedding_dim, num_ids)

        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, features_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(features_dim, features_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(features_dim, features_dim, kernel_size=1),
            nn.ReLU(),
        )

        if pooling == 'lstm':
            self.pool = LSTMPool(total_features, total_features, batch_first=True, bidirectional=True)
        elif pooling == 'max':
            self.pool = MaxPool()
        elif pooling == 'mean':
            self.pool = MeanPool()
        else:
            raise ValueError(f'pooling type "{pooling}" not supported')

        self.linear = nn.Sequential(
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
            )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        idxs = observations['objects_discrete_id'][:, :, 0].int()
        chunk_idxs = observations['objects_string_chunk'][:, :, 0]
        embeds = self.embedding(idxs, chunk_idxs)
        cont = observations['objects_continuous']
        ret = torch.concat([cont, embeds], dim=2)
        ret = ret.transpose(1,2)
        ret = self.cnn(ret)
        ret = self.pool(ret)
        ret = self.linear(ret)
        return ret

class ObjectEmbeddingWithDiff(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gymnasium.Space,
        embedding_dim: int = 64,
        features_dim: int = 64,
        difference_dim: int = 8,
        pooling: str = 'mean',
    ) -> None:
        super().__init__(observation_space, features_dim)

        num_continuous = observation_space['objects_continuous'].shape[1]
        n_features = num_continuous + embedding_dim + difference_dim * num_continuous

        num_ids = int(np.max(observation_space['objects_discrete'].high)) + 1 # +1 to include 0 id
        self.embedding = ChunkedEmbedding(embedding_dim, num_ids)
        self.diff_weights = ChunkedEmbedding(difference_dim)

        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, features_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(features_dim, features_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(features_dim, features_dim, kernel_size=1),
            nn.ReLU(),
        )

        if pooling == 'lstm':
            self.pool = LSTMPool(total_features, total_features, batch_first=True, bidirectional=True)
        elif pooling == 'max':
            self.pool = MaxPool()
        elif pooling == 'mean':
            self.pool = MeanPool()
        else:
            raise ValueError(f'pooling type "{pooling}" not supported')

        self.linear = nn.Sequential(
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
            )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        idxs = observations['objects_discrete_id'][:, :, 0].int()
        chunk_idxs = observations['objects_string_chunk'][:, :, 0]
        embeds = self.embedding(idxs, chunk_idxs)

        diff_weights = self.diff_weights(idxs, chunk_idxs)
        #print(f"diff_weights.shape={diff_weights.shape}")

        cont = observations['objects_continuous']
        #print(f"cont.shape={cont.shape}")

        diff_side1 = torch.einsum('abc,abd->abcd', cont, diff_weights)
        #print(f"diff_side1.shape={diff_side1.shape}")
        diff = diff_side1 - cont.unsqueeze(3)
        diff = diff.reshape(diff.shape[0], diff.shape[1], diff.shape[2] * diff.shape[3])
        #print(f"diff.shape={diff.shape}")

        ret = torch.concat([cont, diff, embeds], dim=2)
        ret = ret.transpose(1,2)
        ret = self.cnn(ret)
        ret = self.pool(ret)
        ret = self.linear(ret)
        return ret


class Revents_Linear(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        revents_dim,
        ):
        super().__init__(observation_space, features_dim=revents_dim*2)
        self.extractors = nn.ModuleDict({
            'events': nn.Linear(observation_space['events'].shape[0], revents_dim, bias=False),
            'rewards': nn.Linear(observation_space['rewards'].shape[0], revents_dim, bias=False),
            })

    def forward(self, observations):
        encoded_tensor_list = [
            self.extractors['events'](observations['events']),
            self.extractors['rewards'](observations['rewards']),
            ]
        return torch.cat(encoded_tensor_list, dim=1)


class ScreenCNN(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        features_dim = 64,
        ):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 8, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations):
        return self.linear(self.cnn(observations))


class ExtractorManager(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        objects_extractor,
        revents_extractor,
        screen_extractor = None,
        **kwargs,
    ) -> None:
        # When calling super().__init__, we are supposed to know the output features_dim.
        # But we do not know features_dim here before going over all the items;
        # so put something there temporarily, and update features_dim later.
        # (This trick is copied from the SB3 MultiPolicy implementation.)
        super().__init__(observation_space, features_dim=1)

        # create the feature networks
        self.extractors = nn.ModuleDict({})
        if objects_extractor is not None:
            self.extractors['objects'] = objects_extractor(observation_space, **kwargs['objects_kwargs'])
        if revents_extractor is not None:
            self.extractors['revents'] = revents_extractor(observation_space, **kwargs['revents_kwargs'])
        if screen_extractor is not None:
            self.extractors['screen'] = screen_extractor(observation_space['screen'], **kwargs['screen_kwargs'])

        # Update the features dim manually
        self._features_dim = sum([x.features_dim for x in self.extractors.values()])

    def forward(self, observations: TensorDict) -> torch.Tensor:
        #tensors = [x(observations) for x in self.extractors]
        features = []
        if 'objects' in self.extractors:
            features.append(self.extractors['objects'](observations))
        if 'revents' in self.extractors:
            features.append(self.extractors['revents'](observations))
        if 'screen' in self.extractors:
            obs = observations['screen']
            features.append(self.extractors['screen'](obs))
        return torch.cat(features, dim=1)


################################################################################
# MARK: helper modules
################################################################################

class ChunkedEmbedding(nn.Module):
    def __init__(self, embedding_dim, chunk_size=512):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.chunk_size = chunk_size
        self.chunk_embeddings = nn.ModuleDict()

    def forward(self, ids, chunk_ids):

        assert ids.shape == chunk_ids.shape, f"Shape mismatch: ids has shape {ids.shape} but chunk_ids has shape {chunk_ids.shape}"

        # we flatten the input tensors so that we can operate on tensors of any shape
        original_shape = list(ids.shape)
        flat_ids = ids.reshape(-1)
        flat_chunk_ids = chunk_ids.reshape(-1)

        flat_output = torch.zeros(flat_ids.shape[0], self.embedding_dim, device=ids.device)

        # loop over each *unique* chunk_id for efficiency purposes;
        # in a typical application, we expect a small number of chunk_ids,
        # and so this for loop will run only 1-2 times,
        # and inside we perform vectorized update operations
        unique_chunk_ids = set(chunk_ids.flatten().tolist())
        for chunk_id in unique_chunk_ids:

            # FIXME:
            # SB3 converts all the inputs into torch.float32 for some reason;
            # this is my hacky way of getting back to a string;
            chunk_id_str = str(chunk_id).replace('.', '_')

            # if this is the first time we've seen the chunk_id,
            # we create a new embedding
            if chunk_id_str not in self.chunk_embeddings:
                logging.info(f'ChunkedEmbedding: chunk_id_str={chunk_id_str} not in self.chunk_embeddings; creating new embedding')
                logging.info(f'ChunkedEmbedding: keys={self.chunk_embeddings.keys()}')
                self.chunk_embeddings[chunk_id_str] = nn.Embedding(self.chunk_size, self.embedding_dim).to(ids.device)

            # update the output tensor with the embeddings from this chunk
            mask = flat_chunk_ids == chunk_id
            if torch.any(mask):
                flat_output[mask] = self.chunk_embeddings[chunk_id_str](flat_ids[mask])

        # restore shape
        output_shape = original_shape + [self.embedding_dim]
        output = flat_output.reshape(output_shape)

        return output

    def _load_from_state_dict(self, state_dict, *args, **kwargs):
        '''
        This function is used internally by pytorch to load a Module from a file.
        Normally it is not necessary to overload this function,
        but we must overload it because we are modifying self.chunk_embeddings in forward().
        ModuleDict will raise an error if we load a dictionary with keys that it does not know about.
        Normally these keys are only specified in the __init__ function,
        but we are dynamically adding keys in the forward function.
        To ensure these keys get added to the model loading and do not result in an error,
        we add the keys to the ModuleDict before calling the built-in load function.
        '''
        for key in state_dict.keys():
            chunk_id_str = key.split('.')[-2]
            if chunk_id_str not in self.chunk_embeddings:
                logging.debug(f'ChunkedEmbedding: chunk_id_str={chunk_id_str} not in self.chunk_embeddings; creating new embedding')
                self.chunk_embeddings[chunk_id_str] = nn.Embedding(self.chunk_size, self.embedding_dim)
        super()._load_from_state_dict(state_dict, *args, **kwargs)


################################################################################
# MARK: debug utils
################################################################################

def str_hash_as_float(s):
    hash_object = hashlib.blake2b(s.encode(), digest_size=4)
    hash_as_float = struct.unpack('f', hash_object.digest())[0]
    return hash_as_float
