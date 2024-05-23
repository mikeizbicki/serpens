from pysad.models import xStream


class NoveltyReward(gymnasium.Wrapper):
    '''
    '''
    def __init__(
            self,
            ):
        super().__init__(env)
        self.model = None

    def reset(self, **kwargs):
        results = super().reset(**kwargs)
        del self.model
        self.model = xStream()
        return results

    def step(self, action):
        self.env.step(action, want_render=want_render)
        observation, reward, terminated, truncated, info = super().step(action)
        score = model.fit_score_partial(observation)
        new_reward = reward + score
        return observation, new_reward, terminated, truncated, info

