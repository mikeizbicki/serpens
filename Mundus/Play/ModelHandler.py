import multiprocessing
import queue
import os
import time
import warnings

import setproctitle

# setup logging
import logging
logger = logging.getLogger(__name__)


class ModelHandler():
    '''
    This class is responsible for loading a model and using the model to generate actions.
    This functionality is not normally abstracted out into a separate class,
    but instead included directly in the main game loop.
    I have moved it into a class to make it easy to swap the forked model handler with the standard model handler.
    '''
    def __init__(self, model_path, env):
        from stable_baselines3 import PPO
        custom_objects = {
            'observation_space': env.observation_space,
            'action_space': env.action_space,
            }
        self.model = PPO.load(model_path, custom_objects=custom_objects)

    def get_action(self, observation):
        action, _states = self.model.predict(observation, deterministic=True)
        return action


class ModelHandlerForked():
    '''
    This class reimplements the interface of ModelHandler,
    but all processsing is done in a separate process.
    This enables a multi-cpu machine to perform the model computations
    on a different CPU than the environment.
    The downside of using this class is that the action returned by .get_action
    will be computed on an observation with at least a one frame delay.
    '''
    def __init__(self, model_path, env, sleep_time=0.0):
        from stable_baselines3 import PPO

        # qin stores messages that go to the worker;
        # qout stores messages that come from the worker;
        # they are of size 1 to ensure syncronization between the worker and parent process;
        # if one process gets ahead of the other process,
        # it will block trying to add a second entry into the queue
        self.qin = multiprocessing.Queue(maxsize=1)
        self.qout = multiprocessing.Queue(maxsize=1)
        self.qout_lock = multiprocessing.Lock()

        # start the emulator process
        def model_worker():
            current_name = multiprocessing.current_process().name
            setproctitle.setproctitle(f'{current_name}: model_worker')

            # NOTE:
            # loading the models can cause a large number of warnings;
            # this with block prevents those warnings from being displayed
            logger.info('loading model')
            model = None
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from stable_baselines3 import PPO
                custom_objects = {
                    'observation_space': env.observation_space,
                    'action_space': env.action_space,
                    }
                model = PPO.load(model_path, custom_objects=custom_objects)

            # set the first action to be empty;
            # we must do this after the model is loaded
            # so that the main loop doesn't start playing before the model is loaded
            action = env.action_space.sample() * 0
            self.qout.put(action)

            # set this process to have the lowest priority;
            # the user won't notice if this process lags,
            # but they will notice if this process interrupts the emulator or main processes
            os.nice(19)

            # the worker will loop forever;
            # it blocks on the qin.get() line waiting for a new observation;
            # then it puts the appropriate action in the queue
            logger.info('entering model_worker main loop')
            while True:
                observation = self.qin.get()
                if model is not None:
                    action, _states = model.predict(observation, deterministic=True)
                else:
                    action = env.action_space.sample() * 0
                with self.qout_lock:
                    self.qout.get()
                    self.qout.put(action)
                # wait before selecting another action;
                # in the meantime, the same action will remain in qout,
                # and so the main loop will just repeat the action;
                # this should have minimal adverse effects on model performance;
                # the purpose of waiting is to reduce CPU load for low-resource devices;
                time.sleep(sleep_time)
        p = multiprocessing.Process(name='serpens: model_worker', target=model_worker)
        p.start()
    
    def get_action(self, observation):
        # get the next action from the model
        # (which is the action from the previous iteration's observation)
        with self.qout_lock:
            action = self.qout.get()
            self.qout.put(action)

        # send the observation to the model_worker process
        # if there is already an observation,
        # that means the model is running slowly and falling behind the main process;
        # this happens when the model is larger than the hardware can run at 60fps;
        # we overwrite the existing observation with the latest observation
        # so that the model will always make decision on the latest observation
        # even if it has to skip frames
        try:
            self.qin.get_nowait()
        except queue.Empty:
            pass
        self.qin.put(observation)

        # only return after the interprocess communication is done
        return action


