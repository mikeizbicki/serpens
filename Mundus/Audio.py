import subprocess
import logging
import os
import time
import fcntl
import re
import time

import numpy as np
import pyaudio
import gymnasium
import simpleaudio as sa

from elevenlabs.client import ElevenLabs
from elevenlabs import play
import elevenlabs
from io import BytesIO
from pydub import AudioSegment
import threading

class PlayAudio(gymnasium.Wrapper):
    '''
    FIXME:
    The playback of this audio is not smooth due to the blocking interface.
    I've tried a bunch of other blocking interfaces, and this one is the best.
    The non-blocking interface (below) has lag issues.
    '''
    def __init__(self, env):
        super().__init__(env)

    def step(self, actions):
        data = self.unwrapped.em.get_audio().astype('int16')
        sa.play_buffer(data, 2, 2, 32000)


class PlayAudio_ElevenLabs(gymnasium.Wrapper):
    def __init__(self, env, ignore_ALSA_warnings=True):
        super().__init__(env)
        self.text_audio = None
        self.audio_env = self.env
        self.elevenlabs = ElevenLabs()
        self.cache_dir = '.audio_cache'
        os.makedirs(self.cache_dir, exist_ok=True)

    def register_text(self, text):
        time_start = time.time()
        try:
            with open(os.path.join(self.cache_dir, text), 'rb') as fin:
                self.audio_env.text_audio = np.repeat(np.frombuffer(fin.read(), dtype=np.int16), 2)
                self.audio_env.text_audio_pos = 0
        except FileNotFoundError:
            def thread_function():
                response = self.elevenlabs.text_to_speech.convert(
                    text=text,
                    voice_id="JBFqnCBsd6RMkjVDRZzb",
                    model_id="eleven_turbo_v2_5",
                    output_format="pcm_16000",
                    voice_settings=elevenlabs.VoiceSettings(
                        stability=0.0,
                        similarity_boost=1.0,
                        style=0.0,
                        use_speaker_boost=True,
                    ),
                )
                lag = time.time() - time_start
                logging.info(f"API lag={lag}")
                audio_stream = BytesIO()
                for i, chunk in enumerate(response):
                    lag = time.time() - time_start
                    if i == 0:
                        logging.info(f"for loop: lag={lag}")
                    if chunk:
                        audio_stream.write(chunk)
                audio_stream.seek(0)
                self.audio_env.text_audio = np.repeat(np.frombuffer(audio_stream.read(), dtype=np.int16), 2)
                self.audio_env.text_audio_pos = 0
                lag = time.time() - time_start
                logging.info(f"mp3 convert; lag={lag}")
                with open(os.path.join(self.cache_dir, text), 'wb') as fout:
                    audio_stream.seek(0)
                    fout.write(audio_stream.read())
            thread = threading.Thread(target=thread_function)
            thread.start()

        return self._recursive_register_text(self.env, text)

    def _recursive_register_text(self, env, text):
        if hasattr(env, 'register_text'):
            return env.register_text(text)
        if isinstance(env, gymnasium.Wrapper):
            return self._recursive_register_text(env.env, text)


class PlayAudio_pyaudio(gymnasium.Wrapper):
    '''
    This wrapper uses pyaudio to play the sound of the NES stable-retro environment.
    '''
    def __init__(self, env, ignore_ALSA_warnings=True):
        super().__init__(env)
        self.PlayAudio_buffer = np.zeros([0,2], dtype='int16')
        self.text_audio = None
    
        # create the audio stream
        def playing_callback(in_data, frame_count, time_info, status):
            if self.PlayAudio_buffer.shape[0] >= frame_count:
                data = self.PlayAudio_buffer[:frame_count, :]
                self.PlayAudio_buffer = self.PlayAudio_buffer[frame_count:, :]
                data = data.flatten()
                #logging.info(f"self.PlayAudio_buffer.shape={self.PlayAudio_buffer.shape}")
                return (data.tobytes(), pyaudio.paContinue)
            else:
                #logging.warning('PlayAudio: buffer underrun')
                return (b'\x00' * frame_count * 4, pyaudio.paContinue)  # 4 bytes per frame (2 channels * 2 bytes per sample)

        logging.info(f"env.unwrapped.em.get_audio_rate()={env.unwrapped.em.get_audio_rate()}")
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=2,
                rate=32000, #int(env.unwrapped.em.get_audio_rate()),
                output=True,
                stream_callback=playing_callback,
                frames_per_buffer=128,
                start=False,
                )
        #self.stream._stream.suggestedLatency = 0.1  # Try to achieve 100ms latency
        self.stream.start_stream()

    def step(self, actions):
        data = self.env.unwrapped.em.get_audio().astype('int16')
        #if self.mpg123_handle is not None:
            #text_chunk = self.mpg123_handle.read(len(data))
        if self.text_audio is not None:
            text_chunk = self.text_audio[self.text_audio_pos:self.text_audio_pos + len(data)]
            if len(text_chunk) > 0:
                self.text_audio_pos += len(data)
                text_chunk = np.pad(text_chunk, (0, len(data)-len(text_chunk)))
                data += text_chunk[:, np.newaxis]
            else:
                self.text_audio = None
        self.PlayAudio_buffer = np.concatenate([self.PlayAudio_buffer, data])
        if len(self.PlayAudio_buffer) > 32000:
            self.PlayAudio_buffer = np.zeros([0,2], dtype='int16')
        return super().step(actions)

    def close(self):
        self.stream.close()
        self.p.terminate()
        super().close()

################################################################################


class WhisperProcess():
    def __init__(
            self,
            stderr = None,
            #stderr = subprocess.PIPE,
            ):
        logging.info('creating whisper.cpp backend process')

        # create the command
        basepath = './whisper.cpp/'
        cmd = []
        cmd.extend(['stdbuf', '-oL']) # forces unbuffered stdout
        cmd.extend([basepath + 'command'])
        cmd.extend(['-m', basepath + 'models/ggml-tiny.en.bin'])
        #cmd.extend(['-cmd', basepath + 'examples/command/commands.txt'])
        cmd.extend(['-cmd', 'whisper_commands.txt'])
        #cmd.extend(['--grammar', 'zelda.gbnf'])
        cmd.extend(['-ac', '128'])
        cmd.extend(['-t', '2'])
        cmd.extend(['-l', 'en'])
        cmd.extend(['-vth', '0.8'])
        #cmd.extend(['-c', '0'])
        logging.info(f'cmd={cmd}')

        # NOTE:
        # gymnasium changes the environment variables to use DSP instead of ALSA for audio;
        # this breaks whisper.cpp, and to we create new environment variables without this change
        my_env = os.environ.copy()
        if 'SDL_AUDIODRIVER' in my_env:
            del my_env['SDL_AUDIODRIVER']

        # create the process
        self.proc = subprocess.Popen(
                cmd,
                stderr=subprocess.STDOUT,
                stdout=subprocess.PIPE,
                text=True,
                env=my_env,
                )
       
        # make stdout a non-blocking file
        fd = self.proc.stdout.fileno()
        fl = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

        # bookkeeping
        self.rawlines = [None]
        self.commands = [None]
        self.newcommands = []

    def read_buffer(self):
        # NOTE/FIXME:
        # we put .stdout into a non-blocking mode in the constructor;
        # for some unknown reason, .read() does not throw a reasonable error when the buffer is empty,
        # but instead throws a TypeError;
        # my best guess is that this is a python bug that is specific to the local settings,
        # and that this should be fixed to something more general
        try:
            buff = self.proc.stdout.read()
            #print(f"buff={buff}")
        except TypeError:
            return

        # search through the buffer and add lines that contain commands to self.lines
        lines = buff.split('\n')
        for line in lines:
            parts = line.split(':')
            #print(f"parts={parts}")

            if len(parts) == 2 and parts[1].strip().startswith('Speech'):
                self.commands.append(None)
                self.newcommands.append(None)

            # extract commands from the --grammar flag
            if len(parts) >= 3 and parts[1].strip().startswith('DEBUG'):
                rawline = ':'.join(parts[2:])
                self.rawlines.append(rawline)
                m = re.search("'(.*)', prob = (.*)%", rawline)
                if m:
                    command = {
                        'command': m.group(1),
                        'p': m.group(2),
                        }
                    self.newcommands.append(command)
                    self.commands.append(command)

            # extract commands from the -cmd flag
            if len(parts) >= 3 and parts[1].strip().startswith('detected'):
                rawline = ':'.join(parts[2:])
                self.rawlines.append(rawline)
                m = re.search('\x1b\\[1m(.*)\x1b\\[0m \\| p = (.*) \\| t = (.*)', rawline)
                if m:
                    command = {
                        'command': m.group(1),
                        'p': m.group(2),
                        't': m.group(3),
                        }
                    self.newcommands.append(command)
                    self.commands.append(command)

    def most_recent_command(self):
        self.read_buffer()
        if len(self.newcommands) > 0:
            ret = self.newcommands[-1]
        else:
            ret = self.commands[-1]
        self.newcommands = []
        return ret


import atexit
import psutil  # Requires installing psutil library: pip install psutil
import os
import logging
import time

def kill_children():
    parent = psutil.Process(os.getpid())  # Get current process
    for child in parent.children(recursive=True):  # Get all child processes recursively
        child.terminate()
        try:
            logging.info(f'waiting for {child}')
            child.wait()
        except (psutil.TimeoutExpired, psutil.NoSuchProcess) as e:
            logging.warning(f'{e}')
atexit.register(kill_children)


class Whisper(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.whisper = WhisperProcess()
        self.command = None

    def step(self, action):
        ret = super().step(action)
        command = self.whisper.most_recent_command()
        if self.command != command and self.command is not command:
            print(f"self.command={self.command}")
        self.command = command
        return ret
