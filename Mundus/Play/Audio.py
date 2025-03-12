import fcntl
import io
import logging
import os
import re
import subprocess
import sys
import threading
import time

import numpy as np
import gymnasium

# before loading pyaudio, we set an error handler;
# this error handler will cause buffer underrun warnings to not get displayed to stderr;
# a buffer underrun happens when the fps drops too low,
# and we don't want these messages cluttering other debug information
from ctypes import *
ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
def py_error_handler(filename, line, function, err, fmt):
    pass
c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
asound = cdll.LoadLibrary('libasound.so.2')
asound.snd_lib_error_set_handler(c_error_handler)
import pyaudio

# the following context manager can be used to prevent messages to stderr;
# pyaudio automatically outputs to stderr,
# and so this context manager is used to make the pyaudio code silent
import contextlib
devnull = os.open(os.devnull, os.O_WRONLY)
old_stderr = os.dup(2)
@contextlib.contextmanager
def ignore_stderr():
    sys.stderr.flush()
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)

from elevenlabs.client import ElevenLabs
from elevenlabs import play
import elevenlabs


'''
speaker, lang, voice_id
link, en, JBFqnCBsd6RMkjVDRZzb
link, es, 7imL5gCghUGRIJcFUust
link, el, KDImLuG6RkuyuX5httC7
link, ko, TRO4gatqxbbwLXHLDLSk
'''

def text_info_to_voice_id(text_info):
    if text_info['lang'] == 'en':
        return 'JBFqnCBsd6RMkjVDRZzb'
    if text_info['lang'] == 'es' and text_info['speaker'] == 'Link':
        return 'z3kTTwYbQrmL7ckdGcJi'
        #return 'JBFqnCBsd6RMkjVDRZzb'
        #return 'Nh2zY9kknu6z4pZy6FhD'
    if text_info['lang'] == 'es' and text_info['speaker'] == 'Navi':
        return '9oPKasc15pfAbMr7N6Gs'
    return 'JBFqnCBsd6RMkjVDRZzb'


from scipy import signal
def make_fairy(text_frame_audio):
    # Pitch shift (resample at higher rate for higher pitch)
    #pitch_factor = 1.1  # Higher = more fairy-like
    #text_frame_audio = signal.resample(text_frame_audio, int(len(text_frame_audio)/pitch_factor))
  # 
    ## Add shimmer/sparkle (add high frequency oscillation)
    #t = np.arange(len(text_frame_audio))
    #shimmer = (np.sin(t * 0.3) * 0.15 * text_frame_audio).astype(text_frame_audio.dtype)
    #text_frame_audio = text_frame_audio + shimmer

    # Add echo/reverb (simple implementation)
    echo_delay = 2000  # samples
    echo_volume = 0.3
    echo = np.pad(text_frame_audio, (0, echo_delay))[-len(text_frame_audio):]
    text_frame_audio = text_frame_audio + (echo * echo_volume).astype(text_frame_audio.dtype)

    return text_frame_audio.astype('int16')


def make_monster(text_frame_audio):
    # Pitch shift (resample at higher rate for higher pitch)
    pitch_factor = 0.8  # Higher = more fairy-like
    audio_len = len(text_frame_audio)
    #text_frame_audio = text_frame_audio.astype(np.float32)
    text_frame_audio = signal.resample(text_frame_audio, int(audio_len/pitch_factor))
    #text_frame_audio = signal.resample(text_frame_audio, audio_len)
    #text_frame_audio = text_frame_audio.astype(np.int16)

    # Add distortion/growl effect
    distortion_amount = 0.2
    text_frame_audio = np.clip(text_frame_audio * (1 + distortion_amount), -32768, 32767).astype(text_frame_audio.dtype)

    # Add rumble (low frequency modulation)
    t = np.arange(len(text_frame_audio))
    rumble = (np.sin(t * 0.01) * 0.2 * text_frame_audio).astype(text_frame_audio.dtype)
    text_frame_audio = text_frame_audio + rumble

    # Add reverb for menacing echo
    echo_delay = 3000  # samples
    echo_volume = 0.4
    echo = np.pad(text_frame_audio, (0, echo_delay))[-len(text_frame_audio):]
    text_frame_audio = text_frame_audio + (echo * echo_volume).astype(text_frame_audio.dtype)

   
    return text_frame_audio.astype('int16')


class PlayAudio_ElevenLabs(gymnasium.Wrapper):
    '''
    ElevenLabs provides an API for voice generation.
    This sends the generated audio to pyaudio.
    '''
    def __init__(self, env, ignore_ALSA_warnings=True):
        super().__init__(env)
        self.audio_env = self.env
        self.elevenlabs = ElevenLabs()
        self.cache_dir = '.audio_cache'
        os.makedirs(self.cache_dir, exist_ok=True)

        env.unwrapped.RetroKB.add_text_callback(lambda text_info: self.register_text(text_info))

    def register_text(self, text_info):
        audio_transform = lambda x: x
        if text_info['speaker'] == 'Navi':
            audio_transform = make_fairy
        if text_info['speaker'] == 'Monster':
            audio_transform = make_monster

        time_start = time.time()
        try:
            voice_id = text_info_to_voice_id(text_info)
            dirname = f'{self.cache_dir}/{text_info["lang"]}_elevenlabs_{voice_id}'
            os.makedirs(dirname, exist_ok=True)
            filepath = f'{dirname}/{text_info["text"]}'
            with open(filepath, 'rb') as fin:
                self.audio_env.push_audio(text_info['speaker'],audio_transform(np.repeat(np.frombuffer(fin.read(), dtype=np.int16), 2)))
        except FileNotFoundError:
            def thread_function():
                response = self.elevenlabs.text_to_speech.convert(
                    text=text_info['text'],
                    voice_id=voice_id,
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
                logging.info(f"ElevenLabs API initial response; lag={lag}")
                audio_stream = io.BytesIO()
                for i, chunk in enumerate(response):
                    lag = time.time() - time_start
                    if chunk:
                        audio_stream.write(chunk)
                audio_stream.seek(0)
                self.audio_env.push_audio(text_info['speaker'],audio_transform(np.repeat(np.frombuffer(audio_stream.read(), dtype=np.int16), 2)))
                lag = time.time() - time_start
                logging.info(f"saving sound file {filepath}; lag={lag}")
                with open(filepath, 'wb') as fout:
                    audio_stream.seek(0)
                    fout.write(audio_stream.read())
            thread = threading.Thread(target=thread_function)
            thread.start()


class PlayAudio_pyaudio(gymnasium.Wrapper):
    '''
    This wrapper uses pyaudio to play the sound of the NES stable-retro environment.
    '''
    def __init__(self, env, ignore_ALSA_warnings=True):
        super().__init__(env)
        self.PlayAudio_buffer = np.zeros([0,2], dtype='int16')
        self.buffers = {}
        self.buffers_index = {}
    
        # create the audio stream
        def playing_callback(in_data, frame_count, time_info, status):
            if self.PlayAudio_buffer.shape[0] >= frame_count:
                data = self.PlayAudio_buffer[:frame_count, :]
                self.PlayAudio_buffer = self.PlayAudio_buffer[frame_count:, :]
                data = data.flatten()
                #logging.debug(f"self.PlayAudio_buffer.shape={self.PlayAudio_buffer.shape}")
                return (data.tobytes(), pyaudio.paContinue)
            else:
                #logging.warning('PlayAudio: buffer underrun')
                return (b'\x00' * frame_count * 4, pyaudio.paContinue)  # 4 bytes per frame (2 channels * 2 bytes per sample)

        with ignore_stderr():
            logging.info(f"env.unwrapped.em.get_audio_rate()={env.unwrapped.em.get_audio_rate()}")
            self.p = pyaudio.PyAudio()
            self.stream = self.p.open(
                    format=pyaudio.paInt16,
                    channels=2,
                    rate=int(env.unwrapped.em.get_audio_rate()),
                    output=True,
                    stream_callback=playing_callback,
                    frames_per_buffer=128,
                    start=False,
                    )
            self.stream.start_stream()

    def push_audio(self, name, value):
        if name not in self.buffers:
            self.buffers[name] = value
            self.buffers_index[name] = 0

    def step(self, actions):
        game_frame_audio = self.env.unwrapped.em.get_audio().astype('int16')
        game_frame_audio //= 3 # make text audio to be louder than game audio
        keys_to_delete = []
        for k, buff in self.buffers.items():
            index = self.buffers_index[k]
            buff_frame = buff[index:index + len(game_frame_audio)]
            if len(buff_frame) > 0:
                self.buffers_index[k] += len(game_frame_audio)
                buff_frame = np.pad(buff_frame, (0, len(game_frame_audio) - len(buff_frame)))
                game_frame_audio += buff_frame[:, np.newaxis]
            else:
                keys_to_delete.append(k)
        for k in keys_to_delete:
            del self.buffers[k]
        self.PlayAudio_buffer = np.concatenate([self.PlayAudio_buffer, game_frame_audio])
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
