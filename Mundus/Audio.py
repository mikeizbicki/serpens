import subprocess
import logging
import os
import time
import fcntl
import re
import time

import gymnasium


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
