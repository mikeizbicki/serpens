import threading
class KeyboardThread(threading.Thread):
    '''
    Taken from <https://stackoverflow.com/a/57387909>.
    '''

    def __init__(self, prompt=None, input_cbk=None, name='keyboard-input-thread'):
        self.input_cbk = input_cbk
        self.prompt = prompt
        super(KeyboardThread, self).__init__(name=name, daemon=True)
        self.start()

    def run(self):
        while True:
            self.input_cbk(input(self.prompt)) #waits to get input + Return


class ConsoleWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        # create multithreaded keyboard input
        self.current_command = None
        def set_current_command(text):
            self.current_command = text
        self.input_thread = KeyboardThread('input: ', set_current_command)



