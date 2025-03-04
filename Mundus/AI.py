import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor

import groq
import gymnasium

from .Retro import RetroKB


################################################################################
# Generic LLM Code
################################################################################

client = groq.Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)


#def run_llm(system, user, model='llama-3.3-70b-versatile', seed=None):
#def run_llm(system, user, model='llama-3.1-8b-instant', seed=None):
def run_llm(system, user, model='qwen-2.5-32b', seed=None):
#def run_llm(system, user, model='llama-3.3-70B-specdec', seed=None):
    '''
    This is a helper function for all the uses of LLMs in this file.
    '''
    chat_completion = client.chat.completions.create(
        messages=[
            {
                'role': 'system',
                'content': system,
            },
            {
                "role": "user",
                "content": user,
            }
        ],
        model=model,
        seed=seed,
    )
    return chat_completion.choices[0].message.content


thread_pool = ThreadPoolExecutor(max_workers=1)
def run_llm_callback(callback_function, **kwargs):
    def thread_function():
        try:
            result = run_llm(**kwargs)
            callback_function(result)
        except Exception as e:
            logging.error(f'Error in thread: {e}')
            raise

    # only submit the callback to run if we have no other llm API call outstanding
    if thread_pool._work_queue.empty():
        future = thread_pool.submit(thread_function)


def extract_json(text):
    '''
    Extract a JSON object from text, ignoring non-JSON content.
        
    >>> extract_json('foo {"a": 1, "b": 2} bar')
    {'a': 1, 'b': 2}
    >>> extract_json('[1,2,3] extra text')
    [1, 2, 3]
    >>> extract_json('no json here')
    >>> extract_json('{"a": [1,2], "b": {"c": 3}} blah')
    {'a': [1, 2], 'b': {'c': 3}}
    >>> extract_json("""
    ... ```json
    ... {
    ...   "task": "screen_east",
    ...   "explanation_debug": "Given Link's current position at H8, the nearest point of interest is the heart piece H3 at coordinate E8, which is directly east of Link's current location. Moving east would allow Link to potentially acquire more health and increase his chances of survival. This move appears to be a strategic decision to prioritize health before exploring other areas or tackling dungeons."
    ... }
    ... ```
    ... """)
    {'task': 'screen_east', 'explanation_debug': "Given Link's current position at H8, the nearest point of interest is the heart piece H3 at coordinate E8, which is directly east of Link's current location. Moving east would allow Link to potentially acquire more health and increase his chances of survival. This move appears to be a strategic decision to prioritize health before exploring other areas or tackling dungeons."}
    '''
    json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            return None
    return None


################################################################################
# Zelda Specific code
################################################################################

zelda_map = '''
      ---------------------------------------------------------------
 A   |   |   |   |   |   |D9 |   |   |   |   |WS |D5 |   |   | L |   |
      ---------------------------------------------------------------
 B   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
      ---------------------------------------------------------------
 C   |   |MS |D6 |   |PB |   |   |   |   |   |   |   |H2 |   |   |H4 |
      ---------------------------------------------------------------
 D   |   |   |   |   |S5 |   |   |D1 |   |F1 |   |   |D2 |   |   |   |
      ---------------------------------------------------------------
 E   |   |   |D7 |F2 |S4 |D4 |S3 |H3 |   |   |   |   |   |   |   |   |
      ---------------------------------------------------------------
 F   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |H5 |
      ---------------------------------------------------------------
 G   |   |   |   |   |MS |   |S1 |   |   |   |   |   |D8 |   |   |S2 |
      ---------------------------------------------------------------
 H   |   |   |   |   |D3 |   |   | X |   |   |   |H1 |   |   |   |   |
      ---------------------------------------------------------------
       1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16

KEY/LEGEND:

Map coordinates are in the form A1, where the letter indicates the Y-axis and the number the X-axis
screen_north changes the letter to the previous letter of the alphabet (e.g. B3 -> A3)
screen_south changes the letter to the next letter of the alphabet (e.g. B3 -> C3)
screen_east changes the number to the next number (e.g. B3 -> B4)
screen_west changes the number to the previous number (e.g. B3 -> B2)

X = The game starts here at the beginning, and whenever you die or restart a 
game, unless you're in a dungeon/labyrinth.  The wood sword is located in the cave in this location.

D1 = Dungeon One ("Eagle") [Coordinate D8]
D2 = Dungeon Two ("Moon") [Coordinate D13]
D3 = Dungeon Three ("Manji") [Coordinate H5]
D4 = Dungeon Four ("Snake") [Coordinate E6]
D5 = Dungeon Five ("Lizard") [Coordinate A12]
D6 = Dungeon Six ("Dragon") [Coordinate C3]
D7 = Dungeon Seven ("Demon") [Coordinate E3]
D8 = Dungeon Eight ("Lion") [Coordinate G14]
D9 = Dungeon Nine ("Death Mountain") [Coordinate A6]

H1 = Heart Piece One [Coordinate H12]
H2 = Heart Piece Two [Coordinate C13]
H3 = Heart Piece Three [Coordinate E8]
H4 = Heart Piece Four [Coordinate C16]
H5 = Heart Piece Five [Coordinate F16]

F1 = Fairy One (will give link more hearts) [Coordinate D10]
F2 = Fairy Two (will give link more hearts) [Coordinate E4]

S1 = Shop One (Magic Shield [160], Key [100], Blue Candle [60]) [Coordinate G7]
S2 = Store Two (Magic Shield [130], Bombs [20], Arrows [80]) [Coordinate G16]
S3 = Store Three (Magic Shield [90], Monster Bait [100], Heart [10]) [Coordinate E7]
S4 = Store Four (Magic Shield [130], Bombs [20], Arrows [80]) [Coordinate E5]
S5 = Store Five (Key [80], Blue Ring [250], Monster Bait [60]) [Coordinate D5]

WS = White Sword [Coordinate A11]
L = Letter [Coordinate A15]
MS = Medicine Shop [Coordinate G5]
PB = Power Bracelet [Coordinate C5]
MS = Magical Sword [Coordinate C2]
'''


class ZeldaAI(gymnasium.Wrapper):
    '''
    FIXME:
    Make generic to the RetroKB class.
    '''
    def __init__(self, env):
        super().__init__(env)
        zelda_env = self.env
        while not isinstance(zelda_env, RetroKB):
            assert zelda_env.env is not None
            zelda_env = zelda_env.env
        self.ZeldaWrapper = zelda_env

        self.objective = {
                "objective": "Obtain the white sword at A11",
                "what_zelda_says": "Link, go north and get the powerful white sword to defeat stronger monsters.",
                "explanation_debug": "The white sword is a significant upgrade from the wood sword and will make combat much easier. It is located nearby at A11, so obtaining it should be Link's top priority before attempting to make progress in the dungeons. The player doesn't have enough rupees to buy helpful items from the shops yet.",
                "sign_of_failure": "If Link moves too far away from A11 or dies before obtaining the white sword, he has failed the objective.",
                "sign_of_success": "If Link reaches map location A11 and obtains the white sword, he has succeeded in completing the objective."
            }
        self.objective = {
                "objective": "Find Dungeon One",
                "what_zelda_says": "Go find the Eagle dungeon to get more power and help me!",
                "explanation_debug": "The closest dungeon is D1, also known as the Eagle dungeon, which is located at coordinate D8. Since Link has not completed any dungeons, it makes sense for him to start with the nearest one. This will allow him to gain more power and items to aid in his quest.",
                "sign_of_failure": "Link has failed if he dies or runs out of health before reaching the dungeon.",
                "sign_of_success": "Link has succeeded if he enters the Eagle dungeon at coordinate D8."
            }

        self.task_history = []

    def _get_map_coordinates(self):
        eb = self.ZeldaWrapper.ram[0xEB]
        x = eb % 0x10 + 1
        y = eb // 0x10
        X = str(x)
        Y = chr(ord('A') + y)
        coord = Y + X
        logging.debug(f'_get_map_coordinates() = {coord}')
        return coord

    def _generate_newtask(self):
        valid_tasks = self.ZeldaWrapper._get_valid_tasks()
        valid_tasks = [task for task in valid_tasks if 'screen' in task or task == 'attack']
        prompt = f'''
You are the spirit of Princess Zelda and your job is to help Link defeat Ganon.

<map>
{zelda_map}
</map>

<status>
Link has the following items:
- wood sword
- 4 bombs
- 23 rupees
- blue candle

Link's health is:
- 3 hearts (this is Link's current health level; fairies and heart pieces increase this number, but it cannot go higher than the number of heart containers)
- 3 heart containers (this is Link's maximum health level; only heart containers increase this number)

Link has not completed any dungeons.

Link's current position is {self._get_map_coordinates()}.
</status>

Link's current objective is {self.objective}

The most recent tasks towards completing this objective are:
<task_history>
{self.task_history[-10:]}
</task_history>

What should Link's task be for the next 10-30 seconds?
Possible values are {valid_tasks}.
Note that the map is a maze, and sometimes Link will have to move in a non-obvious direction in order to make progress towards his task.
Link should never repeat a position listed in task_history above unless it is the only legal move.

Format your answer as a JSON object with the following keys:
1. "explanation_debug" should contain a detailed explanation about your reasoning process (it should be 1-4 sentences, state the current map position and expected new map position, use map coordinates to explain the expected path, explain how the path will avoid previous map positions listed in task_history, explain why the choice is contained in the possible values, and be in the voice of a neutral observer [and not princess Zelda])
2. "task" should exactly match one of the above tasks (link should not revisit a position from his most resent tasks and the task must be in the possible values list above)
3. "link_say" should be a very short phrase describing what Link will do (is should be in the 1st person, use simple language understandable by a 6 year old, and be <10 words long; it may or may not reference the larger objective)

The response should contain only a single JSON object and no explanatory text or markdown.
'''
        logging.info(f"prompt={prompt}")
        def callback(text_response):
            json_response = extract_json(text_response)
            if json_response is None:
                logging.warning(f'text_response={text_response}')
            logging.debug(f"json_response={json_response}")
            self.ZeldaWrapper.register_text(json_response['explanation_debug'], speaker='llm')
            newtask = json_response['task']
            logging.debug(f'setting newtask={newtask}, selected from valid_tasks={valid_tasks}')
            self.ZeldaWrapper._set_episode_task(newtask)
            self.task_history.append(json_response)
        run_llm_callback(callback, system='', user=prompt)

    '''
    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)

        #logging.info(f"self.ZeldaWrapper.frames_without_attack={self.ZeldaWrapper.frames_without_attack}; valid_tasks={valid_tasks}")

        # FIXME:
        # this is a hack to "reset" the state
        if self.ZeldaWrapper.frames_without_attack > 10*60:
            self.ZeldaWrapper._set_episode_task('interactive_onmouse')
            x, y = self.ZeldaWrapper._get_random_link_position()
            self.ZeldaWrapper.mouse = {'x': x, 'y': y}
            self.ZeldaWrapper.frames_without_attack = 0

        # we should only ever consider setting the task by AI
        # if the task is not a manually generated task
        if 'interactive' not in self.ZeldaWrapper.episode_task:

            # if we have completed the previous task,
            # select a new one
            if info['is_success']: # or terminated or truncated:
                if self.ZeldaWrapper.episode_task != 'attack':
                    logging.debug(f'previous task={self.ZeldaWrapper.episode_task} succeeded; new task=attack')
                    self.ZeldaWrapper._set_episode_task('attack')
                else:
                    self._generate_newtask()
        return observation, reward, terminated, truncated, info

    '''
