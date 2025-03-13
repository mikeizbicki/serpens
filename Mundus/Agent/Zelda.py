import json
import os
import random
import re
import time
import traceback

import gymnasium

from Mundus.Retro import RetroKB
from Mundus.Agent.LLM import LLM, extract_json

# setup logging
import logging
logger = logging.getLogger(__name__)


################################################################################
# Zelda-specific hacks
################################################################################

class UnstickLink(gymnasium.Wrapper):
    '''
    Link sometimes gets stuck on an individual screen.
    This wrapper tries to detect when this is happening,
    and then moves link to a random location.
    '''
    def __init__(self, env):
        super().__init__(env)

        # find the ZeldaWrapper environment
        env = self.env
        while not isinstance(env, RetroKB):
            assert env.env is not None
            env = env.env
        self.ZeldaWrapper = env

    def reset(self, **kwargs):
        ret = super().reset(**kwargs)
        self.link_xy = {}
        self.link_xy_nochange_count = 0
        self.same_screen_count = 0
        self.ramEB_history = [(0, self.ZeldaWrapper.ram[0xEB])]
        return ret

    def step(self, action):
        result = super().step(action)

        # count how long link has been in the same screen
        ramEB = self.ZeldaWrapper.ram[0xEB]
        if self.ramEB_history[-1][1] == ramEB:
            self.same_screen_count += 1
        else:
            self.same_screen_count = 0
            self.ramEB_history.append((self.ZeldaWrapper.stepcount, ramEB))

        # count how long link has been stuck at the same pixels
        new_link_xy = {
            'x': self.ZeldaWrapper.ram[112],
            'y': self.ZeldaWrapper.ram[132],
            }
        if new_link_xy == self.link_xy:
            self.link_xy_nochange_count += 1
        else:
            self.link_xy_nochange_count = 0
        self.link_xy = new_link_xy

        # change task if link is stuck
        change_task = False

        if self.ZeldaWrapper.frames_without_attack > 10*60:
            logger.info(f'UnstickLink: frames_without_attack={self.ZeldaWrapper.frames_without_attack}')
            change_task = True
            self.ZeldaWrapper.register_text("Estoy confundido")

        elif self.link_xy_nochange_count >= 60 and self.link_xy_nochange_count % 20 == 0:
            logger.info(f'UnstickLink: link_xy_nochange_count={self.link_xy_nochange_count}')
            change_task = True
            self.ZeldaWrapper.register_text("Estoy atascado")

        elif len(self.ramEB_history) > 5 and self.ramEB_history[-5][0] > self.ZeldaWrapper.stepcount - 60:
            logger.info(f'UnstickLink: self.ramEB_history[-5:]={self.ramEB_history[-5:]}')
            change_task = True
            self.ZeldaWrapper.register_text("Estoy mareado")

        if change_task:
            self.ZeldaWrapper._set_episode_task('interactive_onmouse')
            x, y = self.ZeldaWrapper._get_random_link_position()
            self.ZeldaWrapper.mouse = {'x': x, 'y': y}
            self.ZeldaWrapper.frames_without_attack = 0
            self.ZeldaWrapper.link_xy_nochange_count = 0

        # if we've been on the same screen too long, change screens
        if self.same_screen_count >= 30*60 and self.same_screen_count % (10*60) == 0:
            logger.info(f'UnstickLink: same_screen_count={self.same_screen_count}')
            self.env.generate_newtask()
            self.ZeldaWrapper.register_text("Estaba aquí demasiado tiempo")

        return result


################################################################################
# Coordinate Systems
################################################################################

class BattleshipCoords():
    @staticmethod
    def move(coord, direction):
        '''
        >>> BattleshipCoords.move('B3', 'north')
        'A3'
        >>> BattleshipCoords.move('B3', 'south')
        'C3'
        >>> BattleshipCoords.move('B3', 'east')
        'B4'
        >>> BattleshipCoords.move('B3', 'west')
        'B2'
        >>> BattleshipCoords.move('E10', 'north')
        'D10'
        >>> BattleshipCoords.move('E10', 'south')
        'F10'
        >>> BattleshipCoords.move('E10', 'west')
        'E9'
        >>> BattleshipCoords.move('E10', 'east')
        'E11'
        >>> BattleshipCoords.move('E10', 'screen_east')
        'E11'
        '''
        if direction.startswith('screen_'):
            direction = direction[7:]
        if direction == 'north':
            return chr(ord(coord[0]) - 1) + coord[1:]
        if direction == 'south':
            return chr(ord(coord[0]) + 1) + coord[1:]
        if direction == 'east':
            return coord[0] + str(int(coord[1:]) + 1)
        if direction == 'west':
            return coord[0] + str(int(coord[1:]) - 1)

    def dist(coord1, coord2):
        '''
        >>> BattleshipCoords.dist('A1', 'C2')
        3
        >>> BattleshipCoords.dist('C2', 'A1')
        3
        >>> BattleshipCoords.dist('C20', 'C0')
        20
        >>> BattleshipCoords.dist('A1', 'B1')
        1
        '''
        x1 = ord(coord1[0]) - ord('A')
        y1 = int(coord1[1:])
        x2 = ord(coord2[0]) - ord('A')
        y2 = int(coord2[1:])
        return abs(x1 - x2) + abs(y1 - y2)


class ChessCoords():
    @staticmethod
    def move(coord, direction):
        '''
        >>> ChessCoords.move('B3', 'north')
        'B4'
        >>> ChessCoords.move('B3', 'south')
        'B2'
        >>> ChessCoords.move('B3', 'east')
        'C3'
        >>> ChessCoords.move('B3', 'west')
        'A3'
        >>> ChessCoords.move('E10', 'north')
        'E11'
        >>> ChessCoords.move('E10', 'south')
        'E9'
        >>> ChessCoords.move('E10', 'west')
        'D10'
        >>> ChessCoords.move('E10', 'east')
        'F10'
        '''
        if direction == 'west':
            return chr(ord(coord[0]) - 1) + coord[1:]
        if direction == 'east':
            return chr(ord(coord[0]) + 1) + coord[1:]
        if direction == 'north':
            return coord[0] + str(int(coord[1:]) + 1)
        if direction == 'south':
            return coord[0] + str(int(coord[1:]) - 1)


################################################################################
# Agent
################################################################################

class Agent(gymnasium.Wrapper):

    def __init__(self, env, task_selector, objective_selector):
        super().__init__(env)

        # track agent state/history
        self.coords_history = []
        self.task_selector = task_selector(self)
        self.objective_selector = objective_selector(self)

        # find the ZeldaWrapper environment
        env = self.env
        while not isinstance(env, RetroKB):
            assert env.env is not None
            env = env.env
        self.ZeldaWrapper = env

    def get_current_coords(self):
        eb = self.ZeldaWrapper.ram[0xEB]
        x = eb % 0x10 + 1
        y = eb // 0x10
        X = str(x)
        Y = chr(ord('A') + y)
        coord = Y + X
        #logger.debug(f'_get_map_coordinates() = {coord}')
        return coord

    def generate_newtask(self):
        valid_tasks = self.ZeldaWrapper._get_valid_tasks()
        valid_tasks = [task for task in valid_tasks if 'screen' in task or task == 'attack']
        newtask = self.task_selector.select_task(valid_tasks)
        if newtask is not None:
            logger.debug(f'setting newtask={newtask}, selected from valid_tasks={valid_tasks}')
            self.ZeldaWrapper._set_episode_task(newtask)

    def generate_objective(self):
        self.objective = self.objective_selector.select_objective()
        #self.ZeldaWrapper.register_text(f"Mi objetivo nuevo es ir a {self.objective}.")
        self.ZeldaWrapper.register_text(f"Link, debes ir a {self.objective}.", speaker='Navi')

    def reset(self, **kwargs):
        ret = super().reset(**kwargs)
        self.generate_objective()
        coords = self.get_current_coords()
        self.coords_history.append(coords)
        self.task = self.generate_newtask()
        self.is_success_count = 0
        return ret

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)

        if self.objective_selector.is_objective_complete():
            self.objective = self.objective_selector.select_objective()

        # FIXME:
        # sometimes is_success can "incorrectly" be set on a single frame
        # due to changing the task;
        # we track how long it has been set for to create a more stable success signal
        if info['is_success']:
            self.is_success_count += 1
        else:
            self.is_success_count = 0

        # update history of map coordinates
        coords = self.get_current_coords()
        if self.coords_history[-1] != coords:
            self.coords_history.append(coords) 
            logging.debug(f'objetive={self.objective}, coords={coords}, ram[0xEB]={hex(self.ZeldaWrapper.ram[0xEB])}')

        # we should only ever consider setting the task by AI
        # if the task is not a manually generated task
        if 'interactive' not in self.ZeldaWrapper.episode_task:

            # if we have completed the previous task,
            # select a new one
            if self.is_success_count >= 10:
                if self.ZeldaWrapper.episode_task != 'attack':
                    logger.info(f'previous task={self.ZeldaWrapper.episode_task} succeeded; new task=attack')
                    self.ZeldaWrapper._set_episode_task('attack')
                else:
                    self.generate_newtask()

        return observation, reward, terminated, truncated, info


################################################################################
# ObjecticeSelectors
################################################################################


class RandomObjective():
    def __init__(self, agent, seed=None):
        self.agent = agent
        if seed is None:
            seed = time.time()
        self.random = random.Random(seed)

    def select_objective(self):
        y = self.random.choice('ABCDEFGH')
        x = str(self.random.randint(1, 16))
        coords = y + x
        logging.info(f'select_objective: {coords}')
        return coords

    def is_objective_complete(self):
        # FIXME:
        # we should change this interface to be more functional
        return self.agent.objective == self.agent.coords_history[-1]


################################################################################
# TaskSelectors
################################################################################


class RandomTask():
    def __init__(self, agent):
        self.agent = agent
        self.random = random.Random(0)

    def select_task(self, valid_tasks):
        return self.random.choice(valid_tasks)


class RandomExplorer():
    def __init__(self, agent):
        self.agent = agent
        self.random = random.Random(0)

    def select_task(self, valid_tasks):
        coords = self.agent.coords_history[-1]
        horizon = self.agent.coords_history[-3:]
        moves = [task for task in valid_tasks if 'screen' in task]
        valid_moves = [move for move in moves if BattleshipCoords.move(coords, move) not in horizon]
        if len(valid_moves) == 0:
            valid_moves = moves
        return self.random.choice(valid_moves)


class SimpleNavigator():
    def __init__(self, agent):
        self.agent = agent
        self.random = random.Random(0)

    def select_task(self, valid_tasks):
        coords = self.agent.coords_history[-1]
        horizon = self.agent.coords_history[-3:]
        moves = []
        movement_tasks = ['screen_north', 'screen_south', 'screen_east', 'screen_west']
        for task in valid_tasks:
            if task in movement_tasks:
                newcoords = BattleshipCoords.move(coords, task)
                dist = BattleshipCoords.dist(self.agent.objective, newcoords)
                moves.append((dist, task, newcoords))
        moves.sort()
        valid_moves = [move for move in moves if move[2] not in horizon]
        if len(valid_moves) == 0:
            valid_moves = moves
        closest_moves = [move for move in valid_moves if move[0] == valid_moves[0][0]]
        if len(closest_moves) > 0:
            return self.random.choice(closest_moves)[1]
        else:
            logging.error(f'SimpleNavigator: len(closest_moves) = 0; valid_tasks={valid_tasks}, valid_moves={valid_moves}, closest_moves={closest_moves}')
            return 'attack'


class LLM_Navigator():
    def __init__(self, agent, threaded=True):
        self.agent = agent
        self.llm = LLM()
        self.threaded = threaded

    def select_task(self, valid_tasks):
        current_position = self.agent.coords_history[-1]
        valid_tasks = [task for task in valid_tasks if 'screen' in task or task == 'attack']
        invalid_directions = []
        if 'screen_north' not in valid_tasks: invalid_directions.append('north')
        if 'screen_south' not in valid_tasks: invalid_directions.append('south')
        if 'screen_east' not in valid_tasks: invalid_directions.append('east')
        if 'screen_west' not in valid_tasks: invalid_directions.append('west')
        invalid_directions_str = '\n'.join([f' - {dir} (go to {BattleshipCoords.move(current_position, dir)})' for dir in invalid_directions])
        valid_directions = []
        if 'screen_north' in valid_tasks: valid_directions.append('north')
        if 'screen_south' in valid_tasks: valid_directions.append('south')
        if 'screen_east' in valid_tasks: valid_directions.append('east')
        if 'screen_west' in valid_tasks: valid_directions.append('west')
        valid_directions_str = '\n'.join([f' - {dir} (go to {BattleshipCoords.move(current_position, dir)})' for dir in valid_directions])
        prompt = f'''
You are navigating a 2D grid using algebraic notation.
Your current position is {current_position} and target position is H5.

You can move in the following directions:
{valid_directions_str}

You cannot move
{invalid_directions_str}

You have previously visited positions {str(self.agent.coords_history[-10:-1]).replace('\'', '')}.
DO NOT REVISIT THESE POSITIONS.

Output a JSON dictionary with the following keys:
0. "debug_explanation": a <4 sentence explanation of your reasoning process (you should explicitly mention the direction you will move, the position you will be in, your current distance to target, and the distance to target after your move)
1. "direction": which direction you will move (it should exactly match one of the directions above)
2. "next_position": the coordinates you will be at after the move (format ass a string)
3. "distance": estimate the number of tiles (in L1 distance) from next_position to the target position

Only output the JSON.
Do not provide any explanations.
'''

        def callback_debug():
            self.agent.ZeldaWrapper.register_text("consulting llm", speaker='llm_debug')
        def callback(text_response):
            logger.info(f"prompt={prompt}")
            json_response = extract_json(text_response)
            if json_response is None:
                logger.warning(f'text_response={text_response}')
            logger.debug(f"json_response={json_response}")
            self.agent.ZeldaWrapper.register_text(text_response, speaker='llm')
            json_response['task'] = 'screen_' + json_response['direction']
            newtask = json_response['task']
            logger.debug(f'setting newtask={newtask}, selected from valid_tasks={valid_tasks}')
            self.agent.ZeldaWrapper._set_episode_task(newtask)
        self.llm.run_llm_callback(callback, callback_debug=callback_debug, threaded=self.threaded, prompt=prompt)











################################################################################
# old
################################################################################

# FIXME:
# for now, hardcode objectives that we're trying to reach
objective = {
        "objective": "Obtain the white sword at A11",
        "what_zelda_says": "Link, go north and get the powerful white sword to defeat stronger monsters.",
        "explanation_debug": "The white sword is a significant upgrade from the wood sword and will make combat much easier. It is located nearby at A11, so obtaining it should be Link's top priority before attempting to make progress in the dungeons. The player doesn't have enough rupees to buy helpful items from the shops yet.",
        "sign_of_failure": "If Link moves too far away from A11 or dies before obtaining the white sword, he has failed the objective.",
        "sign_of_success": "If Link reaches map location A11 and obtains the white sword, he has succeeded in completing the objective."
    }
objective = {
        "objective": "Find Dungeon One",
        "what_zelda_says": "Go find the Eagle dungeon to get more power and help me!",
        "explanation_debug": "The closest dungeon is D1, also known as the Eagle dungeon, which is located at coordinate D8. Since Link has not completed any dungeons, it makes sense for him to start with the nearest one. This will allow him to gain more power and items to aid in his quest.",
        "sign_of_failure": "Link has failed if he dies or runs out of health before reaching the dungeon.",
        "sign_of_success": "Link has succeeded if he enters the Eagle dungeon at coordinate D8."
    }

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

