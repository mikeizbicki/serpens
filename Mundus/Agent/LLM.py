from concurrent.futures import ThreadPoolExecutor
import json
import re
import time
import traceback
import groq

# setup logging
import logging
logger = logging.getLogger(__name__)


class LLM():
    def __init__(self):
        # variables for storing hyperparameters
        self.minimum_time_between_llm_calls = 1
        self.seed = None
        self.model = 'llama-3.1-8b-instant'

        # variables for doing work
        self.thread_pool = ThreadPoolExecutor(max_workers=1)
        self.client = client = groq.Groq()

        # variables for logging purposes
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.llm_calls = []
        self._current_future = None

    def run_llm(self, prompt, callback_debug=None, system='', max_tokens=None):
        # only run the llm if we have waited sufficiently long between API calls;
        # this check ensures that we don't spam the LLM and waste tokens,
        # and can help us find bugs in the code that is calling the LLM
        # (and potentially discarding the results of a previous call)
        last_llm_call = 0
        if len(self.llm_calls) > 0:
            last_llm_call = self.llm_calls[-1].get('stop_time')
            if last_llm_call is None:
                logging.warning('rum_llm: not calling API; last_llm_call = None')
                return
        time_since_llm_call = time.time() - last_llm_call
        if time_since_llm_call < self.minimum_time_between_llm_calls:
            logging.warning(f'rum_llm: not calling API; time_since_llm_call < self.minimum_time_between_llm_calls = {time_since_llm_call} < {self.minimum_time_between_llm_calls}')
            return

        # connect to the API
        if callback_debug is not None:
            callback_debug()
        runid = len(self.llm_calls)
        self.llm_calls.append({
            'start_time': time.time(),
            })
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    'role': 'system',
                    'content': system,
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.model,
            seed=self.seed,
            max_tokens=max_tokens,
        )

        # bookkeeping
        self.prompt_tokens += chat_completion.usage.prompt_tokens
        self.completion_tokens += chat_completion.usage.completion_tokens
        logger.info(f"self.prompt_tokens, self.completion_tokens={self.prompt_tokens, self.completion_tokens}")
        self.llm_calls[runid]['stop_time'] = time.time()
        self.llm_calls[runid]['wait_time'] = self.llm_calls[runid]['stop_time'] - self.llm_calls[runid]['start_time']
        logging.debug(f"self.llm_calls[runid]['wait_time'] ={self.llm_calls[runid]['wait_time'] }")
        return chat_completion.choices[0].message.content

    def run_llm_callback(self, callback_function, threaded=True, **kwargs):
        def thread_function():
            try:
                result = self.run_llm(**kwargs)
                if result is not None:
                    callback_function(result)
            except Exception as e:
                logger.error(f'Error in thread: {e}')
                logger.error(traceback.format_exc())
                raise

        # only submit the callback to run if we have no other llm API call outstanding
        if threaded:
            if not self._current_future or self._current_future.done(): #self.thread_pool._work_queue.empty():
                self._current_future = self.thread_pool.submit(thread_function)
            else:
                pass
                #logger.debug('run_llm_callback: callback_function not submited')
        else:
            thread_function()


def extract_json(text):
    '''
    Extract a JSON object from text, ignoring non-JSON content.
        
    >>> extract_json(None)
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
    >>> extract_json("""
    ... <think>
    ... Okay, so I'm trying to figure out which direction to move next in this maze. Let me start by understanding the problem.
    ... 
    ... We're currently at position F8, and the goal is to get to D8. The previous positions we've been to are [E11, E10, D10, E10, F10, F9]. So, I need to make sure I don't go back to any of these unless I have no other choice.
    ... 
    ... The valid directions I can move are "east" and "west". Moving east would take me to F9, and moving west would take me to F7. I can't move north or south, so those are off the table.
    ... 
    ... Looking at the previous positions, I see that F9 is already in the list. So, moving east to F9 would take me back to a position I've already been to, which I should avoid if possible. That leaves moving west to F7 as the better option because it's a new position and gets me closer to D8.
    ... 
    ... Wait, but let me make sure. If I move west to F7, is that adjacent? Yes, because moving west from F8 would be F7. Also, F7 hasn't been visited before, so that's good. Moving east to F9 would be revisiting a previous position, which isn't ideal unless necessary.
    ... 
    ... So, the best move here is to go west to F7. That way, I'm progressing towards D8 without backtracking. Plus, F7 is adjacent and hasn't been visited yet, so it meets all the criteria.
    ... </think>
    ... 
    ... ```json
    ... {
    ... "next_position": "F7",
    ... "direction": "west"
    ... }
    ... ```
    ... """)
    {'next_position': 'F7', 'direction': 'west'}
    '''
    if text is None:
        return

    # remove think tags
    pattern = r'<think>.*?</think>'
    text = re.sub(pattern, '', text, flags=re.DOTALL)

    # extract resulting json
    json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            return None
    return None


