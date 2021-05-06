import gym
from gym import spaces
import numpy as np
from collections import deque
import cv2
from abc import abstractmethod

class AtariWrapper(gym.Env):
    """
    Description: 
        Wrapper for the Atari 2600 environments from OpenAI Gym.
        
    Source: 
        Inspired from the OpenAI baseline wrappers at https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
    """
    
    def __init__(self, env_config):
        super().__init__()
        self.env = gym.make(env_config['env_name'])
        
        self.action_space = self.env.action_space
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata
        
        self.image_size = env_config['image_size']
        self.stack_size = env_config['stack_size']
        
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.stack_size, self.image_size, self.image_size), dtype=np.uint8)
        
    def render(self, mode='rgb_array'):
        return self.env.render(mode=mode)
    
    def _noop_reset_env(self, noop_max=15):
        frame = self.env.reset()

        # some Atari envs require you to press 'FIRE' after a game reset
        frame, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()

        # magic from the OpenAI Atari wrapper
        frame, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()

        # random amount of NOOP actions after a reset
        for i in range(np.random.randint(1, noop_max + 1)):
            frame, _, done, _ = self.env.step(0)
            if done:
                frame = self.env.reset()

        return frame
    
    def _stack_frames(self, frame, frame_stack=None):
        if not frame_stack:
            # frame is the first frame of a new episode
            frame_stack = deque(maxlen=self.stack_size)
            for _ in range(self.stack_size):
                frame_stack.append(frame)
        else:
            frame_stack.append(frame)

        state = np.stack(frame_stack)

        return state, frame_stack
        
    @abstractmethod
    def step(self, action):
        pass
    
    @abstractmethod
    def reset(self):
        pass
        
    @abstractmethod
    def _process_frame(self, frame):
        pass
    
    

class PongWrapper(AtariWrapper):
    """
    Description: 
        Wrapper specific to the Pong environment from OpenAI Gym.
        
    Source: 
        Inspired from the OpenAI baseline wrappers at https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
    """
    
    def __init__(self, env_config):
        super().__init__(env_config)
        
    def step(self, action):
        frame, reward, done, info = self.env.step(action)
        
        # end episode after every life lost
        if reward < 0:
            done = True

        if done:
            frame = np.zeros(self.env.observation_space.shape, dtype=np.uint8)

        state, self.frame_stack = self._stack_frames(self._process_frame(frame), self.frame_stack)
        
        return state, reward, done, info
    
    def reset(self):
        frame = self._noop_reset_env(noop_max=15)
        state, self.frame_stack = self._stack_frames(self._process_frame(frame))
        return state
        
    def _process_frame(self, frame):
        # crop out irrelevant parts of the Atari frame
        frame = frame[34:194, :]

        # convert to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # downscale
        frame = cv2.resize(frame, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

        return frame
