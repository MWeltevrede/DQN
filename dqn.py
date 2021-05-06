from utils import LinearDecay

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from collections import deque
import time, os, shutil, random

class DeepQLearning():
    """
    Description: 
        Double Deep-Learning implementation written in PyTorch.
    """
    
    def __init__(self, env, dqn, total_steps=int(5e5), replay_buffer_size=10000, initial_buffer_size=10000,
                 train_freq=4, target_update_freq=1000, gamma=0.99, lr=0.00025, batch_size=128,
                 max_steps_per_ep=50000, initial_epsilon=1.0, final_epsilon=0.01, num_decay_steps=1e5,
                 save_freq=100000, save_path="models/ddqn", log_path="tensorboard/ddqn", device=torch.device('cpu')):
        """
        Args:
            env: An environment instance that follows the OpenAI Gym API
            
            dqn: A PyTorch Deep (Dueling) Q-Network instance with the following modules:
                ===========  ======================================
                Symbol       Description
                ===========  ======================================
                ``online``   | PyTorch Module, containing
                             | the trainable DQN network.
                ``target``   | PyTorch Module, containing
                             | the fixed target DQN network.
                ===========  ======================================
                
                It also should contain the following function:
                =======================   ==================================  =============================
                Symbol                    Input                               Return
                =======================   ==================================  =============================
                ``forward(x, version)``   | Torch tensor x corresponding to   | Q values corresponding to 
                                          | a batch of states and the         | the states in x.
                                          | version (online/target) on which
                                          | to perform the forward pass.                        
                =======================   ==================================  =============================
            
            total_steps (int): Number of total environment steps to run for.
                
            replay_buffer_size (int): Maximum size of the replay buffer.
            
            initial_buffer_size (int): Number of exploration steps to prepopulate the replay buffer with.
            
            train_freq (int): Frequency with which to train the DQN.
            
            target_update_freq (int): Frequency with which to update the target network.
                
            gamma (float): Discount factor.
                
            lr (float): Learning rate for the neural network updates.
            
            batch_size (int): Batch size used for gradient descent.
            
            max_steps_per_ep (int): Maximum length of an episode. 
            
            initial_epsilon (float): Initial epsilon for the linearly decaying epsilon-greedy strategy.
            
            final_epsilon (float): Final epsilon for the linearly decaying epsilon-greedy strategy.
            
            num_decay_steps (int): Number of steps in which to decay epsilon from initial_epsilon to final_epsilon.
                                   Note that decay only starts after the initial prepopulation of the replay buffer. 

            save_freq (int): Frequency (steps) with which to save the model.
            
            save_path (string): Directory in which to save the models.
            
            log_path (string): Directory in which to log the tensorboard scalars.
                               Note that it will overwrite any logs already in this directory.
            
            device: PyTorch device to be used.
        """
        self.env = env
        self.dqn = dqn
        self.total_steps = total_steps
        self.initial_buffer_size = initial_buffer_size
        self.train_freq = train_freq  
        self.target_update_freq = target_update_freq
        self.lr = lr
        self.gamma = gamma
        self.max_steps_per_ep = max_steps_per_ep
        self.batch_size = batch_size
        self.device = device

        self.save_freq = save_freq
        self.save_path = save_path
        
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
        
        self.optimizer = Adam(self.dqn.online.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()    # Huber loss
        
        self.exploration = LinearDecay(initial_epsilon, final_epsilon, num_decay_steps)
        
        self.replay_memory = deque(maxlen=replay_buffer_size)
        
        self._prepopulate_replay_buffer()
        
        if os.path.exists(log_path):
            shutil.rmtree(log_path)
        self.writer = SummaryWriter(log_dir=log_path)
        
    def _prepopulate_replay_buffer(self):
        """
            Prepopulate the replay buffer with random (exploration) steps before any training starts. 
        """
        state = self.env.reset()

        for s in range(self.initial_buffer_size):

            action = self.env.action_space.sample()

            next_state, reward, done, _ = self.env.step(action)

            self.replay_memory.append((state, action, reward, next_state, done))

            state = next_state

            if done:
                state = self.env.reset()
                
    def _single_env_step(self, state, frame_id, step, episode, ep_rewards, avg_return_100_eps, start):
        """
            Perform a single environment step and add the experience to the replay buffer.
        """
        frame_id += 1
        step += 1
        
        # epsilon greedy strategy
        epsilon = self.exploration.get_epsilon(frame_id)
        if random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            self.dqn.eval()
            with torch.no_grad():
                q_values = self.dqn(torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0), version="online")
                action = q_values.argmax().item()
            self.dqn.train()

        next_state, reward, done, _ = self.env.step(action)

        reward = np.array(reward)
        done = np.array(done)

        ep_rewards.append(reward)

        self.replay_memory.append((state, action, reward, next_state, done))

        if done or step >= self.max_steps_per_ep:
            episode += 1
            ret = sum(ep_rewards)
            avg_return_100_eps.append(ret)

            step = 0
            state = self.env.reset()
            ep_rewards = []

            time_elapsed = time.time() - start
            self.writer.add_scalar("epsilon", epsilon, frame_id)
            self.writer.add_scalar("speed", frame_id/time_elapsed, frame_id)
            self.writer.add_scalar("reward_100", np.mean(np.array(avg_return_100_eps)), frame_id)
            self.writer.add_scalar("reward", ret, frame_id)
        else:
            state = next_state
                
        return state, frame_id, step, episode, ep_rewards
        
    def _train_network(self):
        """
            Perform a single batch update of the DQN network. 
        """
        states, actions, rewards, next_states, dones = zip(*random.sample(self.replay_memory, self.batch_size))

        states = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(np.array(dones), dtype=torch.bool, device=self.device)
        actions = torch.as_tensor(np.array(actions), dtype=torch.long, device=self.device)

        q_values = self.dqn(states, "online")
        q_values_next_target = self.dqn(next_states, "target").detach()
        q_values_next_online = self.dqn(next_states, "online")

        v = q_values.index_select(dim=1, index=actions).diagonal()    # torch magic to index using a tensor of indices

        # double q learning
        idxs = q_values_next_online.argmax(axis=1)
        v_next = q_values_next_target.index_select(dim=1, index=idxs).diagonal() 
        targets = rewards + self.gamma*v_next*(~dones)

        self.optimizer.zero_grad()
        loss = self.criterion(v , targets)
        loss.backward()
        self.optimizer.step()
                
    def train(self, print_freq=10000):
        """
            Train the agent using Double Deep Q Learning.
        """
        frame_id = 0   # number of total steps so far
        step = 0       # number of steps so far in current episode
        state = self.env.reset()

        ep_rewards = []
        avg_return_100_eps = deque(maxlen=100)
        episode = 0 

        start = time.time()
        while frame_id < self.total_steps:
            # collect experience
            state, frame_id, step, episode, ep_rewards = self._single_env_step(state, frame_id, step, episode, ep_rewards, avg_return_100_eps, start)
            
            # train
            if frame_id % self.train_freq == 0:
                self._train_network()
                
            # print results
            if frame_id % print_freq == 0:
                print(f"Frame ID: {frame_id}, episode: {episode}, average return last 100 episodes: {np.mean(np.array(avg_return_100_eps))}, fps: {frame_id / (time.time() - start)}")

            # update target network
            if frame_id % self.target_update_freq == 0:
                self.dqn.target.load_state_dict(self.dqn.online.state_dict())

            # save model
            if frame_id % self.save_freq == 0:
                print(f"Saved model at step: {frame_id}")
                filename = self.save_path + f"/frame{int(frame_id / 1000)}k.pt"
                torch.save(self.dqn.online.state_dict(), filename)

        self.writer.close()