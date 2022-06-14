#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-based Reinforcement Learning policies
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
2021
By Thomas Moerland
"""
import numpy as np
from queue import PriorityQueue
from regex import R
from MBRLEnvironment import WindyGridworld
from random import randint

class DynaAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))
        self.T_counts = np.zeros((n_states, n_actions, n_states))
        self.R_sum = np.zeros((n_states, n_actions, n_states))
        self.dyna_model = np.zeros((n_states, n_actions, 2))

         
        
    def select_action(self, state, epsilon):
        rand_number = np.random.rand()                            
        
        if rand_number > epsilon:       
            return np.argmax(self.Q_sa[int(state)])
        else:                                                              
            action = np.random.choice(self.n_actions)
            while action == np.argmax(self.Q_sa[int(state)]):
                action = np.random.choice(self.n_actions)
            return action
        
    def update(self,state,action,reward,n_state):
        # Update transition counts and rewards
        self.T_counts[state][action][n_state] = self.T_counts[state][action][n_state] + 1
        self.R_sum[state][action][n_state] = self.R_sum[state][action][n_state] + reward

        # Prepare estimation functions | all_counts, t_est_count -> transition function
        all_counts: int = self.T_counts[state][action].mean()                                                                 
        t_est_count = np.zeros(self.n_states)

        # Estimate transition function
        for s in range(self.n_states):                                                                                          
            t_est_count[s] = self.T_counts[state][action][s]/ all_counts
        
        self.dyna_model[state][action][0]= t_est_count.argmax()                              

        # Estimate reward function
        self.dyna_model[state][action][1] = self.R_sum[state][action][n_state] / self.T_counts[state][action][n_state]          

    def update_qa (self, state, action, n_state, reward):
        # Updating Q-table
        self.Q_sa[state][action] = self.Q_sa[state][action] + self.learning_rate * (reward + self.gamma * np.max(self.Q_sa[n_state]) - self.Q_sa[state][action] )

    def random_state_action(self):
        # Retrieve random state and action
        r_state: np.array = self.T_counts[s := randint(0,self.n_states - 1)]        
        r_action: int = randint(0,3)

        # Loop over r_state until state has been found where agent has been, do the same with action in state
        while r_state.sum() == 0:
            r_state: np.array = self.T_counts[s := randint(0,self.n_states - 1)]
            if r_state.sum() > 0:
                while r_state[r_action].sum() == 0:
                    r_action = randint(0,3)

        # Return the random state and action
        return  s, r_action


    def planning_updating(self, n_planning_updates):
        for planning_update in range (n_planning_updates): 
            # Retrieve random state and action, retrieve reward and next state from Dyna model
            r_state, r_action = self.random_state_action()
            reward = self.dyna_model[r_state][r_action][1]
            n_state: int = int(self.dyna_model[r_state][r_action][0])

            # Update Q-table based on planning update
            self.Q_sa[r_state][r_action] = self.Q_sa[r_state][r_action] + self.learning_rate * (reward + self.gamma * np.max(self.Q_sa[n_state]) - self.Q_sa[r_state][r_action])
    
    def reward_summation(self):
        # Return summation of rewards
        return (self.R_sum.sum())

class PrioritizedSweepingAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma, max_queue_size=200, priority_cutoff=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.priority_cutoff = priority_cutoff
        self.queue = PriorityQueue()
        self.Q_sa = np.zeros((n_states,n_actions))
        self.T_counts = np.zeros((n_states, n_actions, n_states))
        self.R_sum = np.zeros((n_states, n_actions, n_states))
        self.sweeping_model = np.zeros((n_states, n_actions, 2))
        

    def select_action(self, state, epsilon):
      rand_number = np.random.rand()                            
        
      if rand_number > epsilon:                                     
            return np.argmax(self.Q_sa[state])
      else:                                                              
          action = np.random.choice(self.n_actions)
          while action == np.argmax(self.Q_sa[state]):
              action = np.random.choice(self.n_actions)
          return action
    
    def compute_priority(self, n_state, gamma, reward,state, action):
        p =abs(( (reward + gamma * (np.max(self.Q_sa[n_state]) - self.Q_sa[state][action]))))
        if p > 0.01:
            state_action = (state, action)
            self.queue.put((-p, state_action))
        return p
        
        
        
    def update(self,state,action,reward,n_state):
        
        # Update transition counts and rewards
        self.T_counts[state][action][n_state] = self.T_counts[state][action][n_state] + 1
        self.R_sum[state][action][n_state] = self.R_sum[state][action][n_state] + reward

        # Prepare estimation functions | all_counts, t_est_count -> transition function
        all_counts: int = self.T_counts[state][action].mean()                                                                 
        t_est_count = np.zeros(self.n_states)

        # Estimate transition function
        for s in range(self.n_states):                                                                                          
            t_est_count[s] = self.T_counts[state][action][s]/ all_counts
        
        self.sweeping_model[state][action][0]= t_est_count.argmax()                              

        # Estimate reward function
        self.sweeping_model[state][action][1] = self.R_sum[state][action][n_state] / self.T_counts[state][action][n_state]    
        

    def random_state_action(self):
        # Retrieve random state and action
        r_state: np.array = self.T_counts[s := randint(0,self.n_states - 1)]        
        r_action: int = randint(0,3)

        # Loop over r_state until state has been found where agent has been, do the same with action in state
        while r_state.sum() == 0:
            r_state: np.array = self.T_counts[s := randint(0,self.n_states - 1)]
            if r_state.sum() > 0:
                while r_state[r_action].sum() == 0:
                    r_action = randint(0,3)

        # Return the random state and action
        return  s, r_action


    def planning_updating(self, n_planning_updates, gamma):
        for planning_update in range (n_planning_updates): 
            if self.queue.empty() == True:
                break
            else:
                # Pop highest priority from PQ
                _,(s,a) = self.queue.get()

                # Retrieve random state and action, retrieve reward and next state from Sweeping model
                #r_state, r_action = self.random_state_action()
                reward = self.sweeping_model[s][a][1]
                n_state: int = self.sweeping_model[s][a][0]

                # Update Q-table based on planning update
                self.Q_sa[s][a] = self.Q_sa[s][a] + self.learning_rate * (reward + self.gamma * np.max(self.Q_sa[int(n_state)]) - self.Q_sa[s][a])

                # Updating priority from previous states
                for past_state in range(s-15, s + 15):
                    if past_state < 0 or past_state >= 70:
                        continue
                    for past_action in range(4):
                        if self.T_counts[past_state][past_action][s] > 0:
                            past_reward = self.sweeping_model[past_state][past_action][1]
                            self.compute_priority( n_state = s, gamma = gamma, reward = past_reward, state = past_state, action = past_action)

    def reward_summation(self):
        # Return summation of rewards
        return (self.R_sum.sum())

            
def test():

    n_timesteps = 10000
    gamma = 0.99
    all_rewards = np.zeros(n_timesteps)

    # Algorithm parameters
    policy = 'dyna' 
    epsilon = 0.1
    learning_rate = 0.5
    n_planning_updates = 5

    # Plotting parameters
    plot = True
    plot_optimal_policy = True
    step_pause = 0.0001
    
    # Initialize environment and policy
    env = WindyGridworld()
    if policy == 'dyna':
        pi = DynaAgent(env.n_states,env.n_actions,learning_rate,gamma) # Initialize Dyna policy
    elif policy == 'ps':    
        pi = PrioritizedSweepingAgent(env.n_states,env.n_actions,learning_rate,gamma) # Initialize PS policy
    else:
        raise KeyError('Policy {} not implemented'.format(policy))
    
    # Prepare for running
    s = env.reset()  
    continuous_mode = False
    
    for t in range(n_timesteps):            
         # Select action, transition, update policy
        a = pi.select_action(s,epsilon)
        s_next,r,done = env.step(a)
        pi.update(state =s ,action=a,reward=r,n_state=s_next)
        if policy == 'dyna':
            pi.update_qa(state = s, action = a, n_state = s_next, reward =r)
            pi.planning_updating( n_planning_updates = n_planning_updates)
            
        if policy == "ps":
            pi.compute_priority(n_state = s_next, gamma = gamma, reward = r,state = s, action = a)
            pi.planning_updating( n_planning_updates = n_planning_updates, gamma = gamma, timestep = t)

        # Save rewards
        all_rewards[t] = r
        #print(all_rewards)
        
        # Render environment
        if plot:
            env.render(Q_sa=pi.Q_sa,plot_optimal_policy=plot_optimal_policy,
                       step_pause=step_pause)
            
        # Ask user for manual or continuous execution
        if not continuous_mode:
            key_input = input("Press 'Enter' to execute next step, press 'c' to run full algorithm")
            continuous_mode = True if key_input == 'c' else False

        # Reset environment when terminated
        if done:
            s = env.reset()
        else:
            s = s_next
            
    
if __name__ == '__main__':
    test()
