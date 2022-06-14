
# -*- coding: utf-8 -*-
"""
Model-based Reinforcement Learning experiments
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
2021
By Thomas Moerland
"""
import numpy as np
from MBRLEnvironment import WindyGridworld
from MBRLAgents import DynaAgent,PrioritizedSweepingAgent
from Helper import LearningCurvePlot, smooth

def run_repetitions(policy, n_repetitions, n_timesteps, smoothing_window, learning_rate, gamma,
                    epsilon, n_planning_updates):
   
    # Initialize all rewards
    all_rewards = np.zeros((n_repetitions, n_timesteps))

    # Plotting parameters
    plot = False
    plot_optimal_policy = False
    step_pause = 0.0001
    
    for repetition in range(n_repetitions):

        # Initialize environment and policy
        env = WindyGridworld()
        if policy == 'Dyna':
            pi = DynaAgent(env.n_states,env.n_actions,learning_rate,gamma) # Initialize Dyna policy
        elif policy == 'Priority Sweeping':    
            pi = PrioritizedSweepingAgent(env.n_states,env.n_actions,learning_rate,gamma) # Initialize PS policy
        else:
            raise KeyError('Policy {} not implemented'.format(policy))

        
        # Prepare for running
        s = env.reset()  
        continuous_mode = True
        
        for t in range(n_timesteps):            
            # Select action, transition, update policy
            a = pi.select_action(s,epsilon)
            s_next,r,done = env.step(a)
            pi.update(state =s ,action=a,reward=r,n_state=s_next)

            # Dyna has a different update pattern
            if policy == 'Dyna':
                pi.update_qa(state = s, action = a, n_state = s_next, reward =r)
                pi.planning_updating( n_planning_updates = n_planning_updates)
            # Else Sweeping policy is used which contains compute priority and different planning_updating
            else:
                pi.compute_priority(n_state = s_next, gamma = gamma, reward = r,state = s, action = a)
                pi.planning_updating( n_planning_updates = n_planning_updates, gamma = gamma)

            # Save rewards
            all_rewards[repetition][t] = r

            # Reset environment when terminated
            if done:
                s = env.reset()
            else:
                s = s_next
            
    
    # Add rewards as learning curve
    learning_curve = np.average(all_rewards, axis = 0)   

    # Apply additional smoothing
    learning_curve = smooth(learning_curve,smoothing_window)
    return learning_curve
    


def experiment():

    n_timesteps = 10000
    n_repetitions = 10
    smoothing_window = 101
    gamma = 0.99

    for policy in ['Dyna', 'Priority Sweeping']:
    
        ##### Assignment a: effect of epsilon ######
        print(f"Running assignment part 1a, policy: {policy}")
        learning_rate = 0.5
        n_planning_updates = 5
        epsilons = [0.01,0.05,0.1,0.25]
        Plot = LearningCurvePlot(title = '{}: effect of $\epsilon$-greedy'.format(policy))
        
        for epsilon in epsilons:
            print(f'now running epsilon: {epsilon}')
            learning_curve=run_repetitions(policy, n_repetitions, n_timesteps, smoothing_window, 
                                           learning_rate, gamma, epsilon, n_planning_updates)
            Plot.add_curve(learning_curve,label='$\epsilon$ = {}'.format(epsilon))
        Plot.save('{}_egreedy.png'.format(policy))
        
        ##### Assignment b: effect of n_planning_updates ######
        print(f"Running assignment part 1b, policy: {policy}")

        epsilon=0.05
        n_planning_updatess = [1,5,15]
        learning_rate = 0.5
        Plot = LearningCurvePlot(title = '{}: effect of number of planning updates per iteration'.format(policy))

        for n_planning_updates in n_planning_updatess:
            print(f'now running {n_planning_updates} planning updates')
            learning_curve=run_repetitions(policy, n_repetitions, n_timesteps, smoothing_window, 
                                           learning_rate, gamma, epsilon, n_planning_updates)
            Plot.add_curve(learning_curve,label='Number of planning updates = {}'.format(n_planning_updates))
        Plot.save('{}_n_planning_updates.png'.format(policy))  
        
        ##### Assignment 1c: effect of learning_rate ######
        print(f'Running assignment 1c, policy: {policy}')
        epsilon=0.05
        n_planning_updates = 5
        learning_rates = [0.1,0.5,1.0]
        Plot = LearningCurvePlot(title = '{}: effect of learning rate'.format(policy))
    
        for learning_rate in learning_rates:
            print(f'now running learning rate: {learning_rate}')
            learning_curve=run_repetitions(policy, n_repetitions, n_timesteps, smoothing_window, 
                                           learning_rate, gamma, epsilon, n_planning_updates)
            Plot.add_curve(learning_curve,label='Learning rate = {}'.format(learning_rate))
        Plot.save('{}_learning_rate.png'.format(policy)) 
    
if __name__ == '__main__':
    experiment()