# Introduction-to-Reinforcement-Learning-3

Assignment, ITRL

This repository was originally created as an assignment for the ITRL course. The code presents two different model-based reinforcement learning algorithms; Dyna and Prioritized Sweeping.

The two different algorithms are tested on an evironment where the agent starts on the left-hand side of the environment and has to find the end state on the right side. The surrounding states of the end goal have a property, "blowing" the agent upwards, increasing the difficulty. 

The output of the program are learning curves indicating the performance of each reinforcement learning algorithm. The learning curves are based on different learning speeds, exploration rated and planning updates. The results are averaged over multiple repitions.

In order to run the script call: ```MBRLExperiment.py```

