# Memory-Management-with-RL
A Reinforcement Learning Agent Managing Allocation Of Memory 

# Setup
Clone the repository:
```shell
$ git clone https://github.com/blurry-mood/Memory-Management-with-RL
```
Install Requirements (or first create a virual environment):
```shell
$ cd Memory-Management-with-RL
$ pip3 install -r requirements
```

# Train
To train a Q-learning agent, use these two commands:
```shell
$ cd Memory-Management-with-RL/src
$ python train_agent.py
```
To compare against a random agent, use:
```shell
$ cd Memory-Management-with-RL/src
$ python random_agent.py
```
Both agents run for 1000 episodes, after which, a plot depicting the reward evolution is saved.

# Environment
TODO