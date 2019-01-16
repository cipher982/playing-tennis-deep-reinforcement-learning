[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://raw.githubusercontent.com/cipher982/playing-tennis-deep-reinforcement-learning/master/images/Screen%20Shot%202019-01-15%20at%2011.06.50%20PM.png
[image3]:https://raw.githubusercontent.com/cipher982/playing-tennis-deep-reinforcement-learning/master/images/final_score.png


# Playing Tennis with Dual Agents
![image1]

In this environment there are two agents that play against eachother simultaneously, and must cooperate to keep the ball in play as long as they can. The goal (to solve the environment) is to reach an average of 0.5 over 100 consecutive episodes.

This agent is fairly similar to the previous excersize here https://github.com/cipher982/Robotic-Control-in-Unity-with-DRL where I had 20 agents running at the same time. The difference here is that they must cooperate and share information of the same state space, so when running the the neural networks we actually combine the actions of both agents to the same network.

The ideas behind all of this is layed out in the paper **Multi-Agent Actor-Critic for Mixed
Cooperative-Competitive Environment** by the OpenAI team in 2018. It discusses the issue of the increasing variance that occurs when increasing agent count, and lays out methods of coordination where the collections of agents discover said coordinating strategies.

### Why MADDPG?
Like standard actor/critic model the idea using two separate networks (actor/critic) enables the model to generalize and estimate reward values (using the critic) from the chosen action of the actor, therefore reducing the need for as much exploration of state/value combinations. This implentation takes that further and enables the coordination between agents, which is precisely what is needed for this tennis environment.

### Incorporating MADDPG to this environment
The state-space of the environment is 8 per agent (16) with 2 actions per agent (4). The hyperparameters I chose are listed below:
- BATCH_SIZE = 128        # minibatch size
- BUFFER_SIZE = int(1e6)  # replay buffer size
- GAMMA = 0.99            # discount factor
- LR_ACTOR = 1e-3         # learning rate of the actor
- LR_CRITIC = 1e-3        # learning rate of the critic
- SEED = 42
- TAU = 6e-2    
- Model Weights:
    - Actor(Layer1=256, Layer2=128) 
    - Critic(Layer1=256, Layer2=128)
    
Final chart of performance below:
![image3]
  
### A quick overview of the learning process
1. As usual with reinforcement learning we **begin with random noise** values to begin the exploration process. 
2. With each step of taking an action we **add to the experiences memory**.
    - For the coordination of the agents, we add their actions and experiences together to one policy.
3. If enough experiences get loaded in to the memory we can start the learning steps:
    a. **Update the Critic**: compute Q-targets, compute Critic loss, minimize the loss
    b. **Update the Actor**: compute Actor loss, minimize the loss
    c. **Update the target networks** (as in the DQN algorithm, we like to hold separate target/local models and periodically update the target to enable more steady aiming towards the goal and stable learning overall)
4. Periodically check for the average (mean) score over the previous 100 episodes to see if an average score of +0.5 was reach. If so, consider it solved!

### Issues
With this project I kept encountering crashing performance, where it either rose too slowly, or rose too fast and then went back down. As you can see in the images below I tried a few different attempts before getting it right.

![image2]

### Exploding Training Time
As the episodes went on the training time began to expand from ~.3 seconds an episode to almost 15 seconds an episode. I'm not sure why this occurs but it seemed to correlate with higher performance scores.


### Future Work
I think an automated or scientific method of setting hyperparameters could be very useful in the future. Due to the training time and possible options to set, I mostly just sat around and plugged a few different numbers in to watch the performance. In the end I just went with what seemed right and performed best within my constraints. I'm sure I left a lot of options on the table as for as optimization. 

There are also of course numerous different algorithms and methods for reinforcement learning that are now popular, and this is only one of them. A recent approach has been using ensemble methods of combining multiple methods, such as Rainbow.


## Running the Model
1. Download the environment from one of the links below. You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
2. Place the file in the DRLND GitHub repository, and unzip (or decompress) the file.
3. Open CMD/Terminal and run 'pip install -r requirements.txt'
4. Run 'python main.py' in your terminal to begin the training process.



