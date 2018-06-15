Project Notes :

--> The actor model is meant to map states to actions, the critic model needs to map (state, action) pairs to their Q-values. This is reflected in the input layers.

--> Two copies of each model - one local and one target is required because of the "Fixed Q Targets" technique from Deep Q-Learning, and is used to decouple the parameters being updated from the ones that are producing target values.

--> Use Ornstein–Uhlenbeck Noise  process to add some noise to our actions, in order to encourage exploratory behavior. And since actions translate to force and torque being applied to a quadcopter, we want consecutive actions to not vary wildly.


-->SDepending on the random initialization of parameters, sometimes agent may learn a task in the first 20-50 episodes, but you It can be expected most algorithms to be able to learn these tasks in 500-1000 episodes. It is also possible for them to get stuck in local minima, and never make it out . It's possible for training algorithm to take longer, e.g. depending on the learning rate parameter chosen.