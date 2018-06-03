import numpy as np
from collections import defaultdict


class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha=0.90
        self.gamma=0.96
        self.epsilon=0.005
        self.i_episode=1
        self.policy_s=None
           
    
    def get_probs(self, Q_s, i_episode, eps=None):
        epsilon=1.0/i_episode
        if eps is not None:
            epsilon=eps
        policy_s=np.ones(self.nA) * epsilon / self.nA
        best_a = np.argmax(Q_s)
        policy_s[best_a]=1-epsilon+(epsilon/self.nA)
        return policy_s
    
    def Update_Q(self, Q_sa, Q_sa_next, reward):
        return Q_sa + (self.alpha *(reward + (self.gamma*Q_sa_next) - Q_sa))
    
    def select_action(self, state):
        self.policy_s=self.get_probs(self.Q[state],self.i_episode,eps=0.0005)
        self.i_episode+=1
        action = np.random.choice(np.arange(self.nA),p=self.policy_s)
        return action

    def step(self, state, action, reward, next_state, done):
        
        self.Q[state][action] = self.Update_Q(self.Q[state][action],np.dot(self.Q[next_state], self.policy_s), reward) 