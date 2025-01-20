import numpy as np

class e_greedy():
    def __init__(self, bandit, epsilon, init_Q):
        self.Q = np.zeros(bandit.N) # Estimated Values
        self.N = np.zeros(bandit.N) # Number of times each action is taken
        self.epsilon = epsilon
        self.total_actions = bandit.N
        
        if init_Q:
            if len(init_Q)==1:
                self.Q = np.ones(bandit.N)*init_Q
            elif len(init_Q) == len(self.Q):
                self.Q = init_Q
            else:
                raise Exception("init_Q must be scalar or have same shape as bandit.N")
        self.last_action = -1
        
    def action(self):
        if np.random.random()<self.epsilon:
            action = np.random.randint(low=0,high=self.total_actions)
            self.last_action = action
            return action
        else:
            action = np.argmax(self.Q)
            self.last_action = action
            return action # +1 enables 1 indexing used in other scripts
        
    def update(self, reward):
        self.N[self.last_action] +=1
        self.Q[self.last_action] = self.Q[self.last_action] + (reward-self.Q[self.last_action])/self.N[self.last_action]