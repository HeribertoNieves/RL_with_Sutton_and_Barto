import numpy as np

class e_greedy():
    def __init__(self, bandit, epsilon, init_Q=[0]):
        self.Q = np.zeros(bandit.N) # Estimated Values
        self.N = np.zeros(bandit.N) # Number of times each action is taken
        self.epsilon = epsilon
        self.total_actions = bandit.N
        
        if init_Q is not None:
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
            return action
        
    def update(self, reward):
        self.N[self.last_action] +=1
        self.Q[self.last_action] = self.Q[self.last_action] + (reward-self.Q[self.last_action])/self.N[self.last_action]

class UCB():
    def __init__(self, bandit, c, init_Q=[0]):
        self.Q = np.zeros(bandit.N)
        self.N = np.zeros(bandit.N)
        self.c = c
        self.total_actions = bandit.N
        self.t = 1

        if init_Q is not None:
            if len(init_Q)==1:
                self.Q = np.ones(bandit.N)*init_Q
            elif len(init_Q) == len(self.Q):
                self.Q = init_Q
            else:
                raise Exception("init_Q must be scalar or have same shape as bandit.N")
        self.last_action = -1

    def action(self):
        untried_actions = np.where(self.N == 0)[0]
        if len(untried_actions)>0: # first do actions that have not been done
            action = untried_actions[0]
        else:
            UCB = self.Q+self.c * np.sqrt(np.log(self.t)/self.N)
            action = np.argmax(UCB)
        self.last_action = action
        self.t+=1
        self.N[action]+=1
        return action
    
    def update(self,reward):
        self.N[self.last_action]+=1
        self.Q[self.last_action] = self.Q[self.last_action] + (reward-self.Q[self.last_action])/self.N[self.last_action]