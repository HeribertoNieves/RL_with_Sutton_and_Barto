import numpy as np
import plotly.graph_objects as go
import bandit_src.algorithms as algorithms


class bandit_experiment():
    def __init__(self, bandit, steps=1000, runs=100, algos=['random'], epsilon = [0], c=[0], init_Q = [0], visualize=False):
        self.bandit = bandit
        self.steps = steps
        self.runs = runs
        self.algos = algos
        self.visualize = visualize
        self.rewards_log = {algo: [] for algo in self.algos}
        self.actions_log = {algo: [] for algo in self.algos}
        self.optimal_action_counts = {algo: np.zeros(self.steps) for algo in self.algos}
        self.epsilon = epsilon
        self.epsilon_ct = -1
        self.init_Q = init_Q
        self.c = c
        self.c_ct = -1
        self.previous_reward = None

    def run(self):
        for algo in self.algos:
            print(f'Running experiment for algorithm: {algo}')
            algo_rewards = []
            algo_actions = []
            if 'e_greedy' in algo:
                self.epsilon_ct+=1
            elif 'UCB' in algo:
                self.c_ct += 1
            self.previous_reward = None
            for run in range(1,self.runs+1):
                print(f'Run {run}/{self.runs}')
                run_rewards = []
                run_actions = []
                
                for step in range(1,self.steps+1):
                    if step%100==0:
                        print(f'Step {step}/{self.steps}')
                    action = self._action_selection(algo,step,self.previous_reward)
                    reward = self.bandit.action(action)
                    self.previous_reward= reward
                    run_rewards.append(reward)
                    run_actions.append(action)

                    if action == self.bandit.optimal_action():
                        self.optimal_action_counts[algo][step-1]+=1
                
                algo_rewards.append(run_rewards)
                algo_actions.append(run_actions)

            self.rewards_log[algo] = algo_rewards
            self.actions_log[algo] = algo_actions
        
        if self.visualize:
              self.visualize_rewards()          
                    
    def _action_selection(self,algo,step,previous_reward=None):
        if algo == 'random':
            return np.random.randint(low=0, high=self.bandit.N)
        elif algo == 'perfect':
            return self.bandit.optimal_action()
        elif 'e_greedy' in algo:
            if step==1:
                self.e_greedy = algorithms.e_greedy(self.bandit, self.epsilon[self.epsilon_ct], self.init_Q)
            else:
                self.e_greedy.update(previous_reward)
            return self.e_greedy.action()
        elif 'UCB' in algo:
            if step==1: # self, bandit, c, init_Q=0
                self.UCB = algorithms.UCB(self.bandit,self.c[self.c_ct],self.init_Q)
            else:
                self.UCB.update(previous_reward)
            return self.UCB.action()
        else:
            raise NotImplementedError(f'Algorithm "{algo}" not implemented')

    def visualize_rewards(self):
        # Show reward distributions of each action
        self.bandit.visualize()
        self.epsilon_ct=-1
        self.c_ct=-1
        # Make figure showing traces of average reward 
        fig=go.Figure()
        for key in self.rewards_log.keys():
            rewards_array = np.array(self.rewards_log[key])
            mean_rewards_array = np.mean(rewards_array,axis=0)
            if 'e_greedy' in key:
                self.epsilon_ct+=1
                name = key + f' Epsilon: {self.epsilon[self.epsilon_ct]}'
            elif 'UCB' in key:
                self.c_ct+=1
                name = key + f' c: {self.c[self.c_ct]}'
            else:
                name = key
            fig.add_trace(go.Scatter(x=np.arange(self.steps),y=mean_rewards_array,name=f'{name}',mode='lines'))
        
        fig.update_layout(
            title=f"Average Reward of each Step across {self.runs} Runs",
            xaxis_title="Step",
            yaxis_title="Average Reward Value",
            #xaxis=dict(tickmode='linear', tick0=1, dtick=1),  # Ensure action numbers are evenly spaced
        )
        fig.update_layout(showlegend=True)
        fig.show()

        # Make figure showing % of optimal action completed
        fig2 = go.Figure()
        self.epsilon_ct=-1
        self.c_ct=-1
        for algo in self.optimal_action_counts:
            if 'e_greedy' in algo:
                self.epsilon_ct+=1
                name = algo + f' Epsilon: {self.epsilon[self.epsilon_ct]}'
            elif 'UCB' in algo:
                self.c_ct+=1
                name = algo + f' c: {self.c[self.c_ct]}'
            else:
                name = algo
            optimal_percentage = (self.optimal_action_counts[algo]/self.runs)*100
            fig2.add_trace(go.Scatter(x=np.arange(self.steps),y=optimal_percentage,name=f'{name}',mode='lines'))
        fig2.update_layout(
            title=f"Percentage of Optimal Actions per Step across {self.runs} Runs",
            xaxis_title="Step",
            yaxis_title="Runs that Made Optimal Actions (%)"
        )
        fig2.update_layout(showlegend=True)
        fig2.show()


