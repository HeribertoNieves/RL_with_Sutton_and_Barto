import numpy as np

class bandit_experiment():
    def __init__(self, bandit, steps=1000, runs=100, algos=['random'], visualize=False):
        self.bandit = bandit
        self.steps = steps
        self.runs = runs
        self.algos = algos
        self.visualize = visualize
        self.rewards_log = {algo: [] for algo in self.algos}
        self.optimal_action_counts = {algo: np.zeros(self.steps) for algo in self.algos}

    def run(self):
        for algo in self.algos:
            print(f'Running experiment for algorithm: {algo}')
            algo_rewards = []
            
            for run in range(1,self.runs+1):
                print(f'Run {run}/{self.runs}')
                run_rewards = []
                
                for step in range(1,self.steps+1):
                    if step%100==0:
                        print(f'Step {step}/{self.steps}')
                    action = self._action_selection(algo,step)
                    reward = self.bandit.action(action)
                    run_rewards.append(reward)

                    if action == self.bandit.optimal_action():
                        self.optimal_action_counts[algo][step-1]+=1
                
                algo_rewards.append(run_rewards)

            self.rewards_log[algo] = algo_rewards
        
        if self.visualize:
              self.visualize_rewards()          
                    
    def _action_selection(self,algo,step):
        if algo == 'random':
            return np.random.randint(low=1,high=self.bandit.N)
        elif algo == 'perfect':
            return self.bandit.optimal_action()
        else:
            raise NotImplementedError(f'Algorithm "{algo}" not implemented')

    def visualize_rewards(self):
        # Show reward distributions of each action
        self.bandit.visualize()
        
        # Make figure showing traces of average reward 
        fig=go.Figure()
        for key in self.rewards_log.keys():
            rewards_array = np.array(self.rewards_log[key])
            mean_rewards_array = np.mean(rewards_array,axis=0)
            fig.add_trace(go.Scatter(x=np.arange(self.steps),y=mean_rewards_array,name=f'{key}',mode='lines'))
        
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
        for algo in self.optimal_action_counts:
            optimal_percentage = (self.optimal_action_counts[algo]/self.runs)*100
            fig2.add_trace(go.Scatter(x=np.arange(self.steps),y=optimal_percentage,name=f'{algo}',mode='lines'))
        fig2.update_layout(
            title=f"Percentage of Optimal Actions per Step across {self.runs} Runs",
            xaxis_title="Step",
            yaxis_title="Runs that Made Optimal Actions (%)"
        )
        fig2.update_layout(showlegend=True)
        fig2.show()