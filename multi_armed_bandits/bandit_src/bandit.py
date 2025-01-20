import numpy as np
import pandas as pd
import plotly.graph_objects as go

class n_armed_bandit():
    def __init__(self,N=10,mean_range = [-3,3], std_range = [0.5,1.5], standard = False):
        self.N = N
        if standard:
            self.mean_range = [-3,3]
            self.std_range = [0.99,1]
        else:
            self.mean_range = mean_range
            self.std_range = std_range

        self.means = np.round(np.random.uniform(low=self.mean_range[0], high=self.mean_range[1], size=self.N),2)
        self.stds = np.round(np.random.uniform(low=self.std_range[0],high=self.std_range[1],size=self.N),2)
        
    def inspect(self):
        print(f'Action Range: Means {self.mean_range} and STDev Range {self.std_range}')
        for i in range(0,self.N):
            print(f'Action {i}: Mean {self.means[i-1]} and STDev {self.stds[i-1]}')

    def action(self,action_num):
        if action_num<0 or action_num>self.N-1:
            print(f'Invalid action. Actions must be between 0 and {self.N-1}')
        else:
            return np.random.normal(loc=self.means[action_num], scale=self.stds[action_num])

    def optimal_action(self):
        '''
        Return the action number of the action with the highest mean reward
        '''
        return np.argmax(self.means) 
    

    def visualize(self):
        action_numbers = np.arange(0, self.N)  # Action indices (x-axis)
        fig = go.Figure()
        
        for action, mean, std in zip(action_numbers, self.means, self.stds):
            # Generate random samples for the rewards of the action
            rewards = np.random.normal(loc=mean, scale=std, size=1000)
            # Add a violin plot for the action
            fig.add_trace(
                go.Violin(
                    x=[action] * len(rewards),  # Position along x-axis
                    y=rewards,                 # Reward values (spread along y-axis)
                    name=f"Action {action}",   # Label for the action
                    box_visible=True,          # Show the box plot
                    meanline_visible=True      # Show the mean line
                )
            )
        
        fig.update_layout(
            title="Spread of Rewards for Actions",
            xaxis_title="Action Number",
            yaxis_title="Reward Value",
            xaxis=dict(tickmode='linear', tick0=1, dtick=1),  # Ensure action numbers are evenly spaced
            violingap=0,  # Gap between violins
        )
        fig.update_layout(showlegend=False)
        
        # Show the plot
        fig.show()