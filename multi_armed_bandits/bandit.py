import numpy as np
import pandas as pd
import plotly.graph_objects as go

class n_armed_bandit():
    def __init__(self,N=10,mean_range = [-3,3], std_range = [0.5,1.5], standard = False):
        '''
        Set up the N-Armed Bandit Platform for Experimentation
        Inputs:
            N - Number of arms
            mean_range - range of means to randomly choose the mean reward of each arm
            std_range - range of standard deviations to randomly choose the mean reward of each arm
            standard - sets mean_range to [-3,3] and std_range to approximately 1
        '''
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
        '''
        Print out Reward Range Descriptions and the Mean with Standard Deviation of each Action
        '''
        print(f'Action Reward Range: Means {self.mean_range} and STDev Range {self.std_range}')
        for i in range(1,self.N+1):
            print(f'Action {i}: Mean {self.means[i-1]} and STDev {self.stds[i-1]}')

    def action(self, action_num):
        '''
        Use the inputted arm of the N-Armed Bandit. Outputs a reward based on the probability distribution of the selected arm
        Input:
            action_num - the number of the arm that is selected
        '''
        if action_num<1 or action_num>self.N:
            print(f'Invalid action. Actions must be between 1 and {self.N}')
        else:
            return np.random.normal(loc=self.means[action_num], scale=self.stds[action_num])
   
    def optimal_action(self):
        '''
        Return the action number of the action with the highest mean reward
        '''
        return np.argmax(self.means) + 1 
        
    def visualize(self):
        '''
        Plot out the Reward Distributions of each Arm 
        '''
        action_numbers = np.arange(1, self.N+1)  # Action indices (x-axis)
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