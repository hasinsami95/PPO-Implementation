import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions.categorical import Categorical

class store_steps:
    def __init__(self, batch_size):
       self.states=[]
       self.values=[]
       self.actions=[]
       self.probabilities=[]
       self.rewards=[]
       self.ends=[]
       self.batch_size=batch_size

    def create_batch(self):
        num_states =len(self.states)
        batch_start= np.arange(0, num_states, self.batch_size)
        indices=np.arange(num_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches=[indices[i:i+self.batch_size] for i in batch_start]
        return np.array(self.states), np.array(self.actions),\
               np.array(self.probabilities), np.array(self.values),\
               np.array(self.rewards), np.array(self.ends), batches

    def save_steps(self, state, action, prob, val, reward, end):
        self.states.append(state)
        self.probabilities.append(prob)
        self.actions.append(action)
        self.values.append(val)
        self.rewards.append(reward)
        self.ends.append(end)

    def delete_memory(self):
        self.states=[]
        self.probabilities=[]
        self.values=[]
        self.actions=[]
        self.rewards=[]
        self.ends=[]

class ActorPolicy(nn.Module):
    def __init__(self, num_actions, in_dim, alpha, hidden_dim1=256, hidden_dim2=256, chkpt_dir='tmp'):
        super(ActorPolicy, self).__init__()

        self.file=os.path.join(chkpt_dir, 'actor_ppo') # Creates a file named actor_ppo inside the directory chkpt_dir
        self.actor_net=nn.Sequential(
            nn.Linear(*in_dim, hidden_dim1), # * unpacks the input dimension
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, num_actions),
            nn.Softmax(dim=-1)
        )

        self.optim=optim.Adam(self.parameters(), lr=alpha)
        self.device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        dist = self.actor_net(state)
        dist= Categorical(dist) # Gives a list of probabilites, considers exploration as well

        return dist

    def save_checkpoints (self):
        torch.save(self.state_dict(), self.file)

    def retrieve_checkpoints(self):
        self.load_state_dict(torch.load(self.file))

class CriticNet(nn.Module):
    def __init__(self, in_dim, alpha, hidden_dim1=256, hidden_dim2=256, chkpt_dir='tmp'):
        super(CriticNet, self).__init__()
        self.file=os.path.join(chkpt_dir, 'critic_ppo')
        self.critic_net= nn.Sequential(nn.Linear(*in_dim, hidden_dim1), 
                                           nn.ReLU(),
                                           nn.Linear(hidden_dim1, hidden_dim2),
                                           nn.ReLU(),
                                           nn.Linear(hidden_dim2, 1)
            )   

        self.optim = optim.Adam(self.parameters(), lr=alpha)
        self.device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic_net(state)
        return value

    def save_checkpoints (self):
        torch.save(self.state_dict(), self.file)

    def retrieve_checkpoints(self):
        self.load_state_dict(torch.load(self.file))

        return value

class Agent:
    def __init__(self, num_actions, gamma=0.99, alpha= 0.0003, lambda_factor=0.95, clip_factor=0.1, batch_size=64, N=2048, num_epoch=10,in_dim=1): # N is the number of rollouts
            self.gamma=gamma
            self.clip_factor=clip_factor
            self.num_epoch=num_epoch
            self.lambda_factor=lambda_factor
            self.actor= ActorPolicy(num_actions,in_dim, alpha)
            self.critic= CriticNet(in_dim, alpha)
            self.memory = store_steps(batch_size)

    def remember(self, state, action, prob, val, reward, end):
            self.memory.save_steps(state, action, prob, val, reward, end )

    def save_models(self):
            print('.... saving the models....')
            self.actor.save_checkpoints()
            self.critic.save_checkpoints()

    def retrieve_models(self):
            self.actor.retrieve_checkpoints()
            self.critic.retrieve_checkpoints()

    def select_action(self, option):
            state=torch.tensor([option], dtype=torch.float).to(self.actor.device)
            #print(state)

            dist=self.actor(state)
            #print(dist)
            value=self.critic(state)
            action=dist.sample() # Picks a stochastic action based on distribution, includes exploration 
            #print(action)
            probabilites=torch.squeeze(dist.log_prob(action)).item() # this is logP(action|state) needed for PPO loss calculation
            #print(probabilites)
            action=torch.squeeze(action).item() #.item() converts tensor to float
            # print(value)
            # exit()
            value=torch.squeeze(value).item()
            return action, probabilites, value

    def train(self):
            for _ in range(self.num_epoch):
                states_arr, actions_arr, old_probs_arr, vals_arr, rewards_arr, ends_arr, batches= self.memory.create_batch()
                
                advantage=np.zeros(len(rewards_arr), dtype=np.float32)

                for t in range(len(rewards_arr)-1):
                    discount_factor=1

                    adv_t=0

                    for i in range(t, len(rewards_arr)-1):
                        adv_t+=discount_factor*(rewards_arr[i]+self.gamma*vals_arr[i+1]*(1-int(ends_arr[i]))-vals_arr[i])
                        discount_factor*=self.gamma*self.lambda_factor

                    advantage[t]=adv_t

                advantage= torch.tensor(advantage).to(self.actor.device)
                vals_arr= torch.tensor(vals_arr).to(self.actor.device)
                for batch in batches:
                    #print("batch is ", batch)
                    states=torch.tensor(states_arr[batch], dtype=torch.float).to(self.actor.device)
                    old_probs=torch.tensor(old_probs_arr[batch]).to(self.actor.device)
                    actions=torch.tensor(actions_arr[batch]).to(self.actor.device)

                    dist=self.actor(states)
                    critic_val=self.critic(states)
                    
                    #print("vals arr is ", vals_arr[batch])
                    #exit()

                    critic_val=torch.squeeze(critic_val)
                    #print("critic val is ", critic_val)
                    new_probs=dist.log_prob(actions)
                    prob_ratio=new_probs.exp()/old_probs.exp()
                    #print("prob ratio is ", prob_ratio)
                    scaled_probs= advantage[batch]*prob_ratio
                    scaled_clipped_probs=torch.clamp(prob_ratio, 1-self.clip_factor, 1+self.clip_factor)*advantage[batch]
                    actor_loss=- torch.min(scaled_probs, scaled_clipped_probs).mean() # the loss is negative since gradient ascent
                    returns=advantage[batch]+vals_arr[batch]
                    # print("returns is ", returns)
                    # print("advantage is ", advantage[batch])
                    # print("actor loss is ", actor_loss)
                    # exit()
                    critic_loss= (returns-critic_val)**2
                    critic_loss=critic_loss.mean()
                
                    total_loss=actor_loss + 0.5*critic_loss
                    self.actor.optim.zero_grad()
                    self.critic.optim.zero_grad()
                    total_loss.backward()
                    self.actor.optim.step()
                    self.critic.optim.step()

            self.memory.delete_memory()

                





                
