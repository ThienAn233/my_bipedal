import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym 
import time as t
import bipedal_walker_env as bpd
from torch.distributions.normal import Normal
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# Global variables
PATH = 'C:\\Users\\Duc Thien An Nguyen\\Desktop\\my_collections\\Python\\bipedal_env\\models\\PPO\\'
# model_path = 'C:\\Users\\Duc Thien An Nguyen\\Desktop\\my_collections\\Python\\bipedal_env\\models\\PPO\\2023-06-23-11-23-14_best_0.55'
# optim_path = 'C:\\Users\\Duc Thien An Nguyen\\Desktop\\my_collections\\Python\\bipedal_env\\models\\PPO\\2023-06-23-11-23-14_best_0.55optim'
log_data = True
save_model = True
render_mode = False
thresh = 0.65

epsilon = 3e-2
explore = 5e-2
gamma = 0.99
learning_rate = 4e-4
number_of_envs = 5
epochs = 500
data_size = 4000
batch_size = 2000
reward_index = np.array([[0.9, 0.1, 0.]])
seed = 3009

# local variables
    # Seed & devices
action_space = 6
observation_space = 27
torch.manual_seed(seed)
np.random.seed(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
print('Using device: ', device)
    # Tensor board
if log_data:
    writer = SummaryWriter('C:\\Users\\Duc Thien An Nguyen\\Desktop\\my_collections\\Python\\bipedal_env\\runs\\PPO\\'+t.strftime('%Y-%m-%d-%H-%M-%S', t.localtime()))
    # Envs setup
env = bpd.SyncVectorEnv(bpd.bipedal_walker,num_of_env=number_of_envs,render_mode=render_mode)
print(f'action space of {number_of_envs} envs is: {action_space}')
print(f'observation sapce of {number_of_envs} envs is: {observation_space}')
class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
    # nn setup
        self.actor = nn.Sequential(
            nn.Linear(observation_space,500),
            nn.LeakyReLU(0.2),
            nn.Linear(500,action_space*2),
        )
        self.critic = nn.Sequential(
            nn.Linear(observation_space,500),
            nn.LeakyReLU(0.2),
            nn.Linear(500,1)
        )
    def forward(self,input):
        return self.actor(input),self.critic(input)


#helper functions
def get_actor_critic_action_and_values(obs,eval=True):
    logits, values = mlp(obs)
    probs = Normal(loc = (torch.pi/4)*nn.Tanh()(logits[:,:action_space]),scale=0.5*nn.Sigmoid()(logits[:,action_space:]))
    if eval is True:
        action = probs.sample()
        return action, probs.log_prob(action)
    else:
        action = eval
        return action, probs.log_prob(action), probs.entropy(), values

class custom_dataset(Dataset):
    
    def __init__(self,data):
        self.obs, self.action, self.logprob, self.reward, self.timestep = data
        self.local_return = [0 for i in range(data_size)]
        self.local_return = torch.hstack(self.get_G()).view(-1,1)
        self.local_observation = torch.vstack(self.obs)
        self.local_action = torch.vstack(self.action)
        self.local_logprob = torch.vstack(self.logprob)
        self.local_reward = torch.hstack(self.reward).view(-1,1)

    def __len__(self):
        return data_size*number_of_envs
    
    def __getitem__(self, index):
        return self.local_observation[index], self.local_action[index], self.local_logprob[index], self.local_return[index], self.local_reward[index]
    
    def isnt_end(self, i):
        ### THE FIRST EPS WILL BE TIMESTEP 1, THE FINAL EP WILL BE TIMESTEP 0
        return self.timestep[i] != 0
    
    def get_G(self):
        ### THE FIRST EPS WILL BE TIMESTEP 1, THE FINAL EP WILL BE TIMESTEP 0
        for  i in range(data_size-1,-1,-1):
            if i == data_size-1:
                self.local_return[i] = self.reward[i]
            else:
                self.local_return[i] = self.reward[i] + self.isnt_end(i)*gamma*self.local_return[i+1]
        return self.local_return


def get_data_from_env():
    ### THE FIRST EPS WILL BE TIMESTEP 1, THE FINAL EP WILL BE TIMESTEP 0
    local_observation = []
    local_action = []
    local_logprob = []
    local_reward = []
    local_timestep = []
    
    observation = env.get_obs()[0]
    local_observation.append(torch.Tensor(observation))
    timestep = np.ones((number_of_envs))
    local_timestep.append(torch.Tensor(timestep.copy()))
    for i in range(data_size) :
        
        # act and get observation 
        action, logprob = get_actor_critic_action_and_values(torch.Tensor(observation).to(device))
        action, logprob = action.cpu(), logprob.cpu()
        local_action.append(torch.Tensor(action))
        local_logprob.append(torch.Tensor(logprob))
        env.step(action)
        observation, reward, info= env.get_obs()
        # print(np.sum(reward*reward_index,axis=-1))
        reward = np.sum(reward*reward_index,axis=-1)
        terminated,truncated = False, info[0]

        # save var
        local_reward.append(torch.Tensor(reward))
        local_observation.append(torch.Tensor(observation))
        local_timestep.append(torch.Tensor(timestep.copy()))
        
        timestep = (1 + timestep)*(1-(terminated | truncated))
    return local_observation, local_action, local_logprob, local_reward, local_timestep

### Normalize the return
mlp = MLP().to(device)
with torch.no_grad():
        data = get_data_from_env()
dataset = custom_dataset(data)
dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
var, mean = torch.var_mean(dataset.local_return)
    # optim setup
mlp_optimizer = torch.optim.Adam(mlp.parameters(),lr = learning_rate)
# mlp.load_state_dict(torch.load(model_path,map_location=device))
# mlp_optimizer.load_state_dict(torch.load(optim_path,map_location=device))
mlp_optimizer.param_groups[0]['lr'] = learning_rate
print(mlp_optimizer.param_groups[0]['lr'])



best_reward = 0
for epoch in range(epochs):
    mlp = mlp.eval()
    # Sample data from the environment
    with torch.no_grad():
        data = get_data_from_env()
    dataset = custom_dataset(data)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
    for iteration, data in enumerate(dataloader):
        mlp = mlp.train()
        
        obs, action, logprob, quality, reward = data
        quality = (quality - mean)/var**0.5
        obs, action, logprob, quality, reward = obs.to(device), action.to(device), logprob.to(device), quality.to(device), reward.to(device)
        next_action, next_logprob, entropy, value = get_actor_critic_action_and_values(obs,eval=action)
        # print(reward-quality)
        # Train models
        mlp_optimizer.zero_grad()
        prob_ratio = torch.exp(next_logprob-logprob)
        # print('qua',quality)
        # print('val',value)
        advantage = quality-value
        critic_loss = (advantage**2).mean()
        entropy_loss = entropy.mean()
        actor_loss = - torch.min( prob_ratio*advantage , torch.clamp(prob_ratio, 1-epsilon, 1+epsilon)*advantage ).mean() - explore*entropy_loss
        loss = critic_loss + actor_loss
        loss.backward()
        mlp_optimizer.step()
        
        #save model
        if save_model:
            if (reward.mean().item()>best_reward and reward.mean().item() > thresh) | ((epoch*(len(dataloader))+iteration) % 1000 == 0):
                best_reward = reward.mean().item()
                torch.save(mlp.state_dict(), PATH+t.strftime('%Y-%m-%d-%H-%M-%S', t.localtime())+'_best_'+str(round(reward.mean().item(),2)))
                torch.save(mlp_optimizer.state_dict(), PATH+t.strftime('%Y-%m-%d-%H-%M-%S', t.localtime())+'_best_'+str(round(reward.mean().item(),2))+'optim')
                print('saved at: '+str(round(reward.mean().item(),2)))
        
        # logging info
        if log_data:
            writer.add_scalar('Eval/minibatchreward',reward.mean().item(),epoch*(len(dataloader))+iteration)
            writer.add_scalar('Eval/minibatchreturn',quality.mean().item(),epoch*(len(dataloader))+iteration)
            writer.add_scalar('Train/entropyloss',entropy_loss.item(),epoch*(len(dataloader))+iteration)
            writer.add_scalar('Train/criticloss',critic_loss.detach().mean().item(),epoch*(len(dataloader))+iteration)
            writer.add_scalar('Train/actorloss',actor_loss.detach().mean().item(),epoch*(len(dataloader))+iteration)
        print(f'[{epoch}]:[{epochs}]|| iter [{epoch*(len(dataloader))+iteration}]: rew: {round(reward.mean().item(),2)} ret: {round(quality.mean().item(),2)} cri: {critic_loss.detach().mean().item()} act: {actor_loss.detach().mean().item()} entr: {entropy_loss.detach().item()}')
torch.save(mlp.state_dict(), PATH+t.strftime('%Y-%m-%d-%H-%M-%S', t.localtime()))
torch.save(mlp_optimizer.state_dict(), PATH+t.strftime('%Y-%m-%d-%H-%M-%S', t.localtime())+'optim')
