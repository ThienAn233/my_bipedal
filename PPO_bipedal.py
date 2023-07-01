import torch
import torch.nn as nn
import numpy as np
import time as t
import my_bipedal.bipedal_walker_env as bpd
from torch.distributions.normal import Normal
from torch.utils.data import Dataset, DataLoader
from torchrl.modules import TanhNormal
from torch.utils.tensorboard import SummaryWriter

class PPO_bipedal_walker_train():
    def __init__(self,
                # Global variables
                PATH = None,
                load_model = None,
                log_data = True,
                save_model = True,
                render_mode = False,
                thresh = 0.65,

                epsilon = 0.2,
                explore = 1e-4,
                gamma = 0.99,
                learning_rate = 4e-4,
                number_of_envs = 10,
                epochs = 500,
                data_size = 4000,
                batch_size = 2000,
                reward_index = np.array([[0., 0.8, 0.2]]),
                seed = 3009,
                mlp = None,

                # local variables
                # Seed & devices
                action_space = 6,
                observation_space = 51,
                device = None):

                    
                    
        # Global variables
        self.PATH = PATH
        self.load_model = load_model
        self.log_data = log_data
        self.save_model = save_model
        self.render_mode = render_mode
        self.thresh = thresh
        self.epsilon = epsilon
        self.explore = explore
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.number_of_envs = number_of_envs
        self.epochs = epochs
        self.data_size = data_size
        self.batch_size = batch_size
        self.reward_index = reward_index
        self.seed = seed
        self.mlp = mlp


                    
        # local variables
        # Seed & devices
        self.action_space = action_space
        self.observation_space = observation_space
        self.device = device
        
        
        
        # Load model and device
        if load_model:
            self.model_path = PATH + '//models//PPO//' + load_model
            self.optim_path = PATH + '//models//PPO//' + load_model + 'optim'
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        print(f'Using seed: {self.seed}')
        if self.device:
            pass
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print('Using device: ', self.device)
        # Tensor board
        if self.log_data:
            self.writer = SummaryWriter(PATH + '//runs//PPO//'+t.strftime('%Y-%m-%d-%H-%M-%S', t.localtime()))
            # Envs setup
        self.env = bpd.SyncVectorEnv(bpd.bipedal_walker,num_of_env=self.number_of_envs,render_mode=self.render_mode)
        print(f'action space of {number_of_envs} envs is: {action_space}')
        print(f'observation sapce of {number_of_envs} envs is: {observation_space}')
        
        
        
        # Setup MLP
        if self.mlp:
            self.mlp.to(self.device)
            pass
        else:
            class MLP(nn.Module):
                def __init__(self):
                    super(MLP,self).__init__()
                # nn setup
                    self.actor = nn.Sequential(
                        nn.Linear(observation_space,500),
                        nn.Tanh(),
                        nn.Linear(500,action_space*2),
                    )
                    self.critic = nn.Sequential(
                        nn.Linear(observation_space,500),
                        nn.Tanh(),
                        nn.Linear(500,1)
                    )
                def forward(self,input):
                    return self.actor(input),self.critic(input)
            self.mlp = MLP().to(self.device)
            
            
            
        ### Normalize the return and obs
        self.mlp.eval()
        with torch.no_grad():
                data = self.get_data_from_env()
        data = custom_dataset(data,self.data_size,self.number_of_envs,self.gamma)
        self.obs_var_mean = torch.var_mean(data.local_observation,dim=0)
        with torch.no_grad():
                data = self.get_data_from_env(normalizer=self.obs_var_mean)
        data = custom_dataset(data,self.data_size,self.number_of_envs,self.gamma)
        self.qua_var_mean = torch.var_mean(data.local_return,dim=0)



        # optim setup
        self.mlp_optimizer = torch.optim.Adam(self.mlp.parameters(),lr = self.learning_rate)
        if load_model:
            self.mlp.load_state_dict(torch.load(self.model_path,map_location=self.device))
            self.mlp_optimizer.load_state_dict(torch.load(self.optim_path,map_location=device))
        else:
            pass
        self.mlp_optimizer.param_groups[0]['lr'] = self.learning_rate
        print(self.mlp_optimizer.param_groups[0]['lr'])


    
    #helper functions
    
    def get_actor_critic_action_and_values(self,obs,eval=True):
        logits, values = self.mlp(obs)
        probs = TanhNormal(loc = logits[:,:self.action_space], scale = 1*nn.Sigmoid()(logits[:,self.action_space:]),max=np.pi/2,min=-np.pi/2)
        # probs = TanhNormal(loc = (torch.pi/2)*nn.Tanh()(logits[:,:self.action_space]),scale=0.5*nn.Sigmoid()(logits[:,self.action_space:]))
        if eval is True:
            action = probs.sample()
            return action, probs.log_prob(action)
        else:
            action = eval
            return action, probs.log_prob(action), -probs.log_prob(action).mean(dim=0), values

    def get_data_from_env(self,normalizer = (torch.tensor(1),torch.tensor(1))):
        ### THE FIRST EPS WILL BE TIMESTEP 1, THE FINAL EP WILL BE TIMESTEP 0
        local_observation = []
        local_action = []
        local_logprob = []
        local_reward = []
        local_timestep = []
        
        observation = self.env.get_obs()[0]
        observation = (observation-normalizer[1].numpy())/normalizer[0].numpy()**0.5
        
        local_observation.append(torch.Tensor(observation))
        timestep = np.ones((self.number_of_envs))
        local_timestep.append(torch.Tensor(timestep.copy()))
        for i in range(self.data_size) :
            
            # act and get observation 
            action, logprob = self.get_actor_critic_action_and_values(torch.Tensor(observation).to(self.device))
            action, logprob = action.cpu(), logprob.cpu()
            local_action.append(torch.Tensor(action))
            local_logprob.append(torch.Tensor(logprob))
            self.env.step(action)
            observation, reward, info= self.env.get_obs()
            observation = (observation-normalizer[1].numpy())/normalizer[0].numpy()**0.5
            reward = np.sum(reward*self.reward_index,axis=-1)
            terminated,truncated = False, info[0]

            # save var
            local_reward.append(torch.Tensor(reward))
            local_observation.append(torch.Tensor(observation))
            local_timestep.append(torch.Tensor(timestep.copy()))
            
            timestep = (1 + timestep)*(1-(terminated | truncated))
        return local_observation, local_action, local_logprob, local_reward, local_timestep

    def train(self):
        best_reward = 0
        for epoch in range(self.epochs):
            mlp = self.mlp.eval()
            # Sample data from the environment
            with torch.no_grad():
                data = self.get_data_from_env(self.obs_var_mean)
            dataset = custom_dataset(data,self.data_size,self.number_of_envs,self.gamma)
            dataloader = DataLoader(dataset,batch_size=self.batch_size,shuffle=True)
            for iteration, data in enumerate(dataloader):
                mlp = mlp.train()
                
                obs, action, logprob, quality, reward = data
                obs, action, logprob, quality, reward = obs.to(self.device), action.to(self.device), logprob.to(self.device), quality.to(self.device), reward.to(self.device)
                
                # Normalize return
                quality = (quality-self.qua_var_mean[1])/self.qua_var_mean[0]**.5
                
                next_action, next_logprob, entropy, value = self.get_actor_critic_action_and_values(obs,eval=action)
                # print(reward-quality)
                # Train models
                self.mlp_optimizer.zero_grad()
                prob_ratio = torch.exp(next_logprob-logprob)
                # print('qua',quality)
                # print('val',value)
                advantage = quality-value
                critic_loss = (advantage**2).mean()
                entropy_loss = entropy.mean()
                actor_loss = - torch.min( prob_ratio*advantage , torch.clamp(prob_ratio, 1-self.epsilon, 1+self.epsilon)*advantage ).mean() - self.explore*entropy_loss
                loss = critic_loss + actor_loss
                loss.backward()
                self.mlp_optimizer.step()
                
                #save model
                if self.save_model:
                    if (quality.mean().item()>best_reward and quality.mean().item() > self.thresh) | ((epoch*(len(dataloader))+iteration) % 1000 == 0):
                        best_reward = quality.mean().item()
                        torch.save(mlp.state_dict(), self.PATH+'models//PPO//'+t.strftime('%Y-%m-%d-%H-%M-%S', t.localtime())+'_best_'+str(round(quality.mean().item(),2)))
                        torch.save(self.mlp_optimizer.state_dict(), self.PATH+'models//PPO//'+t.strftime('%Y-%m-%d-%H-%M-%S', t.localtime())+'_best_'+str(round(quality.mean().item(),2))+'optim')
                        print('saved at: '+str(round(quality.mean().item(),2)))
                
                # logging info
                if self.log_data:
                    self.writer.add_scalar('Eval/minibatchreward',reward.mean().item(),epoch*(len(dataloader))+iteration)
                    self.writer.add_scalar('Eval/minibatchreturn',quality.mean().item(),epoch*(len(dataloader))+iteration)
                    self.writer.add_scalar('Train/entropyloss',entropy_loss.item(),epoch*(len(dataloader))+iteration)
                    self.writer.add_scalar('Train/criticloss',critic_loss.detach().mean().item(),epoch*(len(dataloader))+iteration)
                    self.writer.add_scalar('Train/actorloss',actor_loss.detach().mean().item(),epoch*(len(dataloader))+iteration)
                print(f'[{epoch}]:[{self.epochs}]|| iter [{epoch*(len(dataloader))+iteration}]: rew: {round(reward.mean().item(),2)} ret: {round(quality.mean().item(),2)} cri: {critic_loss.detach().mean().item()} act: {actor_loss.detach().mean().item()} entr: {entropy_loss.detach().item()}')
        torch.save(mlp.state_dict(), self.PATH+'models//PPO//'+t.strftime('%Y-%m-%d-%H-%M-%S', t.localtime()))
        torch.save(self.mlp_optimizer.state_dict(), self.PATH+'models//PPO//'+t.strftime('%Y-%m-%d-%H-%M-%S', t.localtime())+'optim')




class custom_dataset(Dataset):
    
    def __init__(self,data,data_size,number_of_envs,gamma):
        self.data_size = data_size
        self.number_of_envs = number_of_envs
        self.gamma = gamma
        self.obs, self.action, self.logprob, self.reward, self.timestep = data        
        self.local_return = [0 for i in range(data_size)]
        self.local_return = torch.hstack(self.get_G()).view(-1,1)
        self.local_observation = torch.vstack(self.obs)
        self.local_action = torch.vstack(self.action)
        self.local_logprob = torch.vstack(self.logprob).view(-1,1)
        self.local_reward = torch.hstack(self.reward).view(-1,1)
        # print(self.local_observation.shape)
        # print(self.local_action.shape)
        # print(self.local_logprob.shape)
        # print(self.local_return.shape)
        # print(self.local_reward.shape)

    def __len__(self):
        return self.data_size*self.number_of_envs
    
    def __getitem__(self, index):
        return self.local_observation[index], self.local_action[index], self.local_logprob[index], self.local_return[index], self.local_reward[index]
    
    def isnt_end(self, i):
        ### THE FIRST EPS WILL BE TIMESTEP 1, THE FINAL EP WILL BE TIMESTEP 0
        return self.timestep[i] != 0
    
    def get_G(self):
        ### THE FIRST EPS WILL BE TIMESTEP 1, THE FINAL EP WILL BE TIMESTEP 0
        for  i in range(self.data_size-1,-1,-1):
            if i == self.data_size-1:
                self.local_return[i] = self.reward[i]
            else:
                self.local_return[i] = self.reward[i] + self.isnt_end(i)*self.gamma*self.local_return[i+1]
        return self.local_return   
