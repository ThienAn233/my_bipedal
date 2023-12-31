import torch
import torch.nn as nn
import numpy as np
import time as t
import my_bipedal.bipedal_walker_env_v1 as bpd
from torch.utils.data import Dataset, DataLoader
from torchrl.modules import TanhNormal
from torch.utils.tensorboard import SummaryWriter
class PPO_bipedal_walker_train():
    def __init__(self,
                # Global variables
                PATH = None,
                load_model = None,
                envi = bpd,
                log_data = True,
                save_model = True,
                render_mode = False,
                thresh = 0.65,

                epsilon = 0.2,
                explore = 1e-4,
                gamma = .99,
                learning_rate = 1e-4,
                number_of_robot = 9,
                epochs = 500,
                data_size = 1000,
                batch_size = 2000,
                reward_index = np.array([[1, 1, 1, 1, 1]]),
                seed = 1107,
                mlp = None,

                # local variables
                # Seed & devices
                action_space = 6,
                observation_space = 24,
                device = None):

                    
                    
        # Global variables
        self.PATH = PATH
        self.load_model = load_model
        self.envi = envi
        self.log_data = log_data
        self.save_model = save_model
        self.render_mode = render_mode
        self.thresh = thresh
        self.epsilon = epsilon
        self.explore = explore
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.number_of_robot = number_of_robot
        self.epochs = epochs
        self.data_size = data_size
        self.batch_size = batch_size
        self.reward_index = reward_index
        self.seed = seed
        self.mlp = mlp


                    
        # local variables
        # Seed & devices
        self.action_space = action_space
        self.observation_space = observation_space + action_space
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
        
        self.env = envi.bipedal_walker(num_robot=self.number_of_robot,render_mode=self.render_mode)
        print('env is ready!')
        print(f'action space of {number_of_robot} robot is: {action_space}')
        print(f'observation sapce of {number_of_robot} robot is: {observation_space}')
        
        
        
        # Setup MLP
        if self.mlp:
            self.mlp.to(self.device)
            pass
        else:
            class MLP(nn.Module):
                def __init__(self):
                    super(MLP,self).__init__()
                # nn setup
                    lin1 = nn.Linear(observation_space,500)
                    torch.nn.init.xavier_normal_(lin1.weight)
                    lin2 = nn.Linear(500,action_space*2)
                    torch.nn.init.xavier_normal_(lin2.weight)
                    self.actor = nn.Sequential(
                        lin1,
                        nn.Tanh(),
                        lin2,
                    )
                    lin3 = nn.Linear(observation_space,500)
                    torch.nn.init.xavier_normal_(lin3.weight)
                    lin4 = nn.Linear(500,1)
                    torch.nn.init.xavier_normal_(lin4.weight)
                    self.critic = nn.Sequential(
                        lin3,
                        nn.Tanh(),
                        lin4,
                    )
                def forward(self,input):
                    return self.actor(input),self.critic(input)
            self.mlp = MLP().to(self.device)
        print('MLP is ready!')
            
        ### Normalize the return and obs
        self.mlp.eval()
        with torch.no_grad():
                data = self.get_data_from_env()
        data = custom_dataset(data,self.data_size,self.number_of_robot,self.gamma)
        self.qua_var_mean = torch.var_mean(data.local_return,dim=0)
        self.val_var_mean = torch.var_mean(data.local_values,dim=0)
        print(f'quality var mean: {self.qua_var_mean}')
        print(f'values var mean: {self.val_var_mean}')




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
        logits = logits.view(*logits.shape,1)
        probs = TanhNormal(loc = logits[:,:self.action_space], scale=0.2*nn.Sigmoid()(logits[:,self.action_space:]),max=np.pi/4,min=-np.pi/4)
        # probs = Normal(loc = logits[:,:self.action_space],scale=nn.Sigmoid()(logits[:,self.action_space:]))
        if eval is True:
            action = probs.sample()
            return action, probs.log_prob(action), values
        else:
            action = eval
            dummy_action = probs.sample((100,))
            return action, probs.log_prob(action), -probs.log_prob(dummy_action).mean(dim=0), values

    def get_data_from_env(self):
        ### THE FIRST EPS WILL BE TIMESTEP 1, THE FINAL EP WILL BE TIMESTEP 0
        local_observation = []
        local_action = []
        local_logprob = []
        local_reward = []
        local_timestep = []
        local_values = []
        
        observation = self.env.get_obs()[0]
        # previous_action = np.zeros((self.number_of_robot,self.action_space))
        # observation = np.hstack([observation,previous_action])
        
        timestep = np.ones((self.number_of_robot))
        local_timestep.append(torch.Tensor(timestep.copy()))
        for i in range(self.data_size) :
            # act and get observation 
            action, logprob, values = self.get_actor_critic_action_and_values(torch.Tensor(observation).to(self.device))
            action, logprob = action.cpu(), logprob.cpu()
            local_observation.append(torch.Tensor(observation))
            local_action.append(torch.Tensor(action))
            local_logprob.append(torch.Tensor(logprob))
            self.env.act(action)
            self.env.sim()
            observation, reward, info= self.env.get_obs()
            
            # stacking obs and previous action
            # previous_action = action.squeeze()
            # observation = np.hstack([observation,previous_action])

            reward = np.sum(reward*self.reward_index,axis=-1)
            truncated = info[0]

            # save var
            local_reward.append(torch.Tensor(reward))
            local_observation.append(torch.Tensor(observation))
            local_timestep.append(torch.Tensor(timestep.copy()))
            local_values.append(torch.Tensor(values))
            
            timestep = (1 + timestep)*(1-truncated)
        return local_observation, local_action, local_logprob, local_reward, local_timestep, local_values

    def train(self):
        best_reward = 0
        for epoch in range(self.epochs):
            mlp = self.mlp.eval()
            # Sample data from the environment
            with torch.no_grad():
                data = self.get_data_from_env()
            dataset = custom_dataset(data,self.data_size,self.number_of_robot,self.gamma)
            dataloader = DataLoader(dataset,batch_size=self.batch_size,shuffle=True)
            for iteration, data in enumerate(dataloader):
                mlp = mlp.train()
                
                obs, action, logprob, quality, reward = data
                # Normalize return
                quality = (quality-self.qua_var_mean[1])/self.qua_var_mean[0]**.5
                obs, action, logprob, quality, reward = obs.to(self.device), action.to(self.device), logprob.to(self.device), quality.to(self.device), reward.to(self.device)
                next_action, next_logprob, entropy, value = self.get_actor_critic_action_and_values(obs,eval=action)
                # Normalize values
                value = (value-self.val_var_mean[1])/self.val_var_mean[0]**.5
                # Train models
                self.mlp_optimizer.zero_grad()
                prob_ratio = torch.exp(next_logprob-logprob)
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
    
    def __init__(self,data,data_size,number_of_robot,gamma):
        self.data_size = data_size
        self.number_of_robot = number_of_robot
        self.gamma = gamma
        self.obs, self.action, self.logprob, self.reward, self.timestep, self.values = data        
        self.local_return = [0 for i in range(data_size)]
        self.local_return = torch.hstack(self.get_G()).view(-1,1)
        self.local_observation = torch.vstack(self.obs)
        self.local_action = torch.vstack(self.action)
        self.local_logprob = torch.vstack(self.logprob).view(-1,1)
        self.local_reward = torch.hstack(self.reward).view(-1,1)
        self.local_values = torch.hstack(self.values).view(-1,1)
        # print(self.local_observation.shape)
        # print(self.local_action.shape)
        # print(self.local_logprob.shape)
        # print(self.local_return.shape)
        # print(self.local_reward.shape)

    def __len__(self):
        return self.data_size*self.number_of_robot
    
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
