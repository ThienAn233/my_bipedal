import pybullet as p 
import pybullet_data
import numpy as np
from numpy.linalg import norm
import time as t 



class bipedal_walker():
    
    def __init__(self,
                 max_length = 1000,
                 num_step = 10,
                 render_mode = None,
                 robot_file = None,
                 seed = 0):
        
        # Configure-able variables
        self.num_step = num_step
        self.max_length = max_length//num_step
        self.render_mode = render_mode
        self.robot_file = robot_file
        if render_mode:
            self.physicsClient = p.connect(p.GUI)
            self.sleep_time = 1./240.
        else:
            self.physicsClient = p.connect(p.DIRECT)
        if robot_file:
            self.robot_file = robot_file
        else:
            self.robot_file = 'my_bipedal//bipedal.urdf'
        self.target_file = 'my_bipedal//target.urdf'
        self.target_radius = [0,2]
        self.target_height = [0.2,0.5]
        self.target = None
        self.thresh = 0.1
        self.initialPos = None
        self.initialHeight = 0.6
        self.jointId_list = []
        self.jointName_list = []
        self.jointRange_list = []
        self.jointMaxForce_list = []
        self.jointMaxVeloc_list = []
        self.mode = p.POSITION_CONTROL
        self.seed = seed
        np.random.seed(self.seed)
        
        # Constants (DO NOT TOUCH)
        self.g = (0,0,-9.81) 
        self.pi = np.pi
        self.total_episode = 0
        self.time_steps_in_current_episode = 0
        self.vertical = np.array([0,0,1])
        
        # Settup the environment and print out some variables
        # print('-----------------------------------')
        # print(f'ENVIRONMENT STARTED WITH SEED {self.seed}')
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId = self.physicsClient)
        
        # Load URDF and print info
        p.setGravity(*self.g, physicsClientId = self.physicsClient)
        self.robotId = p.loadURDF(self.robot_file, physicsClientId = self.physicsClient)
        self.targetId = p.loadURDF(self.target_file, physicsClientId = self.physicsClient)
        self.planeId = p.loadURDF('plane.urdf', physicsClientId = self.physicsClient)
        self.number_of_joints = p.getNumJoints(self.robotId, physicsClientId = self.physicsClient)
        # print(f'Robot id: {self.robotId}')
        # print(f'number of robot joints: {self.number_of_joints}')
        for jointIndex in range(0,self.number_of_joints):
            data = p.getJointInfo(self.robotId, jointIndex, physicsClientId = self.physicsClient)
            self.jointId_list.append(data[0])                                                                                # Create list to store joint's Id
            self.jointName_list.append(str(data[1]))                                                                         # Create list to store joint's Name
            self.jointRange_list.append((data[8],data[9]))                                                                   # Create list to store joint's Range
            self.jointMaxForce_list.append(data[10])                                                                         # Create list to store joint's max Force
            self.jointMaxVeloc_list.append(data[11])                                                                         # Create list to store joint's max Velocity
            # print(f'Id: {data[0]}, Name: {str(data[1])}, Range: {(data[8],data[9])}')
        p.setJointMotorControlArray(self.robotId,self.jointId_list,self.mode, physicsClientId = self.physicsClient)
        # print(f'Control mode is set to: {"Velocity" if self.mode==0 else "Position"}')
        
        # Sample and calculate a random point as target and inital position
        self.sample_target(origin=True)
        self.sample_target()
        self.target_maintainer()
        
        # print('-----------------------------------')
        
        
        
    def get_obs(self):
        
        self.time_steps_in_current_episode += 1
        temp_obs_value = None
        temp_info = None
        temp_reward_value = []
        
        for _ in range(self.num_step):
            p.stepSimulation( physicsClientId = self.physicsClient)
        
        # GET OBSERVATION
        temp_obs_value = self.get_all_obs()

        # GET INFO
        # Check weather the target is reached, if no, pass, else sammple new target
        goal, temp_info = self.auto_reset(temp_obs_value[0],temp_obs_value[4],temp_obs_value[5:8])
        
        # GET REWARD
        temp_reward_value = self.get_reward_value(temp_obs_value,goal)
        
        # MAINTAIN TARGET
        self.target_maintainer()

        return temp_obs_value, temp_reward_value, temp_info
    
    
    
    def step(self,action):
        p.setJointMotorControlArray(self.robotId,self.jointId_list,self.mode,targetPositions = action, forces = self.jointMaxForce_list, targetVelocities = self.jointMaxVeloc_list, physicsClientId = self.physicsClient)
    
    def close(self):
        p.disconnect(physicsClientId = self.physicsClient)

    
    def sample_target(self,origin=False):
        random_radius = np.random.uniform(*self.target_radius)
        random_heights = np.random.uniform(*self.target_height)
        random_angle = np.random.uniform(0,2*self.pi)
        random_Ori = p.getQuaternionFromEuler([0,0,np.random.uniform(0,2*self.pi)], physicsClientId = self.physicsClient)
        if origin:      # if origin: sample and reset the origin position
            self.initialPos = np.array([np.sin(random_angle)*random_radius, np.cos(random_angle)*random_radius, self.initialHeight])
            p.resetBasePositionAndOrientation(self.robotId, self.initialPos, random_Ori, physicsClientId = self.physicsClient)
        else:           # else: sample new target position
            self.target = np.array([np.sin(random_angle)*random_radius, np.cos(random_angle)*random_radius, random_heights])

    def target_maintainer(self):
        # p.resetBasePositionAndOrientation(self.targetId, self.target, [0,0,0,1], physicsClientId = self.physicsClient)
        return
        
    def get_distance_and_ori_and_velocity_from_target(self):
        temp_obs_value = []
        
        # Get target cordinate in robot reference and distance
        base_position, base_orientation =  p.getBasePositionAndOrientation(self.robotId, physicsClientId = self.physicsClient)
        temp_obs_value += [np.sum((self.target - base_position)**2)**.5, *(self.target - base_position)]
        
        # Get base height and base orientation in quaternion
        temp_obs_value += [base_position[-1],*base_orientation]
        
        # Get base linear and angular velocity
        linear_velo, angular_velo = p.getBaseVelocity(self.robotId, physicsClientId = self.physicsClient)
        temp_obs_value += [*linear_velo, *angular_velo]
        
        return temp_obs_value
    
    def get_joints_values(self):
        temp_obs_value = []
        
        # Get joints position and velocity
        for Id in self.jointId_list:
            temp_obs_value += [*p.getJointState(self.robotId,Id, physicsClientId = self.physicsClient)[:2]]
            
        return temp_obs_value
    
    def get_links_values(self):
        temp_obs_vaule = []

        #get links orientation of links in quaternion
        for Id in self.jointId_list:
            temp_obs_vaule += [*p.getLinkState(self.robotId,Id,physicsClientId = self.physicsClient)[1]]
        return temp_obs_vaule
    
    def get_all_obs(self):
        temp_obs_value = []
        
        # Base position state
        base_info = self.get_distance_and_ori_and_velocity_from_target()
        
        # Joints state
        joints_info = self.get_joints_values()
        
        # Links state
        links_info = self.get_links_values()
        
        # Full observation
        temp_obs_value = [*base_info,*joints_info,*links_info]
        
        return temp_obs_value
        
    def terminate_check(self, distance):
        return (distance <= self.thresh) 
    
    def truncation_check(self,height,vec):
        vec = np.array(vec)
        cosin = np.dot(vec,self.vertical)/(norm(vec))
        return (self.time_steps_in_current_episode >= self.max_length) | (self.target_height[0] > height) | (cosin < 0.93)
    
    def auto_reset(self,distance,height,vec):
        termination = self.terminate_check(distance)
        truncation = self.truncation_check(height,vec)
        goal = 0
        if termination:
            goal = 1
            self.sample_target()
            self.total_episode += 1
        if truncation:
            self.sample_target()
            self.sample_target(origin=True)
            self.total_episode +=1
            self.time_steps_in_current_episode = 0
        return goal, [truncation]
    
    def get_reward_value(self,obs,goal=0):
        
        # Reward for reaching the goal
        reach = np.exp(-0.25*obs[0]**2)
        
        # Reward for being high
        high = np.exp(-2*(obs[4]-0.4)**2)
        
        # Reward for good base orientation
        vec = np.array(obs[5:8])
        cosin = np.dot(vec,self.vertical)/(norm(vec))
        ori = np.exp(-(cosin-1)**2)
        
        # # Survival reward: embeded in reward for being high
        # sur = 1
        
        return [reach, high, ori]
        
class SyncVectorEnv():
    def __init__(self,env,num_of_env = 5,render_mode = None):
        self.env = env
        self.num_of_env = num_of_env
        if render_mode:
            self.env_list = [env(render_mode = render_mode)]
        else:
            self.env_list = [env()]
        self.env_list += [env() for i in range(num_of_env-1)]
    
    def get_obs(self):
        obs_list = []
        rew_list = []
        inf_list = []
        for env in self.env_list:
            obs, rew, inf = env.get_obs()
            obs_list.append(obs)
            rew_list.append(rew)
            inf_list.append(inf)
        return np.array(obs_list), np.array(rew_list), np.array(inf_list)
    
    def step(self, actions):
        for i in range(self.num_of_env):
            self.env_list[i].step(actions[i,:])
    
    def close(self):
        for env in self.env_list:
            env.close()
        
# TEST ###
# env = bipedal_walker(render_mode='human')
# for _ in range(100):
#     # env.step()
#     obs,rew,_ = env.get_obs()
#     print(len(obs))
#     t.sleep(1./240.)
# env.close()
# env = SyncVectorEnv(bipedal_walker)
# for _ in range(1500):
#     # env.step()
#     obs = env.get_obs()
#     print(obs)
# env.close()
