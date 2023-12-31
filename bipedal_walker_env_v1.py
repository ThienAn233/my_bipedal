import pybullet as p 
import pybullet_data
import numpy as np
from numpy.linalg import norm 
import time as t 



class bipedal_walker():
    
    def __init__(self,
                 max_length = 100,
                 num_step = 50,
                 render_mode = None,
                 robot_file = None,
                 num_robot = 9,
                 seed = 0):
        
        # Configure-able variables
        self.num_step = num_step
        self.max_length = max_length
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
        self.num_robot = num_robot
        self.target_height = [0.4,0.6]
        self.target = None
        self.initialPos = None
        self.initialHeight = 0.48
        self.robotId_list = []
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
        self.time_steps_in_current_episode = [0 for _ in range(self.num_robot)]
        self.vertical = np.array([0,0,1])
        
        # Settup the environment and print out some variables
        print('-----------------------------------')
        # print(f'ENVIRONMENT STARTED WITH SEED {self.seed}')
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId = self.physicsClient)
        
        # Load URDF and print info
        p.setGravity(*self.g, physicsClientId = self.physicsClient)
        self.get_init_pos()
        for pos in self.corr_list:
            self.robotId_list.append(p.loadURDF(self.robot_file, physicsClientId = self.physicsClient,basePosition=pos,baseOrientation=[0,0,1,0]))
            print(f'robot with Id: {self.robotId_list[-1]} is loaded')
        # self.targetId = p.loadURDF(self.target_file, physicsClientId = self.physicsClient)
        self.planeId = p.loadURDF('plane.urdf', physicsClientId = self.physicsClient)
        self.number_of_joints = p.getNumJoints(self.robotId_list[0], physicsClientId = self.physicsClient)

        for jointIndex in range(0,self.number_of_joints):
            data = p.getJointInfo(self.robotId_list[0], jointIndex, physicsClientId = self.physicsClient)
            self.jointId_list.append(data[0])                                                                                # Create list to store joint's Id
            self.jointName_list.append(str(data[1]))                                                                         # Create list to store joint's Name
            self.jointRange_list.append((data[8],data[9]))                                                                   # Create list to store joint's Range
            self.jointMaxForce_list.append(data[10])                                                                         # Create list to store joint's max Force
            self.jointMaxVeloc_list.append(data[11])                                                                         # Create list to store joint's max Velocity
            # print(f'Id: {data[0]}, Name: {str(data[1])}, Range: {(data[8],data[9])}')
        for robotId in self.robotId_list:
            p.setJointMotorControlArray(robotId,self.jointId_list,self.mode, physicsClientId = self.physicsClient)
            self.sample_target(robotId)
        print('-----------------------------------')
        
    def get_init_pos(self):
        if self.num_robot/np.sqrt(self.num_robot)==int(np.sqrt(self.num_robot)):
            pass
        else:
            print('num_robot must be a prime')
        nrow = int(self.num_robot)
        x = np.linspace(-(nrow+1)/2,(nrow+1)/2,nrow)
        xv,yv = np.meshgrid(0,x)
        xv, yv = np.hstack(xv), np.hstack(yv)
        zv = self.initialHeight*np.ones_like(xv)
        self.corr_list = np.vstack((xv,yv,zv)).transpose()
        
    def sim(self,real_time = False):
        self.time_steps_in_current_episode = [self.time_steps_in_current_episode[i]+1 for i in range(self.num_robot)]
        for _ in range(self.num_step):
            p.stepSimulation( physicsClientId = self.physicsClient)
            if real_time:
                t.sleep(1./240.)
            
    def get_obs(self):
        
        temp_obs_value = []
        temp_info = []
        temp_reward_value = []

        for robotId in self.robotId_list:
            # GET OBSERVATION
            temp_obs_value += [self.get_all_obs(robotId)]

            # GET INFO
            # Check weather the target is reached, if no, pass, else sammple new target
            temp_info += [self.auto_reset(robotId,temp_obs_value[-1])]
            
            # GET REWARD
            temp_reward_value += [self.get_reward_value(temp_obs_value[-1],robotId)]
        

        return np.array(temp_obs_value), np.array(temp_reward_value), np.array(temp_info)
    
    
    def act(self,action):
        for robotId in self.robotId_list:
            p.setJointMotorControlArray(robotId,self.jointId_list,self.mode,targetPositions = action[robotId], forces = self.jointMaxForce_list, targetVelocities = self.jointMaxVeloc_list, physicsClientId = self.physicsClient)
    
    def close(self):
        p.disconnect(physicsClientId = self.physicsClient)
    
    def sample_target(self,robotId):
        random_Ori = [0,0,1,0]
        pos = self.corr_list[robotId]
        p.resetBasePositionAndOrientation(robotId, pos, random_Ori, physicsClientId = self.physicsClient)
        init_vel = [0,0,0]
        # np.random.uniform(-.3,.3,(3))
        p.resetBaseVelocity(robotId,init_vel,[0,0,0],physicsClientId=self.physicsClient)
        for jointId in self.jointId_list:
            p.resetJointState(bodyUniqueId=robotId,jointIndex=jointId,targetValue=0,targetVelocity=0,physicsClientId=self.physicsClient)
        
        
    def get_distance_and_ori_and_velocity_from_target(self,robotId):
        temp_obs_value = []
        
        # Get cordinate in robot reference 
        base_position, base_orientation =  p.getBasePositionAndOrientation(robotId, physicsClientId = self.physicsClient)
        base_position = [-base_position[i]+self.corr_list[robotId][i] for i in range(1,2)] + [base_position[-1]]
        temp_obs_value += [ *base_position]
        
        # Get  base orientation in quaternion
        temp_obs_value += [*base_orientation]
        
        # Get base linear and angular velocity
        linear_velo, angular_velo = p.getBaseVelocity(robotId, physicsClientId = self.physicsClient)
        temp_obs_value += [*linear_velo, *angular_velo]
        
        return temp_obs_value
    
    def get_joints_values(self,robotId):
        temp_obs_value = []
        
        # Get joints position and velocity
        for Id in self.jointId_list:
            temp_obs_value += [*p.getJointState(robotId,Id, physicsClientId = self.physicsClient)[:2]]
            
        return temp_obs_value
    
    def get_links_values(self,robotId):
        temp_obs_vaule = []

        #get links orientation of links in quaternion
        for Id in self.jointId_list:
            temp_obs_vaule += [*p.getLinkState(robotId,Id,physicsClientId = self.physicsClient)[1]]
        return temp_obs_vaule
    
    def get_all_obs(self,robotId):
        temp_obs_value = []
        
        # Base position state
        base_info = self.get_distance_and_ori_and_velocity_from_target(robotId)
        
        # Joints state
        joints_info = self.get_joints_values(robotId)

        # Links state
        # links_info = self.get_links_values(robotId)
        
        # Full observation
        temp_obs_value += [
                        *base_info,
                        *joints_info,
                        #*links_info
                        ]
        return temp_obs_value
    
    def truncation_check(self,height,vec,dir):
        vec = np.array(vec)
        cosin = np.dot(vec,self.vertical)/(norm(vec))
        return  (self.target_height[0] > height) #| (cosin < 0.95) | (np.abs(dir)>0.5)
    
    def auto_reset(self,robotId,obs):
        trunc_list = []
        height, vec, dir = obs[1], obs[2:5], obs[0]
        truncation = self.truncation_check(height,vec,dir)
        if truncation:
            self.sample_target(robotId)
            self.time_steps_in_current_episode[robotId] = 0
        trunc_list += [truncation]
        return trunc_list
    
    def get_reward_value(self,obs,robotId):
        # Reward for high speed in x direction
        speed = -10*obs[6]

        # Reward for being in good y direction
        align = -obs[0]**2
        
        # Reward for being high
        high = -(obs[1]-.5)**2
        
        # Reward for surviving 
        surv = 5
        
        # Reward for minimal force
        force = []
        for jointId in self.jointId_list:
            force.append(p.getJointState(robotId,jointId)[-1])
        force = (-1e-8)*((np.array(force)**2).sum())
        
        return [speed, align, high, surv, force ]
        
# TEST ###
# env = bipedal_walker(render_mode='human')
# for _ in range(1200):
#     env.sim()
#     obs,rew,inf = env.get_obs()
# env.close()
