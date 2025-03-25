import pybullet as p
import pybullet_data
import numpy as np
import time as t

# Variables
PATH = 'my_bipedal//biped.urdf'
sleep_time = 1./240.
initial_height = 1 #0.17
initial_ori = [0,0,0,1]
jointId_list = []
jointName_list = []
jointRange_list = []
jointMaxForce_list = []
jointMaxVeloc_list = []
debugId_list = []
temp_debug_value = []
mode = p.POSITION_CONTROL
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Constants
g = (0,0,-.981) 
pi = np.pi

# Setup the environment
print('-'*100)
p.setGravity(*g)
robotId = p.loadURDF(PATH,[0.,0.,initial_height],initial_ori)
planeId = p.loadURDF('plane.urdf')
number_of_joints = p.getNumJoints(robotId)
print(f'Robot id: {robotId}')
print(f'number of robot joints: {number_of_joints}')
for joint_index in range(number_of_joints):
    data = p.getJointInfo(robotId, joint_index)
    jointId_list.append(data[0])                                                                                # Create list to store joint's Id
    jointName_list.append(str(data[1]))                                                                         # Create list to store joint's Name
    jointRange_list.append((data[8],data[9]))                                                                   # Create list to store joint's Range
    jointMaxForce_list.append(data[10])                                                                         # Create list to store joint's max Force
    jointMaxVeloc_list.append(data[11])                                                                         # Create list to store joint's max Velocity
    debugId_list.append(p.addUserDebugParameter(str(data[1]), rangeMin = data[8], rangeMax = data[9], ))        # Add debug parameters to manually control joints
    p.enableJointForceTorqueSensor(robotId,joint_index,True)
    print(f'Id: {data[0]}, Name: {str(data[1])}, Range: {(data[8],data[9])}, DebugId: {debugId_list[-1]}')
p.setJointMotorControlArray(robotId,jointId_list,mode)
print(f'Control mode is set to: {"Velocity" if mode==0 else "Position"}')
previous_pos = np.zeros((len(jointId_list)))
print('-'*100)

# Simulation loop
while True:
    p.stepSimulation()
    temp_debug_value = []
    for Id in debugId_list:
        temp_debug_value.append(p.readUserDebugParameter(Id))
    filtered_action = previous_pos*.8 + np.array(temp_debug_value)*.2
    p.setJointMotorControlArray(robotId,
                                jointId_list,
                                mode,
                                targetPositions = filtered_action,
                                forces = jointMaxForce_list, 
                                targetVelocities = jointMaxVeloc_list,
                                positionGains = np.ones_like(temp_debug_value)*.5,
                                # velocityGains = np.ones_like(temp_debug_value)*0.,        
                                )
    p.resetBasePositionAndOrientation(robotId,[0.,0.,initial_height],initial_ori)
    # base_inf =  p.getBasePositionAndOrientation(robotId)
    # print(f'robot height: {base_inf[0][-1]}')
    # previous_pos = np.array(temp_debug_value)
    t.sleep(sleep_time)