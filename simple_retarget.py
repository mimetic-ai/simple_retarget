import torch
import kinpy as kp
from sklearn.metrics import mean_squared_error
# import math
from image_proc import *
from robot_analysis import RobotArm
import numpy as np
import matplotlib.pyplot as plt
import pinocchio as pin
import json


model = pin.buildModelFromUrdf('robot_description/panda.urdf')
pos_lim_lo = model.lowerPositionLimit
pos_lim_hi = model.upperPositionLimit
lb = torch.tensor(pos_lim_lo[0:7])
ub = torch.tensor(pos_lim_hi[0:7])

def totalLoss(arm_pos, robot_pos, joint_angles, lb, ub):
    true_elbow = (arm_pos['l_elbow'])/float(torch.norm(arm_pos['l_elbow']))
    true_wrist = (arm_pos['l_wrist'])/float(torch.norm(arm_pos['l_wrist']))
    pred_elbow = (robot_pos['panda_link3'])/float(torch.norm(robot_pos['panda_link3']))
    pred_wrist = (robot_pos['panda_hand'])/float(torch.norm(robot_pos['panda_hand']))
    elbow_loss = torch.norm(true_elbow - pred_elbow)
    wrist_loss = torch.norm(true_wrist - pred_wrist)
    # print("elbow rquires grad ", elbow_loss.grad)
    # print("wrist rquires grad ", wrist_loss.grad)
    joint_loss = torch.div(torch.tensor([0.00000001]), torch.pow(torch.dot((joint_angles - ub), (joint_angles - lb)), 2))
    joint_loss = 1000 * joint_loss
    return elbow_loss + wrist_loss + joint_loss

def newLoss(arm_pos, robot_pos):
    true_elbow = arm_pos['l_elbow'].detach().numpy()
    true_wrist = arm_pos['l_wrist'].detach().numpy()
    pred_elbow = robot_pos['panda_link5'].detach().numpy()
    pred_wrist = robot_pos['panda_hand'].detach().numpy()
    true_elbow = np.asarray([true_elbow[0], true_elbow[2], true_elbow[1]])/np.linalg.norm(true_elbow)
    true_wrist = np.asarray([true_wrist[0], true_wrist[2], true_wrist[1]])/np.linalg.norm(true_wrist)
    pred_elbow = np.asarray([pred_elbow[0], pred_elbow[2], pred_elbow[1]])/np.linalg.norm(pred_elbow)
    pred_wrist = np.asarray([pred_wrist[0], pred_wrist[2], pred_wrist[1]])/np.linalg.norm(pred_wrist)
    elbow_loss = mean_squared_error(true_elbow, pred_elbow)
    wrist_loss = mean_squared_error(true_wrist, pred_wrist)
    return (elbow_loss + wrist_loss)/2


def forward(joint_angles, robot):
    robot.setJointAngles(joint_angles)
    new_keypoints = robot.getKeyPointPoses()
    return new_keypoints

path = 'AMCParser/boxing_arm_data.json'
f = open(path)
data = json.load(f)
arm_pos = {'l_shoulder': None, 'l_elbow': None, 'l_wrist': None}
arm_pos['l_shoulder'] = torch.tensor(data['lclavicle'][0])
arm_pos['l_elbow'] = torch.tensor(data['lhumerus'][0])
arm_pos['l_wrist'] = torch.tensor(data['lhand'][0])

panda_robot = RobotArm('robot_description/panda.urdf')
end_effector_pose = getEndEffectorPose(arm_pos)

step_size = 0.1
loss_BGD = []
n_iter = 2
i = 0
eps = 0.1
loss = float('inf')



#####initial guess
chain = kp.build_serial_chain_from_urdf(open('robot_description/panda.urdf').read(), end_link_name = 'panda_hand')
end_effector_transform = kp.transform.Transform(rot = end_effector_pose[1], pos = end_effector_pose[0])
#end_effector_transform = kp.transform.Transform(rot = None, pos = end_effector_pose[0])
ik_results = chain.inverse_kinematics(end_effector_transform)
joint_angles = torch.tensor(ik_results, requires_grad=True, dtype=torch.float)
#joint_angles = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True, dtype=torch.float)


norm_var = 1000
print(norm_var)
print(eps)
while (i <= 500):
    # making predictions with forward pass
    robot_pos = forward(joint_angles=joint_angles, robot=panda_robot)
    # calculating the loss between original and predicted data points
    loss = totalLoss(arm_pos=arm_pos, robot_pos=robot_pos, joint_angles=joint_angles, lb=lb, ub=ub)
    print("loss ", loss)
    # storing the calculated loss in a list
    loss_BGD.append(loss.item())
    # backward pass for computing the gradients of the loss w.r.t to learnable parameters
    joint_angles.retain_grad()
    loss.backward()
    # updateing the parameters after each iteration
    print("joint angles ", joint_angles)
    norm_var = torch.norm(joint_angles.grad)
    joint_angles = joint_angles - step_size * joint_angles.grad
    i += 1
    # we don't need this because joint angles is populated with 
    #   a vector that doesn't require grad 
    # print("grad later", joint_angles.grad.zero_())

    # priting the values for understanding
    # print('{}, \t{}, \t{}, \t{}'.format(i, loss.item(), w.item(), b.item()))


print(joint_angles)
degrees_list = np.rad2deg(joint_angles.detach().numpy())
print(degrees_list)
print("reached end of file")
# Plot the loss
plt.plot(loss_BGD)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()