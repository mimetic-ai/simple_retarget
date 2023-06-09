import torch
import kinpy as kp
from sklearn.metrics import mean_squared_error
# import math
from image_proc import *
from robot_analysis import RobotArm
import numpy as np
import matplotlib.pyplot as plt

def totalLoss(arm_pos, robot_pos):
    true_elbow = torch.norm(arm_pos['l_elbow'])
    true_wrist = torch.norm(arm_pos['l_wrist'])
    pred_elbow = torch.norm(robot_pos['panda_link5'])
    pred_wrist = torch.norm(robot_pos['panda_hand'])
    elbow_loss = torch.norm(true_elbow - pred_elbow)
    wrist_loss = torch.norm(true_wrist - pred_wrist)
    # print("elbow rquires grad ", elbow_loss.grad)
    # print("wrist rquires grad ", wrist_loss.grad)
    return elbow_loss + wrist_loss

def newLoss(arm_pos, robot_pos):
    true_elbow = arm_pos['l_elbow'].detach().numpy()
    true_wrist = arm_pos['l_wrist'].detach().numpy()
    pred_elbow = robot_pos['panda_link5'].detach().numpy()
    pred_wrist = robot_pos['panda_hand'].detach().numpy()
    true_elbow = true_elbow/np.linalg.norm(true_elbow)
    true_wrist = true_wrist/np.linalg.norm(true_wrist)
    pred_elbow = pred_elbow/np.linalg.norm(pred_elbow)
    pred_wrist = pred_elbow/np.linalg.norm(pred_wrist)
    elbow_loss = mean_squared_error(true_elbow, pred_elbow)
    wrist_loss = mean_squared_error(true_wrist, pred_wrist)
    return (elbow_loss + wrist_loss)/2


def forward(joint_angles, robot):
    robot.setJointAngles(joint_angles)
    new_keypoints = robot.getKeyPointPoses()
    return new_keypoints

arm_pos = wrtShoulder(getArmPosesFrame('media/sample_retarget_pose.jpg'))
panda_robot = RobotArm('robot_description/panda.urdf')
end_effector_pose = getEndEffectorPose(arm_pos)

step_size = 0.1
loss_BGD = []
n_iter = 2
i = 0
eps = 1e-3
loss = float('inf')



#####initial guess
chain = kp.build_serial_chain_from_urdf(open('robot_description/panda.urdf').read(), end_link_name = 'panda_hand')
end_effector_transform = kp.transform.Transform(rot = end_effector_pose[1], pos = end_effector_pose[0])
#end_effector_transform = kp.transform.Transform(rot = None, pos = end_effector_pose[0])
ik_results = chain.inverse_kinematics(end_effector_transform)
joint_angles = torch.tensor(ik_results, requires_grad=True, dtype=torch.float)

# while loss > eps and i < n_iter:
#     # making predictions with forward pass
#     robot_pos = forward(joint_angles=joint_angles, robot=panda_robot)
#     # calculating the loss between original and predicted data points
#     loss = newLoss(arm_pos=arm_pos, robot_pos=robot_pos)
#     print("loss ", loss)
#     # # storing the calculated loss in a list
#     # loss_BGD.append(loss.item())
#     # # backward pass for computing the gradients of the loss w.r.t to learnable parameters
#     # joint_angles.retain_grad()
#     # loss.backward()
#     # # updateing the parameters after each iteration
#     # print("joint angles ", joint_angles)
#     # joint_angles = joint_angles - step_size * joint_angles.grad
#     # i = i + 1

#     # # we don't need this because joint angles is populated with 
#     # #   a vector that doesn't require grad 
#     # # print("grad later", joint_angles.grad.zero_())

#     # # priting the values for understanding
#     # # print('{}, \t{}, \t{}, \t{}'.format(i, loss.item(), w.item(), b.item()))


# print("reached end of file")