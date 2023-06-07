import torch
from sklearn.metrics import mean_squared_error
import math
from image_proc import *
from robot_analysis import RobotArm
import numpy as np
import matplotlib.pyplot as plt

def totalLoss(arm_pos, robot_pos):
    true_elbow = arm_pos['l_elbow']
    true_wrist = arm_pos['l_wrist']
    pred_elbow = robot_pos['panda_link5']
    pred_wrist = robot_pos['panda_hand']
    elbow_loss = torch.norm(true_elbow - pred_elbow)
    wrist_loss = torch.norm(true_wrist - pred_wrist)
    print("elbow rquires grad ", elbow_loss.grad)
    print("wrist rquires grad ", wrist_loss.grad)
    return elbow_loss + wrist_loss



def forward(joint_angles, robot):
    robot.setJointAngles(joint_angles)
    new_keypoints = robot.getKeyPointPoses()
    print("new keypoints returned ")
    print(new_keypoints)
    return new_keypoints

arm_pos = wrtShoulder(getArmPosesFrame('media/sample_retarget_pose.jpg'))
panda_robot = RobotArm('robot_description/panda.urdf')

step_size = 0.1
loss_BGD = []
n_iter = 2

joint_angles = torch.tensor([0,0, 0,0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)


 
for i in range (n_iter):
    # making predictions with forward pass
    robot_pos = forward(joint_angles=joint_angles, robot=panda_robot)
    # calculating the loss between original and predicted data points
    loss = totalLoss(arm_pos=arm_pos, robot_pos=robot_pos)
    print("loss ", loss)
    # storing the calculated loss in a list
    loss_BGD.append(loss.item())
    # backward pass for computing the gradients of the loss w.r.t to learnable parameters
    loss.backward()
    # updateing the parameters after each iteration
    print("joint angles ", joint_angles)
    print(joint_angles.requires_grad)
    print(joint_angles.requires_grad_)
    joint_angles = joint_angles - step_size * joint_angles.grad
    # zeroing gradients after each iteration
    joint_angles.grad.data.zero_()
    # priting the values for understanding
    # print('{}, \t{}, \t{}, \t{}'.format(i, loss.item(), w.item(), b.item()))