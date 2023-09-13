import torch
# import kinpy as kp
from sklearn.metrics import mean_squared_error
# import math
from image_proc import *
from robot_analysis import TiagoDual
import numpy as np
import matplotlib.pyplot as plt
import pinocchio as pin
import json
import rospy
from std_msgs.msg import Float64MultiArray

urdf_left = 'robot_description/tiago_left_arm_1.urdf'
urdf_right = 'robot_description/tiago_right_arm_1.urdf'
model_left = pin.buildModelFromUrdf(urdf_left)
model_right = pin.buildModelFromUrdf(urdf_right)
pos_lim_lo_left = torch.tensor(model_left.lowerPositionLimit)
pos_lim_hi_left = torch.tensor(model_left.upperPositionLimit)
pos_lim_lo_right = torch.tensor(model_right.lowerPositionLimit)
pos_lim_hi_right = torch.tensor(model_right.upperPositionLimit)



def totalLoss(arm_pos, robot_pos, joint_angles, lb, ub, arm_name):
    if arm_name == 'left':
        true_elbow_len = torch.norm(arm_pos['l_elbow'] - arm_pos['l_shoulder'])
        true_wrist_len = torch.norm(arm_pos['l_wrist'] - arm_pos['l_elbow'])

        true_elbow = (arm_pos['l_elbow'] - arm_pos['l_shoulder'])/float(true_elbow_len)
        true_wrist = (arm_pos['l_wrist'] - arm_pos['l_elbow'])/float(true_wrist_len)

        pred_elbow_len = torch.norm(robot_pos['arm_left_3_link'] - robot_pos['arm_left_1_link'])
        pred_wrist_len = torch.norm(robot_pos['arm_left_7_link'] - robot_pos['arm_left_3_link'])
        
        pred_elbow = (robot_pos['arm_left_3_link'] - robot_pos['arm_left_1_link'])/float(pred_elbow_len)
        pred_wrist = (robot_pos['arm_left_7_link'] - robot_pos['arm_left_3_link'])/float(pred_wrist_len)
    elif arm_name == 'right':
        true_elbow_len = torch.norm(arm_pos['r_elbow'] - arm_pos['r_shoulder'])
        true_wrist_len = torch.norm(arm_pos['r_wrist'] - arm_pos['r_elbow'])

        true_elbow = (arm_pos['r_elbow'] - arm_pos['r_shoulder'])/float(true_elbow_len)
        true_wrist = (arm_pos['r_wrist'] - arm_pos['r_elbow'])/float(true_wrist_len)

        pred_elbow_len = torch.norm(robot_pos['arm_right_3_link'] - robot_pos['arm_right_1_link'])
        pred_wrist_len = torch.norm(robot_pos['arm_right_7_link'] - robot_pos['arm_right_3_link'])
        
        pred_elbow = (robot_pos['arm_right_3_link'] - robot_pos['arm_right_1_link'])/float(pred_elbow_len)
        pred_wrist = (robot_pos['arm_right_7_link'] - robot_pos['arm_right_3_link'])/float(pred_wrist_len)
    
    elbow_err = true_elbow - pred_elbow
    elbow_loss = torch.dot(elbow_err, elbow_err)
    
    wrist_err = true_wrist - pred_wrist
    wrist_loss = torch.dot(wrist_err, wrist_err)
    # print("elbow rquires grad ", elbow_loss.grad)
    # print("wrist rquires grad ", wrist_loss.grad)
    joint_loss = torch.div(torch.tensor([1e-8]), torch.pow(torch.dot((joint_angles - ub), (joint_angles - lb)), 2))
    return elbow_loss + wrist_loss + joint_loss



def forward(joint_angles, robot, arm_name):
    if arm_name == 'left':
        robot.setJointAnglesLeft(joint_angles)
        new_keypoints = robot.getKeyPointPosesLeft()
    elif arm_name == 'right':
        robot.setJointAnglesRight(joint_angles)
        new_keypoints = robot.getKeyPointPosesRight()
    return new_keypoints

def retarget(arm_pos, arm_name, robot, initial_guess=None, max_iter=500):
    if arm_name == 'left':
        if initial_guess is None:
            joint_angles = torch.tensor(data=(pos_lim_lo_left+pos_lim_hi_left)/2, dtype=torch.float, requires_grad=True)
        else:
            joint_angles = initial_guess

        step_size = 1e-1
        # loss_BGD = []
        i = 0
        eps = 0.1
        delta = 1000

        while (i < max_iter and delta > eps):
            # making predictions with forward pass
            robot_pos = forward(joint_angles=joint_angles, robot=robot, arm_name=arm_name)
            # calculating the loss between original and predicted data points
            loss = totalLoss(arm_pos=arm_pos, robot_pos=robot_pos, 
                            joint_angles=joint_angles, lb=pos_lim_lo_left, ub=pos_lim_hi_left, arm_name=arm_name)
            print(loss)

            joint_angles.retain_grad()
            loss.backward()

            grad = joint_angles.grad.clone().detach()
            delta = torch.norm(grad)
            joint_angles = joint_angles - step_size * grad
            i += 1
    elif arm_name == 'right':
        if initial_guess is None:
            joint_angles = torch.tensor(data=(pos_lim_hi_right+pos_lim_lo_right)/2, dtype=torch.float, requires_grad=True)
        else:
            joint_angles = initial_guess

        step_size = 1e-1
        # loss_BGD = []
        i = 0
        eps = 0.1
        delta = 1000

        while (i < max_iter and delta > eps):
            # making predictions with forward pass
            robot_pos = forward(joint_angles=joint_angles, robot=robot, arm_name=arm_name)
            # calculating the loss between original and predicted data points
            loss = totalLoss(arm_pos=arm_pos, robot_pos=robot_pos, 
                            joint_angles=joint_angles, lb=pos_lim_lo_right, ub=pos_lim_hi_left, arm_name=arm_name)
            print(loss)

            joint_angles.retain_grad()
            loss.backward()

            grad = joint_angles.grad.clone().detach()
            delta = torch.norm(grad)
            joint_angles = joint_angles - step_size * grad
            i += 1
    return joint_angles

path = 'tf_joint_motion.json'
f = open(path)
data = json.load(f)

tiago_robot = TiagoDual(urdf_right=urdf_right, urdf_left=urdf_left)

joint_trajectory_right = torch.empty((120, 7))
joint_angles_right = None
joint_trajectory_left = torch.empty((120, 7))
joint_angles_left = None
# left_arm_pub = rospy.Publisher('yumi_left_arm_controller/command', Float64MultiArray, queue_size=10)
# right_arm_pub = rospy.Publisher('yumi_right_arm_controller/command', Float64MultiArray, queue_size=10)
arm_pos = {}
# rospy.init_node('yumi_retargeting_node')
# r = rospy.Rate(1)

left_arm_data = Float64MultiArray()
right_arm_data = Float64MultiArray()
T = 120

