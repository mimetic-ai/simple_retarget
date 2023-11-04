import torch
# import kinpy as kp
# import math
from robot_analysis import TiagoDual
import numpy as np
import matplotlib.pyplot as plt
# import pinocchio as pin
import json
import rospy
from std_msgs.msg import Float64MultiArray



urdf = 'robot_description/tiago_dual.urdf'
# urdf_left = 'robot_description/tiago_left_arm_1.urdf'
# urdf_right = 'robot_description/tiago_right_arm_1.urdf'
# model_left = pin.buildModelFromUrdf(urdf_left)
# model_right = pin.buildModelFromUrdf(urdf_right)
# pos_lim_lo_left = torch.tensor(model_left.lowerPositionLimit)
# pos_lim_hi_left = torch.tensor(model_left.upperPositionLimit)
# pos_lim_lo_right = torch.tensor(model_right.lowerPositionLimit)
# pos_lim_hi_right = t)orch.tensor(model_right.upperPositionLimit)


# pos_lim_lo_left = torch.tensor([0.0,-1.57079633,-3.53429174,-0.39269908,-2.0943951,-1.57079633,-2.0943951])
# pos_lim_hi_left = torch.tensor([2.74889357,1.09083078,1.57079633,2.35619449,2.0943951,1.57079633,2.0943951])
pos_lim_hi_left = torch.tensor([1.50098, 1.46608, 3.857, 2.28638, 2.07696, 1.361, 2.0769])
pos_lim_lo_left = torch.tensor([-1.09956, -1.09956, -0.715585, -0.314159, -2.07694, -1.39626, -2.07694])

pos_lim_hi_right = pos_lim_hi_left
pos_lim_lo_right = pos_lim_lo_left


def wrtShoulder(points_dict):
    points_dict['l_wrist'] = points_dict['l_wrist'] - points_dict['l_shoulder']
    points_dict['l_elbow'] = points_dict['l_elbow'] - points_dict['l_shoulder']
    points_dict['l_shoulder'] = [0, 0, 0]
    return points_dict


def totalLossNoNorm(arm_pos, robot_pos, joint_angles, lb, ub, arm_name):
    print("arm_pos ",arm_pos)
    print("robot_pos ",robot_pos)
    if arm_name == 'left':
        true_elbow = (arm_pos['l_elbow'] - arm_pos['l_shoulder'])
        true_wrist = (arm_pos['l_wrist'] - arm_pos['l_elbow'])
        
        pred_elbow = (robot_pos['arm_left_3_link'] - robot_pos['arm_left_1_link'])
        pred_wrist = (robot_pos['arm_left_7_link'] - robot_pos['arm_left_3_link'])
    elif arm_name == 'right':
        true_elbow = (arm_pos['r_elbow'] - arm_pos['r_shoulder'])
        true_wrist = (arm_pos['r_wrist'] - arm_pos['r_elbow'])
        
        pred_elbow = (robot_pos['arm_right_3_link'] - robot_pos['arm_right_1_link'])
        pred_wrist = (robot_pos['arm_right_7_link'] - robot_pos['arm_right_3_link'])
    
    elbow_err = true_elbow - pred_elbow
    elbow_loss = torch.dot(elbow_err, elbow_err)
    
    wrist_err = true_wrist - pred_wrist
    wrist_loss = torch.dot(wrist_err, wrist_err)
    joint_loss = torch.div(torch.tensor([1e-8]), torch.pow(torch.dot((joint_angles - ub), (joint_angles - lb)), 2))
    total_loss = (2 * elbow_loss) + wrist_loss + (2 * joint_loss)
    print("total_loss ",total_loss)
    return total_loss


def totalLoss(arm_pos, robot_pos, joint_angles, lb, ub, arm_name):
    print("arm_pos ", arm_pos)
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
    total_loss = elbow_loss + wrist_loss + joint_loss
    print(total_loss)
    return total_loss



def forward(joint_angles, robot, arm_name):
    if arm_name == 'left':
        robot.setJointAnglesLeft(joint_angles)
        new_keypoints = robot.getKeyPointPosesLeft()
    elif arm_name == 'right':
        robot.setJointAnglesRight(joint_angles)
        new_keypoints = robot.getKeyPointPosesRight()
    return new_keypoints

def retarget_multi_seeds(arm_pos, arm_name, robot, num_seeds):
    loss_run_map = {}
    loss_list = []
    for runs in range(num_seeds):
        print("SEED NUMBER ", runs)
        joint_angles, loss = retarget(arm_pos=arm_pos, arm_name=arm_name, robot=robot, initial_guess=None)
        loss_run_map[loss] = joint_angles
        loss_list.append(loss)
    min_loss = min(loss_list)
    return loss_run_map[min_loss]


def retarget(arm_pos, arm_name, robot, initial_guess=None, max_iter=500):
    if arm_name == 'left':
        if initial_guess is None:
            joint_angles = []
            for x in range(7):
                joint_angles.append(torch.rand(1) * (pos_lim_hi_left[x] - pos_lim_lo_left[x]) + pos_lim_lo_left[x])
            joint_angles = torch.tensor(joint_angles, requires_grad=True)
        else:
            joint_angles = initial_guess

        step_size = 1e-1
        # loss_BGD = []
        i = 0
        eps = 0.001
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
            joint_angles = []
            for x in range(7):
                joint_angles.append(torch.rand(1) * (pos_lim_hi_right[x] - pos_lim_lo_right[x]) + pos_lim_lo_right[x])
            joint_angles = torch.tensor(joint_angles, requires_grad=True)
        else:
            joint_angles = initial_guess

        step_size = 1e-1
        # loss_BGD = []
        i = 0
        eps = 0.001
        delta = 1000

        while (i < max_iter and delta > eps):
            # making predictions with forward pass
            robot_pos = forward(joint_angles=joint_angles, robot=robot, arm_name=arm_name)
            # calculating the loss between original and predicted data points
            loss = totalLoss(arm_pos=arm_pos, robot_pos=robot_pos, 
                            joint_angles=joint_angles, lb=pos_lim_lo_right, ub=pos_lim_hi_right, arm_name=arm_name)
            print(loss)

            joint_angles.retain_grad()
            loss.backward()

            grad = joint_angles.grad.clone().detach()
            delta = torch.norm(grad)
            joint_angles = joint_angles - step_size * grad
            i += 1
    return (joint_angles, loss)

# path = 'tf_joint_motion.json'
# f = open(path)
# data = json.load(f)
# print(data[0])

tiago_robot = TiagoDual(urdf)

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

