import torch
# import kinpy as kp
from sklearn.metrics import mean_squared_error
# import math
from image_proc import *
from robot_analysis import YumiArm
import numpy as np
import matplotlib.pyplot as plt
import pinocchio as pin
import json
import rospy
from std_msgs.msg import Float64MultiArray

model = pin.buildModelFromUrdf('robot_description/yumi.urdf')
pos_lim_lo = model.lowerPositionLimit
pos_lim_hi = model.upperPositionLimit
pos_lim_hi_left = torch.tensor(pos_lim_hi[0:7])
pos_lim_hi_right = torch.tensor(pos_lim_hi[7:14])

pos_lim_lo_left = torch.tensor(pos_lim_lo[0:7])
pos_lim_lo_right = torch.tensor(pos_lim_lo[7:14])


def totalLoss(arm_pos, robot_pos, joint_angles, lb, ub, arm_name):
    if arm_name == 'left':
        true_elbow_len = torch.norm(arm_pos['l_elbow'] - arm_pos['l_shoulder'])
        true_wrist_len = torch.norm(arm_pos['l_wrist'] - arm_pos['l_elbow'])

        true_elbow = (arm_pos['l_elbow'] - arm_pos['l_shoulder'])/float(true_elbow_len)
        true_wrist = (arm_pos['l_wrist'] - arm_pos['l_elbow'])/float(true_wrist_len)

        pred_elbow_len = torch.norm(robot_pos['yumi_link_3_l'] - robot_pos['yumi_link_1_l'])
        pred_wrist_len = torch.norm(robot_pos['yumi_link_7_l'] - robot_pos['yumi_link_3_l'])
        
        pred_elbow = (robot_pos['yumi_link_3_l'] - robot_pos['yumi_link_1_l'])/float(pred_elbow_len)
        pred_wrist = (robot_pos['yumi_link_7_l'] - robot_pos['yumi_link_3_l'])/float(pred_wrist_len)
    elif arm_name == 'right':
        true_elbow_len = torch.norm(arm_pos['r_elbow'] - arm_pos['r_shoulder'])
        true_wrist_len = torch.norm(arm_pos['r_wrist'] - arm_pos['r_elbow'])

        true_elbow = (arm_pos['r_elbow'] - arm_pos['r_shoulder'])/float(true_elbow_len)
        true_wrist = (arm_pos['r_wrist'] - arm_pos['r_elbow'])/float(true_wrist_len)

        pred_elbow_len = torch.norm(robot_pos['yumi_link_3_r'] - robot_pos['yumi_link_1_r'])
        pred_wrist_len = torch.norm(robot_pos['yumi_link_7_r'] - robot_pos['yumi_link_3_r'])
        
        pred_elbow = (robot_pos['yumi_link_3_r'] - robot_pos['yumi_link_1_r'])/float(pred_elbow_len)
        pred_wrist = (robot_pos['yumi_link_7_r'] - robot_pos['yumi_link_3_r'])/float(pred_wrist_len)
    
    elbow_err = true_elbow - pred_elbow
    elbow_loss = torch.dot(elbow_err, elbow_err)
    
    wrist_err = true_wrist - pred_wrist
    wrist_loss = torch.dot(wrist_err, wrist_err)
    # print("elbow rquires grad ", elbow_loss.grad)
    # print("wrist rquires grad ", wrist_loss.grad)
    joint_loss = torch.div(torch.tensor([1e-8]), torch.pow(torch.dot((joint_angles - ub), (joint_angles - lb)), 2))
    return elbow_loss + wrist_loss + joint_loss


# def newLoss(arm_pos, robot_pos):
#     true_elbow = arm_pos['elbow'].detach().numpy()
#     true_wrist = arm_pos['wrist'].detach().numpy()
#     pred_elbow = robot_pos['panda_link5'].detach().numpy()
#     pred_wrist = robot_pos['panda_hand'].detach().numpy()
#     true_elbow = np.asarray([true_elbow[0], true_elbow[2], true_elbow[1]])/np.linalg.norm(true_elbow)
#     true_wrist = np.asarray([true_wrist[0], true_wrist[2], true_wrist[1]])/np.linalg.norm(true_wrist)
#     pred_elbow = np.asarray([pred_elbow[0], pred_elbow[2], pred_elbow[1]])/np.linalg.norm(pred_elbow)
#     pred_wrist = np.asarray([pred_wrist[0], pred_wrist[2], pred_wrist[1]])/np.linalg.norm(pred_wrist)
#     elbow_loss = mean_squared_error(true_elbow, pred_elbow)
#     wrist_loss = mean_squared_error(true_wrist, pred_wrist)
#     return (elbow_loss + wrist_loss)/2


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
# # print('data type', type(data['rwrist']))
# # arm_pos = {'shoulder': None, 'elbow': None, 'wrist': None}
# # idx = 247
# # arm_pos['shoulder'] = torch.tensor(data['rclavicle'][idx])
# # arm_pos['elbow'] = torch.tensor(data['rhumerus'][idx])
# # arm_pos['wrist'] = torch.tensor(data['rwrist'][idx])
# # print('arm pose', arm_pos)

yumi_robot = YumiArm('robot_description/yumi.urdf')
# # end_effector_pose = getEndEffectorPose(arm_pos)

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

# for i in range(T):
#     arm_pos['shoulder_r'] = torch.tensor(data[i]['rclavicle'])
#     arm_pos['elbow_r'] = torch.tensor(data[i]['rhumerus'])
#     arm_pos['wrist_r'] = torch.tensor(data[i]['rwrist'])
#     joint_angles_right = retarget(arm_pos=arm_pos, robot=yumi_robot, initial_guess=joint_angles_right, arm_name="right")
#     arm_pos = {'shoulder_l': None, 'elbow_l': None, 'wrist_l': None}
#     arm_pos['shoulder_l'] = torch.tensor(data[i]['lclavicle'])
#     arm_pos['elbow_l'] = torch.tensor(data[i]['lhumerus'])
#     arm_pos['wrist_l'] = torch.tensor(data[i]['lwrist'])
#     joint_angles_left = retarget(arm_pos=arm_pos, robot=yumi_robot, initial_guess=joint_angles_left, arm_name="left")
#     joint_trajectory_right[i] = joint_angles_right
#     joint_trajectory_left[i] = joint_angles_left
#     left_arm_data.data = [0 for _ in range(7)]
#     right_arm_data.data = [0 for _ in range(7)]
#     left_arm_pub.publish(left_arm_data)
#     right_arm_pub.publish(right_arm_data)
    
# while not rospy.is_shutdown():
#     for i in range(T):
#         left_arm_data.data = [val.item() for val in joint_trajectory_left[i]]
#         right_arm_data.data = [val.item() for val in joint_trajectory_right[i]]
#         left_arm_pub.publish(left_arm_data)
#         right_arm_pub.publish(right_arm_data)
#         print(f'published frame {i}')
#         rospy.sleep(rospy.Duration(0.1))

# print(joint_trajectory_left)
# print(joint_trajectory_right)
# step_size = 1e-2
# loss_BGD = []
# n_iter = 2
# i = 0
# eps = 0.1
# loss = float('inf')

#####initial guess
# chain = kp.build_serial_chain_from_urdf(open('robot_description/panda.urdf').read(), end_link_name = 'panda_hand')
# end_effector_transform = kp.transform.Transform(rot = end_effector_pose[1], pos = end_effector_pose[0])
#end_effector_transform = kp.transform.Transform(rot = None, pos = end_effector_pose[0])
# ik_results = chain.inverse_kinematics(end_effector_transform)
# joint_angles = torch.tensor(ik_results, requires_grad=True, dtype=torch.float)
# joint_angles = torch.tensor(data=(lb+ub)/2, dtype=torch.float, requires_grad=True)

# norm_var = 1000
# print(norm_var)
# print(eps)
# while (i< 500 and norm_var > eps):
#     # making predictions with forward pass
#     robot_pos = forward(joint_angles=joint_angles, robot=panda_robot)
#     # calculating the loss between original and predicted data points
#     loss = totalLoss(arm_pos=arm_pos, robot_pos=robot_pos, joint_angles=joint_angles, lb=lb, ub=ub)
#     # print("loss ", loss)
#     # storing the calculated loss in a list
#     loss_BGD.append(loss.item())
#     # backward pass for computing the gradients of the loss w.r.t to learnable parameters
#     joint_angles.retain_grad()
#     loss.backward()
#     # updateing the parameters after each iteration3
#     # print("joint angles ", joint_angles)
#     grad = joint_angles.grad.clone().detach()
#     norm_var = torch.norm(grad)
#     joint_angles = joint_angles - step_size * grad
#     i += 1
    # we don't need this because joint angles is populated with 
    #   a vector that doesn't require grad 
    # print("grad later", joint_angles.grad.zero_())

    # priting the values for understanding
    # print('{}, \t{}, \t{}, \t{}'.format(i, loss.item(), w.item(), b.item()))


# print("loss ", loss)
# print(joint_angles)
# degrees_list = np.rad2deg(joint_angles.detach().numpy())
# print(degrees_list)
# print("reached end of file")
# Plot the loss
# plt.plot(loss_BGD)
# plt.xlabel("Iteration")
# plt.ylabel("Loss")
# plt.show()