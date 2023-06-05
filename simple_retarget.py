import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math
from image_proc import *
from robot_analysis import RobotArm

def totalLoss(arm_pos, robot_pos):
    true_elbow = arm_pos['l_elbow']
    true_wrist = arm_pos['l_wrist']
    pred_elbow = robot_pos['panda_link5_frame']
    pred_wrist = robot_pos['panda_hand_frame']
    elbow_loss = mean_squared_error(y_true=true_elbow, y_pred=pred_elbow)
    wrist_loss = mean_squared_error(y_true=true_wrist, y_pred=pred_wrist)
    total_loss = tf.reduce_mean(tf.constant([elbow_loss, wrist_loss]))
    return total_loss

def forward(joint_angles, robot):
    robot.setJointAngles(joint_angles)
    print(robot.joint_angles)
    robot.updateArmPose()
    new_keypoints = robot.getKeyPointPoses()
    return new_keypoints

arm_pos = wrtShoulder(getArmPosesFrame('media/sample_retarget_pose.jpg'))
panda_robot = RobotArm('robot_description/panda.urdf')


# Set up the data and model
arm_pos_tensor = tf.constant([arm_pos['l_shoulder'], arm_pos['l_elbow'], arm_pos['l_wrist']])

  
ang1 = tf.Variable(0.)
ang2 = tf.Variable(0.)
ang2 = tf.Variable(0.)
ang3 = tf.Variable(0.)
ang4 = tf.Variable(0.)
ang5 = tf.Variable(0.)
ang6 = tf.Variable(0.)
ang7 = tf.Variable(0.)

joint_angles = [float(ang1), float(ang2), float(ang3), float(ang4), float(ang5), float(ang6), float(ang7)]
  
# Set the learning rate
learning_rate = 0.001
  
# Training loop
losses = []
for i in range(250):
    with tf.GradientTape() as tape:
        new_keypoints = forward(joint_angles=joint_angles, robot=panda_robot)
        current_loss = totalLoss(arm_pos=arm_pos, robot_pos=new_keypoints)
    gradients = tape.gradient(current_loss, [ang1, ang2, ang3, ang4, ang5, ang6, ang7])
    print(current_loss)
    print(gradients)
    ang1.assign_sub(learning_rate * gradients[0])
    ang2.assign_sub(learning_rate * gradients[1])
    ang3.assign_sub(learning_rate * gradients[2])
    ang4.assign_sub(learning_rate * gradients[3])
    ang5.assign_sub(learning_rate * gradients[4])
    ang6.assign_sub(learning_rate * gradients[5])
    ang7.assign_sub(learning_rate * gradients[6])
      
    losses.append(current_loss.numpy())
    joint_angles = [float(ang1), float(ang2), float(ang3), float(ang4), float(ang5), float(ang6), float(ang7)]

  
# # Plot the loss
# plt.plot(losses)
# plt.xlabel("Iteration")
# plt.ylabel("Loss")
# plt.show()


