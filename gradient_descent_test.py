import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math
from image_proc import *
from robot_analysis import RobotArm

arm_pos = wrtShoulder(getArmPosesFrame('media/sample_retarget_pose.jpg'))
panda_robot = RobotArm('robot_description/panda.urdf')


# Set up the data and model
#arm_pos_tensor = tf.constant([arm_pos['l_shoulder'], arm_pos['l_elbow'], arm_pos['l_wrist']])


angles = tf.constant([[.2], [.2], [.2], [.2], [.2], [.2], [.2]])
  
w_1 = tf.Variable(0.)
b_1 = tf.Variable(0.)
w_2 = tf.Variable(0.)
b_2 = tf.Variable(0.)
w_3 = tf.Variable(0.)
b_3 = tf.Variable(0.)
w_4 = tf.Variable(0.)
b_4 = tf.Variable(0.)
w_5 = tf.Variable(0.)
b_5 = tf.Variable(0.)
w_6 = tf.Variable(0.)
b_6 = tf.Variable(0.)
w_7 = tf.Variable(0.)
b_7 = tf.Variable(0.)
params_list = [(w_1, b_1), (w_2, b_2), (w_3, b_3), (w_4, b_4), (w_5, b_5), (w_6, b_6), (w_7, b_7)]

def forward(angles, params_list, robot):
    joint_angles = []
    for x in range(len(angles)):
        joint_angles.append(float(angles[x] * float(params_list[x][0]) + float(params_list[x][1])))
    robot.setJointAngles(joint_angles)
    print(robot.joint_angles)
    robot.updateArmPose()
    new_keypoints = robot.getKeyPointPoses()
    return new_keypoints
  
# Define the model and loss function
def model(x):
    return w * x + b
  
def loss(predicted_y, true_y):
    return tf.reduce_mean(tf.square(predicted_y - true_y))

def totalLoss(arm_pos, robot_pos):
    true_elbow = arm_pos['l_elbow']
    true_wrist = arm_pos['l_wrist']
    pred_elbow = robot_pos['panda_link5_frame']
    pred_wrist = robot_pos['panda_hand_frame']
    elbow_loss = mean_squared_error(y_true=true_elbow, y_pred=pred_elbow)
    wrist_loss = mean_squared_error(y_true=true_wrist, y_pred=pred_wrist)
    total_loss = tf.reduce_mean(tf.constant([elbow_loss, wrist_loss]))
    return total_loss
  
# Set the learning rate
learning_rate = 0.001
  
# Training loop
losses = []
for i in range(250):
    with tf.GradientTape() as tape:
        predicted_pos = forward(angles, params_list, panda_robot)
        current_loss = totalLoss(arm_pos, predicted_pos)
    gradients = tape.gradient(current_loss, [w_1, b_1])
    w_1.assign_sub(learning_rate * gradients[0])
    b_1.assign_sub(learning_rate * gradients[1])
      
    losses.append(current_loss.numpy())
  
# Plot the loss
plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()