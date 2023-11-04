#!/usr/bin/env python

import rospy
from moveit_commander import RobotCommander
from std_msgs.msg import Float64MultiArray
import json
import sys
import os

# Initialize the ROS node
rospy.init_node('tiago_joint_angles_logger')

# Initialize the MoveIt Commander
robot = RobotCommander()

# Initialize a dictionary to store the joint angles and timestamps
data_dict = {"timestamps": [], "left_arm_angles": [], "right_arm_angles": []}

# Function to log the joint angles
def log_joint_angles():
    current_time = rospy.Time.now()
    left_arm_joint_angles = robot.get_group_joint_values("left_arm")
    right_arm_joint_angles = robot.get_group_joint_values("right_arm")

    data_dict["timestamps"].append(current_time.to_sec())
    data_dict["left_arm_angles"].append(left_arm_joint_angles)
    data_dict["right_arm_angles"].append(right_arm_joint_angles)

# Main loop
try:
    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown():
        log_joint_angles()
        rate.sleep()
except KeyboardInterrupt:
    pass

# Save the data to a JSON file
output_file = "tiago_joint_angles.json"
with open(output_file, 'w') as f:
    json.dump(data_dict, f)

print("Data saved to", output_file)

# Shutdown the ROS node
rospy.signal_shutdown("Keyboard interrupt")

