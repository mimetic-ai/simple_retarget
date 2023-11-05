import json
import numpy as np
import math
import rospy
from control_msgs.msg import FollowJointTrajectoryGoal
from control_msgs.msg import FollowJointTrajectoryActionGoal
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from std_msgs.msg import Float64MultiArray

def norm(array):
    return array / np.linalg.norm(array)

def get_unit_representation(arm_dict, x):
    r_elbow_val = np.asarray(arm_dict['rhumerus'][x])
    r_wrist_val = np.asarray(arm_dict['rwrist'][x])
    l_elbow_val = np.asarray(arm_dict['lhumerus'][x])
    l_wrist_val = np.asarray(arm_dict['lwrist'][x])
    r_wrist_vector = norm(np.asarray(r_wrist_val) - np.asarray(r_elbow_val))
    l_wrist_vector = norm(np.asarray(l_wrist_val) - np.asarray(l_elbow_val))
    l_elbow_vector = norm(l_elbow_val)
    r_elbow_vector = norm(r_elbow_val)
    l_vector = np.asarray([l_elbow_vector[0], l_elbow_vector[1], l_elbow_vector[2], l_wrist_vector[0], l_wrist_vector[1], l_wrist_vector[2]])
    r_vector = np.asarray([r_elbow_vector[0], r_elbow_vector[1], r_elbow_vector[2], r_wrist_vector[0], r_wrist_vector[1], r_wrist_vector[2]])
    return (l_vector, r_vector)


def radians_to_degrees(radians):
    return radians * (180 / math.pi)
def degrees_to_radians(degrees):
    return degrees * (math.pi / 180)

def constrain(value, lower_bound, upper_bound):
    return max(lower_bound, min(value, upper_bound))

##UP is Y
##Z is straight forward
##X is left/right

def get_joint_1(arm_dict, x, arm_name):
    if arm_name == 'left':
        l_elbow_val = np.asarray(arm_dict['lhumerus'][x])
        straight = l_elbow_val[2]
        left_right = l_elbow_val[0]
    if arm_name == 'right':
        r_elbow_val = np.asarray(arm_dict['rhumerus'][x])
        straight = r_elbow_val[2]
        left_right = r_elbow_val[0]
    component_length = math.sqrt(math.pow(straight, 2) + math.pow(left_right, 2))
    calc_angle = math.asin(straight/component_length)
    return constrain(calc_angle, degrees_to_radians(-63), degrees_to_radians(86))


def get_joint_2(arm_dict, x, arm_name):
    if arm_name == 'left':
        l_elbow_val = np.asarray(arm_dict['lhumerus'][x])
        straight = l_elbow_val[2]
        up_down = l_elbow_val[1]
    elif arm_name == 'right':
        r_elbow_val = np.asarray(arm_dict['rhumerus'][x])
        straight = r_elbow_val[2]
        up_down = r_elbow_val[1]
    component_length = math.sqrt(math.pow(straight, 2) + math.pow(up_down, 2))
    calc_angle = math.asin(-1 * up_down/component_length)
    #calc_angle = radians_to_degrees(math.acos(-1 * up_down/component_length))
    return constrain(calc_angle, degrees_to_radians(-63), degrees_to_radians(86))

def get_joint_4(arm_dict, x, arm_name):
    if arm_name == 'left':
        vector = get_unit_representation(arm_dict, x)[0]
    elif arm_name == 'right':
        vector = get_unit_representation(arm_dict, x)[1]
    upper_vector = np.hsplit(vector, 2)[0]
    lower_vector = np.hsplit(vector, 2)[1]
    calc_angle = np.arccos(np.dot(upper_vector, lower_vector))
    #calc_angle = radians_to_degrees(math.acos(-1 * up_down/component_length))
    # return calc_angle
    return constrain(calc_angle, degrees_to_radians(-18), degrees_to_radians(131))

def get_joint_3(arm_dict, x, arm_name):
    if arm_name == 'left':
        vector = get_unit_representation(arm_dict, x)[0]
    elif arm_name == 'right':
        vector = get_unit_representation(arm_dict, x)[1]
    upper_vector = np.hsplit(vector, 2)[0]
    lower_vector = np.hsplit(vector, 2)[1]
    upper_no_vert = np.asarray([upper_vector[0], upper_vector[2]])
    lower_no_vert = np.asarray([lower_vector[0], lower_vector[2]])
    dot = np.dot(upper_no_vert, lower_no_vert)
    norm_upper= np.linalg.norm(upper_no_vert)
    norm_lower = np.linalg.norm(lower_no_vert)
    calc_angle = np.arccos(dot/(norm_upper * norm_lower))
    #calc_angle = radians_to_degrees(math.acos(-1 * up_down/component_length))
    return constrain(calc_angle, degrees_to_radians(-41), degrees_to_radians(221))


def get_joint_trajectory(arm_dict, start, end, arm_name):
    """
    Calculate the joint trajectory for a specific arm.

    Args:
        arm_dict (dict): Loaded data from a JSON file.
        start (int): Start index of the trajectory.
        end (int): End index of the trajectory.
        arm_name (str): Name of the arm ('left' or 'right').

    Returns:
        tuple: A tuple containing the joint trajectory and corresponding time stamps.
    """
    trajectory = []
    time_stamps = []
    for x in range(start, end):
        joint_1 = get_joint_1(arm_dict=arm_dict, x=x, arm_name=arm_name)
        joint_2 = get_joint_2(arm_dict=arm_dict, x=x, arm_name=arm_name)
        joint_3 = get_joint_3(arm_dict=arm_dict, x=x, arm_name=arm_name)
        joint_4 = get_joint_4(arm_dict=arm_dict, x=x, arm_name=arm_name)
        joint_5 = 0
        joint_6 = 0
        joint_7 = 0
        joint_angles = [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7]
        trajectory.append(joint_angles)
        time_stamps.append(x / 120)
    return (trajectory, time_stamps)



def get_vel_and_accel(trajectory, time_stamp):
    joint_angles = np.array(trajectory)
    timestamps = np.array(time_stamp)
    joint_velocities = np.gradient(joint_angles, timestamps, axis=0)
    # Calculate accelerations using np.gradient again
    joint_accelerations = np.gradient(joint_velocities, timestamps, axis=0)
    return joint_velocities, joint_accelerations

def get_joint_name_list(list_len, arm_name):
    joint_names = []
    for x in range(list_len):
        joint_names.append(["arm_{}_1_joint".format(arm_name), "arm_{}_2_joint".format(arm_name), "arm_{}_3_joint".format(arm_name), "arm_{}_4_joint".format(arm_name), "arm_{}_5_joint".format(arm_name), "arm_{}_6_joint".format(arm_name), "arm_{}_7_joint".format(arm_name)])
    return joint_names

def create_joint_trajectory_point_list(trajectory, vel,accel, time_stamp):
    joint_traj_points = []
    for x in range(len(trajectory)):
        joint_traj_point = JointTrajectoryPoint()
        joint_traj_point.positions = trajectory[x]
        joint_traj_point.velocities = vel[x]
        joint_traj_point.accelerations = accel[x]
        joint_traj_point.time_from_start = rospy.Duration(time_stamp[x])
        joint_traj_points.append(joint_traj_point)
    return joint_traj_points

def create_message(arm_dict, start, end, arm_name):
    """
    Create a FollowJointTrajectoryActionGoal message for a specific arm.

    Args:
        arm_dict (dict): Loaded data from a JSON file.
        start (int): Start index of the trajectory.
        end (int): End index of the trajectory.
        arm_name (str): Name of the arm ('left' or 'right').

    Returns:
        FollowJointTrajectoryActionGoal: Trajectory message for the specified arm.
    """
    trajectory, time_stamps = get_joint_trajectory(arm_dict=arm_dict, start=start, end=end, arm_name=arm_name)
    vel, accel = get_vel_and_accel(trajectory=trajectory, time_stamp=time_stamps)

    # Create a joint trajectory message with points and names
    joint_traj = JointTrajectory()
    joint_traj.joint_names = get_joint_name_list(len(time_stamps), arm_name)
    joint_traj.points = create_joint_trajectory_point_list(trajectory, vel, accel, time_stamps)

    # Create a trajectory message for the arm
    goal = FollowJointTrajectoryGoal()
    goal.trajectory = joint_traj

    final_msg = FollowJointTrajectoryActionGoal()
    final_msg.goal = goal

    return final_msg

def main():
    # Initialize the ROS node
    rospy.init_node('simplest_retarget', anonymous=True)

    # Load data from a JSON file
    data_file = 'AMCParser/mopping_data_new.json'
    with open(data_file) as f:
        loaded_data = json.load(f)

    # Define publishers for Tiago's left and right arm trajectory
    tiago_left_arm_pub = rospy.Publisher('/arm_left_controller/follow_joint_trajectory/goal', FollowJointTrajectoryActionGoal, queue_size=10)
    tiago_right_arm_pub = rospy.Publisher('/arm_right_controller/follow_joint_trajectory/goal', FollowJointTrajectoryActionGoal, queue_size=10)

    # Create and publish trajectory messages for the left and right arms
    left_msg = create_message(arm_dict=loaded_data, start=80, end=155, arm_name='left')
    right_msg = create_message(arm_dict=loaded_data, start=80, end=155, arm_name='right')

    tiago_left_arm_pub.publish(left_msg)
    tiago_right_arm_pub.publish(right_msg)


# Define other functions here (get_joint_1, get_joint_2, etc.) with comments explaining their functionality.

if __name__ == '__main__':
    main()
    rospy.spin()  # Keep the node running

