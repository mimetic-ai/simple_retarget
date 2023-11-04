import rospy
import numpy as np
import torch
import json
#from ffa_msgs.msg import Joint3DList
from std_msgs.msg import Float64MultiArray
from simple_retarget import retarget, tiago_robot, retarget_multi_seeds
from control_msgs.msg import FollowJointTrajectoryGoal
from control_msgs.msg import FollowJointTrajectoryActionGoal
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from scipy.spatial.transform import Rotation as R
from moveit_commander import MoveGroupCommander
from moveit_msgs.srv import GetPositionIK
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import PositionIKRequest

nn_joint_angles = torch.tensor([ 2.0001,  0.7291,  0.1522,  1.6892, -0.0733, -0.1736,  0.0605])


moveit_group_left = "arm_left"
group_arm_left = MoveGroupCommander(moveit_group_left)

group_arm_right = MoveGroupCommander("arm_right")

group_arm_left.set_planner_id("SBLkConfigDefault")
group_arm_left.set_pose_reference_frame("base_footprint")


group_arm_right.set_planner_id("SBLkConfigDefault")
group_arm_right.set_pose_reference_frame("base_footprint")

#CAM_TO_BODY_ROT = np.matmul(np.array([[0,1,0],[-1,0,0],[0,0,1]]),np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))
CAM_TO_BODY_ROT = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=torch.float)
joint_angles_left = None
joint_angles_right = None

tiago_left_arm_pub = rospy.Publisher('/arm_left_controller/follow_joint_trajectory/goal', FollowJointTrajectoryActionGoal, queue_size=10)
tiago_right_arm_pub = rospy.Publisher('/arm_right_controller/follow_joint_trajectory/goal', FollowJointTrajectoryActionGoal, queue_size=10)

def createMessage(joint_angles, arm_name):
    ##setting velocities message with constant velocity value across the board
    velocities_msg = Float64MultiArray()
    velocities_msg.data = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    ###setting joint trajectory point message with positions and velocities
    joint_traj_point = JointTrajectoryPoint()
    joint_traj_point.positions = joint_angles
    joint_traj_point.velocities = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    joint_traj_point.time_from_start = rospy.Duration(0.1)
    ###setting joint trajectory message type with points and names
    joint_traj = JointTrajectory()
    joint_traj.joint_names = ["arm_{}_1_joint".format(arm_name), "arm_{}_2_joint".format(arm_name), "arm_{}_3_joint".format(arm_name), "arm_{}_4_joint".format(arm_name), "arm_{}_5_joint".format(arm_name), "arm_{}_6_joint".format(arm_name), "arm_{}_7_joint".format(arm_name)]
    joint_traj.points = [joint_traj_point]
    ###setting trajectory field of followjointtrajectorygoal with joint trajectory message
    goal = FollowJointTrajectoryGoal()
    goal.trajectory = joint_traj
    goal.trajectory = joint_traj
    final_msg = FollowJointTrajectoryActionGoal()
    final_msg.goal = goal
    return final_msg

def rotation_matrix(roll, pitch, yaw):
    # Convert degrees to radians
    roll_rad = np.radians(roll)
    pitch_rad = np.radians(pitch)
    yaw_rad = np.radians(yaw)

    # Calculate trigonometric values
    cos_roll = np.cos(roll_rad)
    sin_roll = np.sin(roll_rad)
    cos_pitch = np.cos(pitch_rad)
    sin_pitch = np.sin(pitch_rad)
    cos_yaw = np.cos(yaw_rad)
    sin_yaw = np.sin(yaw_rad)

    # Create the rotation matrices
    R_x = np.array([[1, 0, 0],
                    [0, cos_roll, -sin_roll],
                    [0, sin_roll, cos_roll]])

    R_y = np.array([[cos_pitch, 0, sin_pitch],
                    [0, 1, 0],
                    [-sin_pitch, 0, cos_pitch]])

    R_z = np.array([[cos_yaw, -sin_yaw, 0],
                    [sin_yaw, cos_yaw, 0],
                    [0, 0, 1]])

    # Combine the rotation matrices (roll, pitch, yaw)
    rotation_matrix = np.matmul(np.matmul(R_z, R_y), R_x)

    return rotation_matrix

def rotToQuat(rotation_matrix):
    r = R.from_matrix(rotation_matrix)
    quat = r.as_quat()
    return quat


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

###takes in position of shoulder, wrist, and elbow. returns orientation of end effector in the gazebo coordinate frame. 
def getOrientation(wrist_pos, elbow_pos):
    elbow_wrt_wrist = np.asarray(elbow_pos) - np.asarray(wrist_pos)
    wrist_wrt_elbow= np.asarray(wrist_pos) - np.asarray(elbow_pos)
    rho, pitch = cart2pol(elbow_wrt_wrist[1], elbow_wrt_wrist[2])
    rho, yaw = cart2pol(wrist_wrt_elbow[1], elbow_wrt_wrist[0])
    yaw -= np.pi/2
    roll = 0
    return (roll, pitch, yaw)



def joints_pose_cb(msg):
    global joint_angles_left, joint_angles_right
    left_arm_pos, right_arm_pos = {}, {}

    for joint in msg.data:
        joint_pos = np.array([[joint.x, joint.y, joint.z]]).T
        joint_pos = CAM_TO_BODY_ROT@joint_pos
        
        if joint.joint_name in ['l_shoulder','l_elbow','l_wrist']: 
            left_arm_pos[joint.joint_name] = torch.from_numpy(np.squeeze(joint_pos))

        elif joint.joint_name in ['r_shoulder','r_elbow','r_wrist']:
            right_arm_pos[joint.joint_name] = torch.from_numpy(np.squeeze(joint_pos))
    
    joint_angles_left = retarget(left_arm_pos, 'left', tiago_robot, joint_angles_left)
    joint_angles_right = retarget(right_arm_pos, 'right', tiago_robot, joint_angles_right)

    left_arm_msg = Float64MultiArray()
    left_arm_msg.data = [val.item() for val in joint_angles_left]

    right_arm_msg = Float64MultiArray()
    right_arm_msg.data = [val.item() for val in joint_angles_right]

    tiago_left_arm_pub.publish(left_arm_msg)
    tiago_right_arm_pub.publish(right_arm_msg)

def moveto_joint_configuration(joint_angles, move_group):
    joint_goal = move_group.get_current_joint_values()
    joint_goal[0] = nn_joint_angles[0]
    joint_goal[1] = nn_joint_angles[1]
    joint_goal[2] =nn_joint_angles[2]
    joint_goal[3] = nn_joint_angles[3]
    joint_goal[4] = nn_joint_angles[4]
    joint_goal[5] = nn_joint_angles[5]  # 1/6 of a turn
    joint_goal[6] = nn_joint_angles[6]
    move_group.go(joint_goal, wait=True)
    move_group.stop()



def mocap_retarget_publisher(path, publish_all):
    f = open(path)
    data = json.load(f)


    global joint_angles_left, joint_angles_right
    left_arm_pos, right_arm_pos = {}, {}
    # lwrist_trans = CAM_TO_BODY_ROT@(np.asarray(data[0]['lwrist']).T)
    # lhumerus_trans = CAM_TO_BODY_ROT@(np.asarray(data[0]['lhumerus']).T)
    # lclavicle_trans = CAM_TO_BODY_ROT@(np.asarray(data[0]['lclavicle']).T)
    # rwrist_trans = CAM_TO_BODY_ROT@(np.asarray(data[66]['rwrist']).T)
    # rhumerus_trans = CAM_TO_BODY_ROT@(np.asarray(data[66]['rhumerus']).T)
    # rclavicle_trans = CAM_TO_BODY_ROT@(np.asarray(data[66]['rclavicle']).T)

    # # lwrist_trans = CAM_TO_BODY_ROT@(np.asarray(data['lwrist'][0]).T)
    # # lhumerus_trans = CAM_TO_BODY_ROT@(np.asarray(data['lhumerus'][0]).T)
    # # lclavicle_trans = CAM_TO_BODY_ROT@(np.asarray(data['lclavicle'][0]).T)
    # rwrist_trans = transform_points(np.asarray(data['rwrist'][66]))
    # rhumerus_trans = transform_points(np.asarray(data['rhumerus'][66]))
    # rclavicle_trans = transform_points(np.asarray(data['rclavicle'][66]))

    # rwrist_trans = CAM_TO_BODY_ROT@(np.asarray(data['rwrist'][0]).T)
    # rhumerus_trans = CAM_TO_BODY_ROT@(np.asarray(data['rhumerus'][0]).T)
    # rclavicle_trans = CAM_TO_BODY_ROT@(np.asarray(data['rclavicle'][0]).T)

    # left_arm_pos['l_wrist'] = torch.tensor([0.0,2.0,0.0])
    # left_arm_pos['l_elbow'] = torch.tensor([0.0,1.0,0.0])
    # left_arm_pos['l_shoulder'] = torch.tensor([0.0,0.0,0.0])
    # trans_wrist = torch.mm(CAM_TO_BODY_ROT, torch.tensor(data['lwrist'][86]))
    # trans_elbow = torch.mm(CAM_TO_BODY_ROT, torch.tensor(data['lhumerus'][86]))
    # trans_shoulder = torch.mm(CAM_TO_BODY_ROT, torch.tensor(data['lclavicle'][86]))
    # left_arm_pos['l_wrist'] = CAM_TO_BODY_ROT.mm(torch.tensor(data['lwrist'][1806]).unsqueeze(-1)).squeeze()
    # left_arm_pos['l_elbow'] = CAM_TO_BODY_ROT.mm(torch.tensor(data['lhumerus'][1806]).unsqueeze(-1)).squeeze()
    # left_arm_pos['l_shoulder'] = CAM_TO_BODY_ROT.mm(torch.tensor(data['lclavicle'][1806]).unsqueeze(-1)).squeeze()
    right_arm_pos['r_wrist'] = CAM_TO_BODY_ROT.mm(torch.tensor(data['rwrist'][166]).unsqueeze(-1)).squeeze()
    right_arm_pos['r_elbow'] = CAM_TO_BODY_ROT.mm(torch.tensor(data['rhumerus'][166]).unsqueeze(-1)).squeeze()
    right_arm_pos['r_shoulder'] = CAM_TO_BODY_ROT.mm(torch.tensor(data['rclavicle'][166]).unsqueeze(-1)).squeeze()
    # joint_angles_left = retarget(left_arm_pos, 'left', tiago_robot, joint_angles_left)
    #joint_angles_right = retarget(right_arm_pos, 'right', tiago_robot, joint_angles_right)
    # joint_angles_guess=joint_angles = group_arm_left.get_current_joint_values()
    # print("initial guess ", joint_angles_guess)
    # joint_angles_left = retarget_multi_seeds(arm_pos=left_arm_pos, arm_name='left', robot=tiago_robot, num_seeds=3)
    joint_angles_right = retarget_multi_seeds(arm_pos=right_arm_pos, arm_name='right', robot=tiago_robot, num_seeds=3)
    # print("joint_angles_left ",joint_angles_left)
    #print("joint_angles_left ",joint_angles_left)
    # ##setting float64 array message witih joint values
    # left_arm_msg = Float64MultiArray()
    # left_arm_msg.data = [val.item() for val in joint_angles_left]
    # final_msg_left = createMessage(joint_angles=joint_angles_left, arm_name="left")
    final_msg_right = createMessage(joint_angles=joint_angles_right, arm_name="right")
    ####publishing message
    # tiago_left_arm_pub.publish(final_msg_left)
    #tiago_left_arm_pub.publish(final_msg_left)
    tiago_right_arm_pub.publish(final_msg_right)



if __name__ == '__main__':
    rospy.init_node('retarget_node', anonymous=True)
    r = rospy.Rate(10)

    while not rospy.is_shutdown():
        try:
            mocap_retarget_publisher(path='striking_data_new.json',publish_all=False)
            rospy.spin()
            r.sleep()
        except rospy.ROSInterruptException:
            exit()
