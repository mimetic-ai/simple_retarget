import rospy
import numpy as np
import torch
import json

#from ffa_msgs.msg import Joint3DList
from std_msgs.msg import Float64MultiArray
from simple_retarget import retarget, tiago_robot
from control_msgs.msg import FollowJointTrajectoryGoal
from control_msgs.msg import FollowJointTrajectoryActionGoal
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint


#CAM_TO_BODY_ROT = np.matmul(np.array([[0,1,0],[-1,0,0],[0,0,1]]),np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))
CAM_TO_BODY_ROT = np.array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]])
joint_angles_left = None
joint_angles_right = None

tiago_left_arm_pub = rospy.Publisher('/arm_left_controller/follow_joint_trajectory/goal', FollowJointTrajectoryActionGoal, queue_size=10)
tiago_right_arm_pub = rospy.Publisher('/arm_right_controller/follow_joint_trajectory/goal', FollowJointTrajectoryActionGoal, queue_size=10)


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


def transform_points(points):
    print(points)
    points[0]*= -1
    points[1]*= 1
    points[2]*= -1
    print(points)
    return points



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

    rwrist_trans = CAM_TO_BODY_ROT@(np.asarray(data['rwrist'][0]).T)
    rhumerus_trans = CAM_TO_BODY_ROT@(np.asarray(data['rhumerus'][0]).T)
    rclavicle_trans = CAM_TO_BODY_ROT@(np.asarray(data['rclavicle'][0]).T)

    # left_arm_pos['l_wrist'] = torch.tensor([0.0,2.0,0.0])
    # left_arm_pos['l_elbow'] = torch.tensor([0.0,1.0,0.0])
    # left_arm_pos['l_shoulder'] = torch.tensor([0.0,0.0,0.0])
    right_arm_pos['r_wrist'] = torch.tensor([0.537, 0.0, -0.214])
    right_arm_pos['r_elbow'] = torch.tensor([0.0, 0.0, -0.214])
    right_arm_pos['r_shoulder'] = torch.tensor([0.0, 0.0, 0.0])
    # joint_angles_left = retarget(left_arm_pos, 'left', tiago_robot, joint_angles_left)
    joint_angles_right = retarget(right_arm_pos, 'right', tiago_robot, joint_angles_right)
    # print("joint_angles_left ",joint_angles_left)
    print("joint_angles_right ",joint_angles_right)
    # ##setting float64 array message witih joint values
    # left_arm_msg = Float64MultiArray()
    # left_arm_msg.data = [val.item() for val in joint_angles_left]

    right_arm_msg = Float64MultiArray()
    right_arm_msg.data = [val.item() for val in joint_angles_right]
    ##setting velocities message with constant velocity value across the board
    velocities_msg = Float64MultiArray()
    velocities_msg.data = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    ###setting joint trajectory point message with positions and velocities
    joint_traj_point_left = JointTrajectoryPoint()
    joint_traj_point_right = JointTrajectoryPoint()
    #joint_traj_point_left.positions = left_arm_m
    joint_traj_point_left.positions = joint_angles_left
    joint_traj_point_left.velocities = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    joint_traj_point_left.time_from_start = rospy.Duration(0.1)
    # joint_traj_point_right.positions = right_arm_msg
    joint_traj_point_right.positions = joint_angles_right
    joint_traj_point_right.velocities = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    joint_traj_point_right.time_from_start = rospy.Duration(0.1)
    ###setting joint trajectory message type with points and names
    joint_traj_left = JointTrajectory()
    joint_traj_right = JointTrajectory()
    joint_traj_left.joint_names = ["arm_left_1_joint","arm_left_2_joint","arm_left_3_joint","arm_left_4_joint","arm_left_5_joint","arm_left_6_joint","arm_left_7_joint"]
    joint_traj_left.points = [joint_traj_point_left]

    joint_traj_right.joint_names = ["arm_right_1_joint","arm_right_2_joint","arm_right_3_joint","arm_right_4_joint","arm_right_5_joint","arm_right_6_joint","arm_right_7_joint"]
    joint_traj_right.points = [joint_traj_point_right]
    ###setting trajectory field of followjointtrajectorygoal with joint trajectory message
    goal_left = FollowJointTrajectoryGoal()
    goal_right = FollowJointTrajectoryGoal()
    goal_left.trajectory = joint_traj_left
    goal_right.trajectory = joint_traj_right
    final_msg_left = FollowJointTrajectoryActionGoal()
    final_msg_right = FollowJointTrajectoryActionGoal()
    final_msg_left.goal = goal_left
    final_msg_right.goal = goal_right    
    ####publishing message
    # tiago_left_arm_pub.publish(final_msg_left)
    tiago_right_arm_pub.publish(final_msg_right)



if __name__ == '__main__':
    rospy.init_node('retarget_node', anonymous=True)
    r = rospy.Rate(10)

    while not rospy.is_shutdown():
        try:
            mocap_retarget_publisher(path='AMCParser/boxing_arm_data_2.json',publish_all=False)
            rospy.spin()
            r.sleep()
        except rospy.ROSInterruptException:
            exit()
