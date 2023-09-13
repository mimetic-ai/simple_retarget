import rospy
import numpy as np
import torch

from ffa_msgs.msg import Joint3DList
from std_msgs.msg import Float64MultiArray
from simple_retarget import retarget, tiago_robot

CAM_TO_BODY_ROT = np.array([[0 ,0, -1], [1, 0, 0], [0, -1, 0]])
joint_angles_left = None
joint_angles_right = None

yumi_left_arm_pub = rospy.Publisher('yumi_left_arm_controller/command', Float64MultiArray, queue_size=10)
yumi_right_arm_pub = rospy.Publisher('yumi_right_arm_controller/command', Float64MultiArray, queue_size=10)


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

    yumi_left_arm_pub.publish(left_arm_msg)
    yumi_right_arm_pub.publish(right_arm_msg)


if __name__ == '__main__':
    joints_list_sub = rospy.Subscriber('/triangulated_joint_pos', Joint3DList, joints_pose_cb, queue_size=10)
    

    rospy.init_node('retarget_node', anonymous=True)
    r = rospy.Rate(10)

    while not rospy.is_shutdown():
        try:
            rospy.spin()
            r.sleep()
        except rospy.ROSInterruptException:
            exit()
