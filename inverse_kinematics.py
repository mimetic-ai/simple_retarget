import numpy as np
from scipy.spatial.transform import Rotation as R
from trac_ik_python.trac_ik import IK




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
    rho, yaw = cart2pol(wrist_wrt_elbow[1], wrist_wrt_elbow[0])
    yaw -= np.pi/2
    roll = 0
    return (roll, pitch, yaw)

def track_ik(position, orientation,arm_name):
    ik_solver = IK("torso_lift_link","arm_{}_7_link".format(arm_name))
    seed_state = [0.0] * ik_solver.number_of_joints
    result = ik_solver.get_ik(seed_state, position[0], position[1], position[2], orientation[0], orientation[1], orientation[2], orientation[3])
    return result


def setTargetPose(wrist_pos, elbow_pos, arm_name):
    roll, pitch, yaw = getOrientation(wrist_pos=wrist_pos, elbow_pos=elbow_pos)
    print("roll", roll)
    print("pitch", pitch)
    print("yaw",yaw)
    rot = rotation_matrix(roll, pitch, yaw)
    quat = rotToQuat(rotation_matrix=rot)
    print("quat", quat)
    joint_angles = track_ik(position=wrist_pos, orientation=quat, arm_name=arm_name)
    return joint_angles

#print(setTargetPose([3,0,0], [1,0,0],"right"))
wrist_pos = [0.5, 0.3, 0.5]
elbow_pos = [0.0, 0.3, 0.5]
print(getOrientation(wrist_pos=wrist_pos, elbow_pos=elbow_pos))
print(setTargetPose(wrist_pos=wrist_pos, elbow_pos=elbow_pos,arm_name="left"))
