import cv2
import mediapipe as mp
import numpy as np
import torch
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


##Z forward
##yaw is around Z

##Y up
##pitch is around y

##X side
##roll is around X





def wrtShoulder(points_dict):
    points_dict['l_wrist'] = points_dict['l_wrist'] - points_dict['l_shoulder']
    points_dict['l_elbow'] = points_dict['l_elbow'] - points_dict['l_shoulder']
    points_dict['l_shoulder'] = [0, 0, 0]
    return points_dict


def getArmPosesFrame(input):
    image = None
    if (type(input) == str):
        frame = cv2.imread(input)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
    else:
        image = input
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        results = pose.process(image)
        try:
            landmarks = results.pose_landmarks.landmark
            # Get coordinates
            l_shoulder = torch.tensor([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z])
            l_elbow = torch.tensor([landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z])
            l_wrist = torch.tensor([landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z]) 
            return {'l_shoulder': l_shoulder, 'l_elbow': l_elbow, 'l_wrist': l_wrist}  
        except:
            return {'l_shoulder': None, 'l_elbow': None, 'l_wrist': None}

def getEndEffectorPose(input):
    wrist_translation = input['l_wrist']
    elbow_translation = input['l_elbow']
    roll = np.arctan2(wrist_translation[1] - elbow_translation[1], wrist_translation[2] - elbow_translation[2])
    pitch = np.arctan2(wrist_translation[0] - elbow_translation[0], wrist_translation[2] - elbow_translation[2])
    yaw = 0
    return (np.asarray(wrist_translation), np.asarray([roll, pitch, yaw]))

    



def getArmPosesVideo(input_path):
    cap = cv2.VideoCapture(input_path)
    ## Setup mediapipe instance
    return_dict = {"l_shoulder": [], "l_elbow": [], 'l_wrist': []}
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                frame_dict = getArmPosesFrame(image)
                print(frame_dict)
                for keys in return_dict:
                    if frame_dict[keys] is not None:
                        return_dict[keys].append(frame_dict[keys])
        cap.release()
    cv2.destroyAllWindows()
    return return_dict


# print(wrtShoulder(getArmPosesFrame('media/sample_retarget_pose.jpg')))


#print(getArmPosesVideo('media/sample_retarget_motion.mp4'))
    






