import xml.etree.ElementTree as ET
import math

# Define a function to calculate the distance between two points in 3D space
def distance_between_points(point1, point2):
    return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))

# Load and parse the URDF file
urdf_file = "robot_description/tiago_left_arm_1.urdf"
tree = ET.parse(urdf_file)
root = tree.getroot()

# Create a dictionary to store joint positions
joint_positions = {}

# Iterate through the URDF elements to extract joint positions
for joint in root.findall(".//joint"):
    joint_name = joint.attrib["name"]
    
    # Extract the origin element of the joint
    origin_element = joint.find("origin")
    
    if origin_element is not None:
        xyz = origin_element.attrib.get("xyz", "0 0 0").split()
        position = [float(coord) for coord in xyz]
        joint_positions[joint_name] = position

# Calculate the distances between adjacent joints
joint_names = list(joint_positions.keys())
distances = {}
for i in range(len(joint_names) - 1):
    joint1 = joint_names[i]
    joint2 = joint_names[i + 1]
    position1 = joint_positions[joint1]
    position2 = joint_positions[joint2]
    dist = distance_between_points(position1, position2)
    distances[(joint1, joint2)] = dist

# Print the distances
for joint1, joint2 in distances:
    dist = distances[(joint1, joint2)]
    print(f"Distance between {joint1} and {joint2}: {dist} units")
