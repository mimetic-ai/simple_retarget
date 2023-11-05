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

# Create a dictionary to store link lengths
link_lengths = {}

# Iterate through the URDF elements to calculate link lengths
for link in root.findall(".//link"):
    link_name = link.attrib["name"]
    
    # Find the child joint of this link
    child_joint = root.find(f".//joint[@parent='{link_name}']")
    
    if child_joint is not None:
        child_joint_name = child_joint.attrib["name"]
        if child_joint_name in joint_positions:
            parent_link = link_lengths.get(link_name, 0)
            child_joint_position = joint_positions[child_joint_name]
            link_lengths[link_name] = parent_link + distance_between_points([0, 0, 0], child_joint_position)

# Print the link lengths
for link_name, length in link_lengths.items():
    print(f"Link {link_name}: Length = {length} units")
