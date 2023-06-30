cmake_minimum_required(VERSION 3.0.2)
project(simple_retarget)

find_package(catkin REQUIRED COMPONENTS
    std_msgs
    rospy
)

# catkin_python_setup()

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

catkin_package()

catkin_install_python(PROGRAMS
  simple_retarget.py
  retarget_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch urdf
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
