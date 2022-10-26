cmake_minimum_required(VERSION 3.0.2)
project(slam_module)

add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  mobinn_nav_msgs
  nav_msgs
  pcl_conversions
  pcl_msgs
  pcl_ros
  roscpp
  sensor_msgs
  std_msgs
  tf
)

find_package(Ceres REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(GTSAM REQUIRED QUIET)


catkin_package(
 INCLUDE_DIRS include
 CATKIN_DEPENDS geometry_msgs mobinn_nav_msgs nav_msgs pcl_conversions pcl_msgs pcl_ros roscpp sensor_msgs std_msgs tf
)

include_directories(
include
  ${catkin_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
	${GTSAM_INCLUDE_DIR}
)

link_directories(
  include
  ${GTSAM_LIBRARY_DIRS}
)

add_executable(${PROJECT_NAME}_node src/main.cpp)

target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
  ${CERES_LIBRARIES}
  Eigen3::Eigen
  gtsam
)
add_dependencies(${PROJECT_NAME}_node mobinn_nav_msgs_generate_messages_cpp)


add_executable(${PROJECT_NAME}_node_indoors src/main_indoors.cpp)
target_link_libraries(${PROJECT_NAME}_node_indoors
  ${catkin_LIBRARIES}
  ${CERES_LIBRARIES}
  Eigen3::Eigen
  gtsam
)
add_dependencies(${PROJECT_NAME}_node_indoors mobinn_nav_msgs_generate_messages_cpp)


add_executable(${PROJECT_NAME}_node_cloud_only src/main_cloud_only.cpp)
target_link_libraries(${PROJECT_NAME}_node_cloud_only
  ${catkin_LIBRARIES}
  ${CERES_LIBRARIES}
  Eigen3::Eigen
  gtsam
)
add_dependencies(${PROJECT_NAME}_node_cloud_only mobinn_nav_msgs_generate_messages_cpp)
