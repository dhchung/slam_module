#include <iostream>
#include <thread>
#include <mutex>
#include <queue>
#include <unordered_map>

#include <ros/ros.h>
#include "tic_toc.h"
#include "lidarOdometry.hpp"
#include "utils.h"

ros::Publisher pubRequestLocalization;
Eigen::Matrix4f currentPose;

bool pose_initialized;

void OnSubscribeLidarPointCloud(const sensor_msgs::PointCloud2::ConstPtr & msg) {
    mobinn_nav_msgs::RequestLocalization req_msg;
    req_msg.header.stamp = msg->header.stamp;
    req_msg.nodeNumber = 0;
    req_msg.currentPose = EigenToOdometry(currentPose);
    req_msg.inputCloud = *msg;
    req_msg.poseKnown = pose_initialized;
    pubRequestLocalization.publish(req_msg);
}

void OnSubscribeLocalizationResult(const mobinn_nav_msgs::AnswerLocalization::ConstPtr & msg) {
    std::cout<<"Received Answer!"<<std::endl;
    if(!pose_initialized) {
        std::cout<<"Pose Initialized!"<<std::endl;
        pose_initialized = true;
    }
    currentPose = OdometryToEigen(msg->candidatePose);

    static tf::TransformBroadcaster tfbr;

    tf::Transform tfT;
    tf::poseMsgToTF(msg->candidatePose.pose.pose, tfT);
    tfbr.sendTransform(tf::StampedTransform(tfT, msg->header.stamp, "map", "/lidar_front/os_sensor"));

}


int main(int argc, char ** argv) {
    ros::init(argc, argv, "localization_test");
    ros::NodeHandle nh;

    currentPose = Eigen::Matrix4f::Identity(4, 4);
 
    pose_initialized = false;
    pubRequestLocalization = nh.advertise<mobinn_nav_msgs::RequestLocalization>("/request_localization", 1);
    ros::Subscriber subAnswerLocalization = nh.subscribe<mobinn_nav_msgs::AnswerLocalization>("/answer_localization", 1, OnSubscribeLocalizationResult);
    ros::Subscriber subPointCloud = nh.subscribe<sensor_msgs::PointCloud2>("/lidar_front/os_cloud_node/points", 1, OnSubscribeLidarPointCloud);

    ros::spin();
}