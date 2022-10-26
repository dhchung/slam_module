#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/publisher.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <mobinn_nav_msgs/FeaturePointsVectors.h>
#include <mobinn_nav_msgs/FeaturePoints.h>

#include <mobinn_nav_msgs/AnswerLocalization.h>
#include <mobinn_nav_msgs/RequestLocalization.h>

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>
#include <signal.h>

extern const size_t px_offset_2048[64] = {37, 24, 12, 0, 37, 24, 12, 0, 37, 24, 12, 0, 37, 25, 12, 1, 36, 25, 13, 1, 37, 25, 13, 1, 37, 25, 13, 1, 37, 25, 13, 1, 37, 25, 13, 1, 37, 25, 13, 1, 37, 25, 13, 1, 37, 25, 13, 2, 38, 26, 14, 2, 38, 26, 14, 2, 38, 26, 14, 1, 38, 26, 14, 1};

#define LIDAR_HORIZONTAL_RESOLUTION 2048
#define LIDAR_CHANNEL_NO 64

struct EIGEN_ALIGN16 LabeledPoint {
    PCL_ADD_POINT4D;
    float intensity;
    uint32_t t;
    uint16_t reflectivity;
    uint8_t ring;
    uint16_t ambient;
    uint32_t range;

    float distance = 0.0;
    int is_valid = 1;
    int label = -1;
    int is_segmented = 0;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

POINT_CLOUD_REGISTER_POINT_STRUCT(LabeledPoint,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (std::uint32_t, t, t)
    (std::uint16_t, reflectivity, reflectivity)
    (std::uint8_t, ring, ring)
    (std::uint16_t, ambient, ambient)
    (std::uint32_t, range, range)
    (double, distance, distance)
    (int, is_valid, is_valid)
    (int, label, label)
    (int, is_segmented, is_segmented)
)

struct PointNode{
    pcl::PointCloud<pcl::PointXYZI>::Ptr totalPoints;
    pcl::PointCloud<pcl::PointXYZI>::Ptr totalPointsDown;
    pcl::PointCloud<pcl::PointXYZI>::Ptr cornerPoints;
    pcl::PointCloud<pcl::PointXYZI>::Ptr surfacePoints;
    gtsam::Pose3 pose;
    double time;
    int node_no;

    void InsertCornerPoints(pcl::PointCloud<pcl::PointXYZI> cornerPoints_) {
        cornerPoints = cornerPoints_.makeShared();
    }
    void InsertSurfacePoints(pcl::PointCloud<pcl::PointXYZI> surfacePoints_) {
        surfacePoints = surfacePoints_.makeShared();
    }

    PointNode() {
        totalPoints.reset(new pcl::PointCloud<pcl::PointXYZI>());
        totalPointsDown.reset(new pcl::PointCloud<pcl::PointXYZI>());
        cornerPoints.reset(new pcl::PointCloud<pcl::PointXYZI>());
        surfacePoints.reset(new pcl::PointCloud<pcl::PointXYZI>());
    }
};

Eigen::Matrix4f OdometryToEigen(nav_msgs::Odometry odom_msg) {

    Eigen::Matrix4f T = Eigen::Matrix4f::Identity(4, 4);
    Eigen::Quaternionf q(odom_msg.pose.pose.orientation.w,
                         odom_msg.pose.pose.orientation.x,
                         odom_msg.pose.pose.orientation.y,
                         odom_msg.pose.pose.orientation.z);

    T.block(0, 0, 3, 3) = q.matrix();
    T.block(0, 3, 3, 1) = Eigen::Vector3f(odom_msg.pose.pose.position.x,
                                          odom_msg.pose.pose.position.y,
                                          odom_msg.pose.pose.position.z);
    return T;

}

nav_msgs::Odometry EigenToOdometry(Eigen::Matrix4f & T) {

    nav_msgs::Odometry odom_msg;
    Eigen::Matrix3f rotMat = T.block(0, 0, 3, 3);
    Eigen::Quaternionf q(rotMat);
    odom_msg.pose.pose.orientation.w = q.w();
    odom_msg.pose.pose.orientation.x = q.x();
    odom_msg.pose.pose.orientation.y = q.y();
    odom_msg.pose.pose.orientation.z = q.z();

    odom_msg.pose.pose.position.x = T(0, 3);
    odom_msg.pose.pose.position.y = T(1, 3);
    odom_msg.pose.pose.position.z = T(2, 3);

    return odom_msg;
}

size_t CloudIndexFromCoords(size_t row, size_t col) {
    size_t vv = (col + LIDAR_HORIZONTAL_RESOLUTION - px_offset_2048[row]) % LIDAR_HORIZONTAL_RESOLUTION;
    size_t point_index = row * LIDAR_HORIZONTAL_RESOLUTION + vv;
    return point_index;
}