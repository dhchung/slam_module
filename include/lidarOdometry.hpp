#include "lidarFactor.hpp"
#include <gtsam/geometry/Pose3.h>
#include "tic_toc.h"
#include <pcl/common/transforms.h>

constexpr double DISTANCE_SQ_THRESHOLD = 0.5;

class LiDAROdometry{
public:
    LiDAROdometry();
    ~LiDAROdometry();

    gtsam::Pose3 getRelativePose(pcl::PointCloud<pcl::PointXYZI>::Ptr lastCorner,
                                 pcl::PointCloud<pcl::PointXYZI>::Ptr lastSurface,
                                 pcl::PointCloud<pcl::PointXYZI>::Ptr currentCorner,
                                 pcl::PointCloud<pcl::PointXYZI>::Ptr currentSurface,
                                 gtsam::Pose3 & lastPose,
                                 gtsam::Pose3 & currPose,
                                 double & final_cost,
                                 bool & successful);


private:
    double para_q[4] = {0, 0, 0, 1};
    double para_t[3] = {0, 0, 0};
    Eigen::Map<Eigen::Quaterniond> * q_last_curr;
    Eigen::Map<Eigen::Vector3d> * t_last_curr;
    // pcl::KdTreeFLANN<pcl::PointXYZI> kdtreeCornerLast;
    // pcl::KdTreeFLANN<pcl::PointXYZI> kdtreeSurfLast;
};

LiDAROdometry::LiDAROdometry(){
    q_last_curr = new Eigen::Map<Eigen::Quaterniond>(para_q);
    t_last_curr = new Eigen::Map<Eigen::Vector3d>(para_t);
};
LiDAROdometry::~LiDAROdometry(){};


gtsam::Pose3 LiDAROdometry::getRelativePose(pcl::PointCloud<pcl::PointXYZI>::Ptr lastCorner,
                                            pcl::PointCloud<pcl::PointXYZI>::Ptr lastSurface,
                                            pcl::PointCloud<pcl::PointXYZI>::Ptr currentCorner,
                                            pcl::PointCloud<pcl::PointXYZI>::Ptr currentSurface,
                                            gtsam::Pose3 & lastPose,
                                            gtsam::Pose3 & currPose,
                                            double & final_cost,
                                            bool & successful) {

    if(lastCorner->empty() || lastSurface->empty()) {
        successful = false;
        return gtsam::Pose3::identity();
    }


    gtsam::Pose3 currToLastPose = lastPose.between(currPose);
    Eigen::Matrix4d T = currToLastPose.matrix();

    pcl::PointCloud<pcl::PointXYZI>::Ptr newCurrCorner(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr newCurrSurface(new pcl::PointCloud<pcl::PointXYZI>);
    
    pcl::transformPointCloud(*currentCorner, *newCurrCorner, T);
    pcl::transformPointCloud(*currentSurface, *newCurrSurface, T);

    pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeCornerLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());
    pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeSurfLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());

    kdtreeCornerLast->setInputCloud(lastCorner);
    kdtreeSurfLast->setInputCloud(lastSurface);


    // kdtreeCornerLast.setInputCloud(lastCorner->makeShared());
    // kdtreeSurfLast.setInputCloud(lastSurface->makeShared());

    int cornerPointsNum = currentCorner->size();
    int surfPointsNum = currentSurface->size();    

    Eigen::Matrix4d iter_T = Eigen::Matrix4d::Identity(4, 4);

    for(size_t opti_counter = 0; opti_counter <2; ++opti_counter) {
        int corner_correspondence = 0;
        int plane_correspondence = 0;
        ceres::LossFunction * loss_function = new ceres::HuberLoss(0.1);

        ceres::Manifold * q_parameterization = new ceres::EigenQuaternionManifold();
        ceres::Problem::Options problem_options;

        ceres::Problem problem(problem_options);
        problem.AddParameterBlock(para_q, 4, q_parameterization);
        problem.AddParameterBlock(para_t, 3);

        pcl::PointXYZI pointSel;
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;

        for(int i = 0; i < cornerPointsNum; ++i) {
            kdtreeCornerLast->nearestKSearch(newCurrCorner->points[i], 2, pointSearchInd, pointSearchSqDis);

            pcl::PointXYZI & curr_corner_point = newCurrCorner->points[i];

            int closestPointInd = -1, minPointInd2 = -1;

            if(pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD) {
                closestPointInd = pointSearchInd[0];
                if(pointSearchSqDis[1] < DISTANCE_SQ_THRESHOLD) {
                    minPointInd2 = pointSearchInd[1];
                }
            }
            if(minPointInd2 >= 0) {
                Eigen::Vector3d curr_point(curr_corner_point.x,
                                            curr_corner_point.y,
                                            curr_corner_point.z);
                Eigen::Vector3d last_point_a(lastCorner->points[closestPointInd].x,
                                             lastCorner->points[closestPointInd].y,
                                             lastCorner->points[closestPointInd].z);
                Eigen::Vector3d last_point_b(lastCorner->points[minPointInd2].x,
                                             lastCorner->points[minPointInd2].y,
                                             lastCorner->points[minPointInd2].z);

                double s = 1.0;
                ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, last_point_a, last_point_b, s);
                problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                corner_correspondence++;
            }


        }

        for(int i = 0; i < surfPointsNum; ++i) {
            // kdtreeSurfLast.nearestKSearch(newCurrSurface->points[i], 3, pointSearchInd, pointSearchSqDis);
            kdtreeSurfLast->nearestKSearch(newCurrSurface->points[i], 3, pointSearchInd, pointSearchSqDis);

            pcl::PointXYZI & curr_surf_point = newCurrSurface->points[i];

            int closestPointInd = -1, minPointInd2 = -1, minPointInd3 = -1;

            if(pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD) {
                closestPointInd = pointSearchInd[0];
                if(pointSearchSqDis[1] < DISTANCE_SQ_THRESHOLD) {
                    minPointInd2 = pointSearchInd[1];
                    
                    if(pointSearchSqDis[2] < DISTANCE_SQ_THRESHOLD) {
                        minPointInd3 = pointSearchInd[2];
                    }
                }                
            }
            if (minPointInd2 >= 0 && minPointInd3 >= 0) {

                Eigen::Vector3d curr_point(curr_surf_point.x,
                                            curr_surf_point.y,
                                            curr_surf_point.z);
                Eigen::Vector3d last_point_a(lastSurface->points[closestPointInd].x,
                                             lastSurface->points[closestPointInd].y,
                                             lastSurface->points[closestPointInd].z);
                Eigen::Vector3d last_point_b(lastSurface->points[minPointInd2].x,
                                             lastSurface->points[minPointInd2].y,
                                             lastSurface->points[minPointInd2].z);
                Eigen::Vector3d last_point_c(lastSurface->points[minPointInd3].x,
                                             lastSurface->points[minPointInd3].y,
                                             lastSurface->points[minPointInd3].z);

                double s = 1.0;
                ceres::CostFunction *cost_function = LidarPlaneFactor::Create(curr_point, last_point_a, last_point_b, last_point_c, s);
                problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                plane_correspondence++;
            }

        }

        if ((corner_correspondence + plane_correspondence) < 10)
        {
            printf("less correspondence! *************************************************\n");
            successful = false;
            return gtsam::Pose3::identity();
        }
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.max_num_iterations = 4;
        options.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        final_cost = summary.final_cost;

        Eigen::Matrix4d relT = Eigen::Matrix4d::Identity(4, 4);
        relT.block(0, 0, 3, 3) = q_last_curr->matrix();
        relT.block(0, 3, 3, 1) = *t_last_curr;
        
        iter_T = relT * iter_T;

        pcl::transformPointCloud(*newCurrCorner, *newCurrCorner, iter_T);
        pcl::transformPointCloud(*newCurrSurface, *newCurrSurface, iter_T);
    }

    Eigen::Matrix3d rotMat = iter_T.block(0, 0, 3, 3);
    gtsam::Rot3 RotMat(rotMat);
    gtsam::Point3 TransVec = gtsam::Point3(iter_T.block(0, 3, 3, 1));
    gtsam::Pose3 rel_pose_pre(RotMat, TransVec);
    successful = true;

    return rel_pose_pre * currToLastPose;

}