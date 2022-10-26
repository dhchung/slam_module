#include <iostream>
#include <thread>
#include <mutex>
#include <queue>
#include <unordered_map>

#include <ros/ros.h>
#include "tic_toc.h"
#include "lidarOdometry.hpp"
#include "utils.h"


using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using gtsam::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)

std::unordered_map<std::string, int> timeToNodeMap;
std::unordered_map<int, std::string> NodeToTime;

std::deque<sensor_msgs::Imu> imuQueOpt;
std::deque<sensor_msgs::Imu> imuQueImu;

std::mutex mtx;

bool doneFirstOpt;
double lastImuMsgTime;
double lastImuOptTime;

double lastKeyframeTime;

double lastPublishMapTime;

std::vector<PointNode> PointNodesForInitialization;

std::vector<PointNode> PointNodes;
std::vector<int> keyFrameToNode;

// IMUPreintegration

double imuAccNoise, imuGyrNoise, imuAccBiasN, imuGyrBiasN, imuGravity, imuRPYWeight;
gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise, priorVelNoise, priorBiasNoise, 
                                        correctionNoise, correctionNoise2, loopPoseNoise;
gtsam::Vector noiseModelBetweenBias;
gtsam::PreintegratedImuMeasurements *imuIntegratorOpt_;
gtsam::PreintegratedImuMeasurements *imuIntegratorImu_;

gtsam::Pose3 prevPose_;
gtsam::Vector3 prevVel_;
gtsam::NavState prevState_;
gtsam::imuBias::ConstantBias prevBias_;

gtsam::NavState prevStateOdom;
gtsam::imuBias::ConstantBias prevBiasOdom;

gtsam::ISAM2 optimizer;
gtsam::NonlinearFactorGraph factorGraph;
gtsam::Values graphValues;

ros::Publisher pubImuOdometry;

int key = 1;

pcl::PointCloud<pcl::PointXYZI>::Ptr mapPoints;

pcl::VoxelGrid<pcl::PointXYZI> voxelGridFilter;

LiDAROdometry LOdom;

ros::Publisher pubRequestLocalization;


std::thread publishMapThread;
bool threadRunning = false;

bool initialPositionFound = false;
int initialPointNode = 0;

double lastMapUpdateTime = -1.0;


void OnSubscribeLocalizationResult(const mobinn_nav_msgs::AnswerLocalization::ConstPtr & msg) {
    std::cout<<"Received Answer!"<<std::endl;
    if(!initialPositionFound) {
        if(!msg->forInitiliaztion) {
            std::cout<<"Something fucked up"<<std::endl;
            return;
        }

        std::cout<<"Pose Initialized!"<<std::endl;
        mtx.lock();
        initialPositionFound = true;
        int query_point_id = msg->nodeNumber;

        double subscribedTime = PointNodesForInitialization[query_point_id].time;
        lastMapUpdateTime = subscribedTime;

        if(imuQueOpt.empty()) {
            return;
        }
        //remove previous imu data
        while(!imuQueOpt.empty()) {
            if(imuQueOpt.front().header.stamp.toSec() < subscribedTime) {
                imuQueOpt.pop_front();
            } else {
                break;
            }
        }

        while (!imuQueImu.empty()) {
            if(imuQueImu.front().header.stamp.toSec() < subscribedTime) {
                imuQueImu.pop_front();
            } else {
                break;
            }
        }

        mtx.unlock();


        gtsam::Rot3 rotMat = gtsam::Rot3(msg->candidatePose.pose.pose.orientation.w,
                                         msg->candidatePose.pose.pose.orientation.x,
                                         msg->candidatePose.pose.pose.orientation.y,
                                         msg->candidatePose.pose.pose.orientation.z);
        gtsam::Vector3 translation = gtsam::Vector3(msg->candidatePose.pose.pose.position.x,
                                                    msg->candidatePose.pose.pose.position.y,
                                                    msg->candidatePose.pose.pose.position.z);



        prevPose_ = gtsam::Pose3(rotMat, translation);
        gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, priorPoseNoise);
        factorGraph.add(priorPose);        

        prevVel_ = gtsam::Vector3(0, 0, 0);
        gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, priorVelNoise);
        factorGraph.add(priorVel);

        prevBias_ = gtsam::imuBias::ConstantBias();
        gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, priorBiasNoise);
        factorGraph.add(priorBias);
        
        graphValues.insert(X(0), prevPose_);
        graphValues.insert(V(0), prevVel_);
        graphValues.insert(B(0), prevBias_);


        
        prevState_ = gtsam::NavState(prevPose_, prevVel_);
        prevStateOdom = prevState_;
        prevBiasOdom  = prevBias_;
        optimizer.update(factorGraph, graphValues);
        optimizer.update();
        factorGraph.resize(0);
        graphValues.clear();


        pcl::PointCloud<pcl::PointXYZI> pointCloud = *PointNodesForInitialization[query_point_id].totalPoints;
        pcl::PointCloud<pcl::PointXYZI> surfacePointCloud = *PointNodesForInitialization[query_point_id].surfacePoints;
        pcl::PointCloud<pcl::PointXYZI> cornerPointCloud = *PointNodesForInitialization[query_point_id].cornerPoints;

        PointNode ptnode;
        ptnode.totalPoints = pointCloud.makeShared();
        voxelGridFilter.setInputCloud(ptnode.totalPoints);
        voxelGridFilter.filter(*ptnode.totalPointsDown);
        
        ptnode.InsertCornerPoints(cornerPointCloud);
        ptnode.InsertSurfacePoints(surfacePointCloud);
        ptnode.pose = prevPose_;
        ptnode.time = PointNodesForInitialization[query_point_id].time;
        ptnode.node_no = 0;
        PointNodes.push_back(ptnode);
        keyFrameToNode.push_back(0);
        lastKeyframeTime = PointNodesForInitialization[query_point_id].time;

        imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_);
        imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
        key = 1;
        lastImuOptTime = PointNodesForInitialization[query_point_id].time;
        return ;

    } else {
        if(msg->forInitiliaztion) {
            std::cout<<"Shadows from the past..."<<std::endl;
            return;
        }

        std::cout<<"[UPDATED] by Map"<<std::endl;

        int query_point_id = msg->nodeNumber;

        gtsam::Rot3 rotMat = gtsam::Rot3(msg->candidatePose.pose.pose.orientation.w,
                                         msg->candidatePose.pose.pose.orientation.x,
                                         msg->candidatePose.pose.pose.orientation.y,
                                         msg->candidatePose.pose.pose.orientation.z);
        gtsam::Vector3 translation = gtsam::Vector3(msg->candidatePose.pose.pose.position.x,
                                                    msg->candidatePose.pose.pose.position.y,
                                                    msg->candidatePose.pose.pose.position.z);
        if(query_point_id > key) {
            std::cout<<"Query is bigger than Key"<<std::endl;
            return;
        }

        gtsam::Pose3 locPose = gtsam::Pose3(rotMat, translation);
        gtsam::PriorFactor<gtsam::Pose3> priorLocPose(gtsam::Symbol('x', query_point_id), locPose, loopPoseNoise);
            
            // X(query_point_id), locPose, priorPoseNoise);

        factorGraph.add(priorLocPose);
        // graphValues.insert(X(query_point_id), locPose);
        // mtx.lock();
        // optimizer.update(factorGraph);
        // optimizer.update();
        // mtx.unlock();

        // optimizer.update(factorGraph, graphValues);
        std::cout<<"Shit3"<<std::endl;
        // graphValues.clear();
        lastMapUpdateTime = msg->header.stamp.toSec();
    }
}




void GraphParamInitialization() {

    imuAccNoise = 3.9939570888238808e-03;
    imuGyrNoise = 1.5636343949698187e-03;
    imuAccBiasN = 6.4356659353532566e-05;
    imuGyrBiasN = 3.5640318696367613e-05;
    imuGravity = 9.80511;
    imuRPYWeight = 0.01;

    boost::shared_ptr<gtsam::PreintegrationParams> p = gtsam::PreintegrationParams::MakeSharedU(imuGravity);
    p->accelerometerCovariance  = gtsam::Matrix33::Identity(3,3) * pow(imuAccNoise, 2); // acc white noise in continuous
    p->gyroscopeCovariance      = gtsam::Matrix33::Identity(3,3) * pow(imuGyrNoise, 2); // gyro white noise in continuous
    p->integrationCovariance    = gtsam::Matrix33::Identity(3,3) * pow(1e-4, 2); // error committed in integrating position from velocities
    gtsam::imuBias::ConstantBias prior_imu_bias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished());; // assume zero initial bias


    loopPoseNoise  = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 100e-2, 100e-2, 100e-2, 100e-2, 100e-2, 100e-2).finished()); // rad,rad,rad,m, m, m
    priorPoseNoise  = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2).finished()); // rad,rad,rad,m, m, m
    priorVelNoise   = gtsam::noiseModel::Isotropic::Sigma(3, 1e4); // m/s
    priorBiasNoise  = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3); // 1e-2 ~ 1e-3 seems to be good
    correctionNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished()); // rad,rad,rad,m, m, m
    correctionNoise2 = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1, 1, 1, 1, 1, 1).finished()); // rad,rad,rad,m, m, m
    noiseModelBetweenBias = (gtsam::Vector(6) << imuAccBiasN, imuAccBiasN, imuAccBiasN, imuGyrBiasN, imuGyrBiasN, imuGyrBiasN).finished();
    
    imuIntegratorImu_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for IMU message thread
    imuIntegratorOpt_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for optimization           
}

void resetParams() {
    lastImuMsgTime = -1;
    lastKeyframeTime = -1;
    doneFirstOpt = false;
}

void resetGraph(){
    gtsam::ISAM2Params optParameters;
    optParameters.relinearizeThreshold = 0.1;
    optParameters.relinearizeSkip = 1;
    optimizer = gtsam::ISAM2(optParameters);

    gtsam::NonlinearFactorGraph newFactorGraph;
    factorGraph = newFactorGraph;
    gtsam::Values newValues;
    graphValues = newValues;
}



void OnSubscribeFeaturePoints(const mobinn_nav_msgs::FeaturePoints::ConstPtr & msg) {


    double subscribedTime = msg->header.stamp.toSec();

    TicToc ProcessTime;

    if(!initialPositionFound) {

        pcl::PointCloud<pcl::PointXYZI> pointCloud;
        pcl::fromROSMsg(msg->originalPoints, pointCloud);

        pcl::PointCloud<pcl::PointXYZI> surfacePointCloud;
        pcl::fromROSMsg(msg->surfacePoints, surfacePointCloud);

        pcl::PointCloud<pcl::PointXYZI> cornerPointCloud;
        pcl::fromROSMsg(msg->cornerPoints, cornerPointCloud);

        PointNode ptnode;
        ptnode.totalPoints = pointCloud.makeShared();
        voxelGridFilter.setInputCloud(ptnode.totalPoints);
        voxelGridFilter.filter(*ptnode.totalPointsDown);
        
        ptnode.InsertCornerPoints(cornerPointCloud);
        ptnode.InsertSurfacePoints(surfacePointCloud);
        ptnode.pose = prevPose_;
        ptnode.time = subscribedTime;
        ptnode.node_no = initialPointNode;
        ptnode.time = subscribedTime;
        PointNodesForInitialization.push_back(ptnode);


        mobinn_nav_msgs::RequestLocalization req_msg;
        req_msg.header.stamp = msg->header.stamp;
        req_msg.nodeNumber = initialPointNode;
        req_msg.poseKnown = false;
        req_msg.inputCloud = msg->originalPoints;
        req_msg.forInitiliaztion = true;

        pubRequestLocalization.publish(req_msg);
        ++initialPointNode;
        return;
    }

    std::deque<sensor_msgs::Imu> lastToPointTimeIMU;
    mtx.lock();
    if(imuQueOpt.empty()) {
        return;
    }
    if(PointNodes.empty()) {
        //remove previous imu data
        while(!imuQueOpt.empty()) {
            if(imuQueOpt.front().header.stamp.toSec() < subscribedTime) {
                imuQueOpt.pop_front();
            } else {
                break;
            }
        }
    } else {
        while(!imuQueOpt.empty()) {
            if(imuQueOpt.front().header.stamp.toSec() < subscribedTime) {
                lastToPointTimeIMU.push_back(imuQueOpt.front());
                imuQueOpt.pop_front();
            } else {
                break;
            }
        }
    }

    while (!imuQueImu.empty()) {
        if(imuQueImu.front().header.stamp.toSec() < subscribedTime) {
            imuQueImu.pop_front();
        } else {
            break;
        }
    }

    mtx.unlock();


    // if(PointNodes.empty()) {
        //System Initialization
        // prevPose_ = gtsam::Pose3::identity();   //To be replaced with initial localization result
        // gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, priorPoseNoise);
        // factorGraph.add(priorPose);
        // //Initial velocity
        // prevVel_ = gtsam::Vector3(0, 0, 0);
        // gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, priorVelNoise);
        // factorGraph.add(priorVel);

        // prevBias_ = gtsam::imuBias::ConstantBias();
        // gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, priorBiasNoise);
        // factorGraph.add(priorBias);

        // mtx.lock();
        // prevState_ = gtsam::NavState(prevPose_, prevVel_);
        // prevStateOdom = prevState_;
        // prevBiasOdom  = prevBias_;
        // mtx.unlock();

        //Add values
        // graphValues.insert(X(0), prevPose_);
        // graphValues.insert(V(0), prevVel_);
        // graphValues.insert(B(0), prevBias_);

        // optimizer.update(factorGraph, graphValues);
        // factorGraph.resize(0);
        // graphValues.clear();

    //     pcl::PointCloud<pcl::PointXYZI> pointCloud;
    //     pcl::fromROSMsg(msg->originalPoints, pointCloud);

    //     pcl::PointCloud<pcl::PointXYZI> surfacePointCloud;
    //     pcl::fromROSMsg(msg->surfacePoints, surfacePointCloud);

    //     pcl::PointCloud<pcl::PointXYZI> cornerPointCloud;
    //     pcl::fromROSMsg(msg->cornerPoints, cornerPointCloud);

    //     PointNode ptnode;
    //     ptnode.totalPoints = pointCloud.makeShared();
    //     voxelGridFilter.setInputCloud(ptnode.totalPoints);
    //     voxelGridFilter.filter(*ptnode.totalPointsDown);
        
    //     ptnode.InsertCornerPoints(cornerPointCloud);
    //     ptnode.InsertSurfacePoints(surfacePointCloud);
    //     ptnode.pose = prevPose_;
    //     ptnode.time = subscribedTime;
    //     ptnode.node_no = 0;
    //     ptnode.time = subscribedTime;
    //     PointNodes.push_back(ptnode);
    //     keyFrameToNode.push_back(0);
    //     lastKeyframeTime = subscribedTime;

    //     imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_);
    //     imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
    //     key = 1;
    //     lastImuOptTime = subscribedTime;
    //     return ;
    // }

    //Estimate current pose

    while(!lastToPointTimeIMU.empty()) {
        sensor_msgs::Imu *thisImu = &lastToPointTimeIMU.front();
        double imuTime = thisImu->header.stamp.toSec();
        double dt = (lastImuOptTime < 0) ? (1.0/100.0) : (imuTime - lastImuOptTime);
        imuIntegratorOpt_->integrateMeasurement(
                gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                gtsam::Vector3(thisImu->angular_velocity.x,    thisImu->angular_velocity.y,    thisImu->angular_velocity.z), dt);
        lastImuOptTime = imuTime;
        lastToPointTimeIMU.pop_front();
    }

    gtsam::NavState propState_ = imuIntegratorOpt_->predict(prevState_, prevBias_);
    gtsam::Pose3 predictedPose = propState_.pose();

    gtsam::Pose3 fromLastPose = prevPose_.between(predictedPose);
    double time_diff = subscribedTime - lastKeyframeTime;


    double distance_diff = std::sqrt(fromLastPose.x() * fromLastPose.x() +
                                     fromLastPose.y() * fromLastPose.y() +
                                     fromLastPose.z() * fromLastPose.z());

    gtsam::Quaternion q = fromLastPose.rotation().toQuaternion();
    double angle_diff = q.angularDistance(gtsam::Quaternion::Identity());

    double current_pitch = 0;

    Eigen::Matrix3d rotMat = predictedPose.matrix().block(0, 0, 3, 3);
    Eigen::Vector3d angles = rotMat.eulerAngles(1, 2, 3);
    // current_pitch = std::abs(angles(1));

    // if(time_diff > 1.0 || distance_diff > 0.5 || angle_diff > 10 * M_PI/180.0 || current_pitch < 5 * M_PI/180.0) {
    if(time_diff > 1.0 || distance_diff > 0.5 || angle_diff > 10 * M_PI/180.0) {

        //Search Nearby Keyframes

        int PointNodeSize = int(PointNodes.size());
        std::vector<int> candidate_indices;

        for(int i = 0; i < 5; ++i) {
            if(PointNodeSize -1 - i < 0) {
                break;
            }
            candidate_indices.push_back(PointNodeSize -1 - i);
        }

        // //Loop Closure
        // for(int i = 0; i < PointNodes.size(); ++i) {
            
        //     if(key - PointNodes[i].node_no < 30) {
        //         break;
        //     }
            
        //     if(candidate_indices.size() > 10) {
        //         break;
        //     }

        //     gtsam::Pose3 poseDiff = predictedPose.between(PointNodes[i].pose);
        //     double distDiff = std::sqrt(poseDiff.x() * poseDiff.x() +
        //                                 poseDiff.y() * poseDiff.y() +
        //                                 poseDiff.z() * poseDiff.z());

        //     if(distDiff < 5.0) {
        //         candidate_indices.push_back(i);
        //         std::cout<<"[Loop Candidate Pushed]"<<std::endl;
        //     }
        // }

        if(candidate_indices.empty()) {
            printf("No candidates\n");
            return;
        }

        //Convert Points
        pcl::PointCloud<pcl::PointXYZI> pointCloud;
        pcl::fromROSMsg(msg->originalPoints, pointCloud);

        pcl::PointCloud<pcl::PointXYZI> surfacePointCloud;
        pcl::fromROSMsg(msg->surfacePoints, surfacePointCloud);

        pcl::PointCloud<pcl::PointXYZI> cornerPointCloud;
        pcl::fromROSMsg(msg->cornerPoints, cornerPointCloud);

        PointNode ptnode;
        ptnode.totalPoints = pointCloud.makeShared();
        voxelGridFilter.setInputCloud(ptnode.totalPoints);
        voxelGridFilter.filter(*ptnode.totalPointsDown);

        ptnode.InsertCornerPoints(cornerPointCloud);
        ptnode.InsertSurfacePoints(surfacePointCloud);
        ptnode.time = subscribedTime;
        //Align to Last poses
        bool haveSuccessiveMatch = false;

        std::vector<gtsam::Pose3> candidate_poses(candidate_indices.size());
        std::vector<double> candidate_costs(candidate_indices.size());

        for(int i = 0; i < candidate_indices.size(); ++i) {
            double cost;
            bool successful = false;
            gtsam::Pose3 rel_pose =  LOdom.getRelativePose(PointNodes[candidate_indices[i]].cornerPoints,
                                     PointNodes[candidate_indices[i]].surfacePoints,
                                     ptnode.cornerPoints,
                                     ptnode.surfacePoints,
                                     PointNodes[candidate_indices[i]].pose,
                                     predictedPose,
                                     cost,
                                     successful);

            if(!successful) {
                std::cout<<"UNSUCCESSFUL MATCH"<<std::endl;
                continue;
            }

            candidate_poses[i] = rel_pose;
            candidate_costs[i] = cost;
            gtsam::BetweenFactor<gtsam::Pose3> pose_factor(X(PointNodes[candidate_indices[i]].node_no), X(key), rel_pose, correctionNoise);
            factorGraph.add(pose_factor);
            haveSuccessiveMatch = true;
        }

        if(haveSuccessiveMatch) {
            std::cout<<"Key: "<<key<<std::endl;
            doneFirstOpt = true;
            const gtsam::PreintegratedImuMeasurements& preint_imu = dynamic_cast<const gtsam::PreintegratedImuMeasurements&>(*imuIntegratorOpt_);
            gtsam::ImuFactor imu_factor(X(key - 1), V(key - 1), X(key), V(key), B(key - 1), preint_imu);


            factorGraph.add(imu_factor);
            factorGraph.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(B(key - 1), B(key), gtsam::imuBias::ConstantBias(),
                                gtsam::noiseModel::Diagonal::Sigmas(sqrt(imuIntegratorOpt_->deltaTij()) * noiseModelBetweenBias)));
            graphValues.insert(X(key), propState_.pose());
            graphValues.insert(V(key), propState_.v());
            graphValues.insert(B(key), prevBias_);

            optimizer.update(factorGraph, graphValues);
            optimizer.update();
            factorGraph.resize(0);
            graphValues.clear();
            

            gtsam::Values result = optimizer.calculateEstimate();
            prevPose_  = result.at<gtsam::Pose3>(X(key));
            prevVel_   = result.at<gtsam::Vector3>(V(key));
            prevState_ = gtsam::NavState(prevPose_, prevVel_);
            prevBias_  = result.at<gtsam::imuBias::ConstantBias>(B(key));
            imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
            prevStateOdom = prevState_;
            prevBiasOdom  = prevBias_;
            ptnode.pose = prevPose_;
            ptnode.node_no = key;


            pcl::PointCloud<pcl::PointXYZI>::Ptr transformedPoints(new pcl::PointCloud<pcl::PointXYZI>);

            for(int i = 0; i < PointNodes.size(); ++i) {
                PointNodes[i].pose = result.at<gtsam::Pose3>(X(PointNodes[i].node_no));
            }

            gtsam::Pose3 fromLastKeyFrame = prevPose_.between(PointNodes.back().pose);

            double distanceDiffFromLastKeyFrame = std::sqrt(fromLastKeyFrame.x() * fromLastKeyFrame.x() +
                                                            fromLastKeyFrame.y() * fromLastKeyFrame.y() +
                                                            fromLastKeyFrame.z() * fromLastKeyFrame.z());

            gtsam::Quaternion qLastKeyFrame = fromLastKeyFrame.rotation().toQuaternion();
            double angleDiffLastKeyFrame = qLastKeyFrame.angularDistance(gtsam::Quaternion::Identity());

            if(distanceDiffFromLastKeyFrame > 0.5 || angleDiffLastKeyFrame > 10 * M_PI/180.0) {
                PointNodes.push_back(ptnode);
                keyFrameToNode.push_back(key);
                //Request Map Localization

                // if(ptnode.time - lastMapUpdateTime > 10.0) {
                    mobinn_nav_msgs::RequestLocalization req_msg;
                    req_msg.header.stamp = msg->header.stamp;
                    req_msg.nodeNumber = key;
                    req_msg.poseKnown = true;
                    gtsam::Quaternion prevPoseQ = prevPose_.rotation().toQuaternion();
                    req_msg.currentPose.pose.pose.orientation.w = prevPoseQ.w();
                    req_msg.currentPose.pose.pose.orientation.x = prevPoseQ.x();
                    req_msg.currentPose.pose.pose.orientation.y = prevPoseQ.y();
                    req_msg.currentPose.pose.pose.orientation.z = prevPoseQ.z();
                    gtsam::Point3 prevPoseT = prevPose_.translation();
                    req_msg.currentPose.pose.pose.position.x = prevPoseT.x();
                    req_msg.currentPose.pose.pose.position.y = prevPoseT.y();
                    req_msg.currentPose.pose.pose.position.z = prevPoseT.z();
                    req_msg.inputCloud = msg->originalPoints;
                    req_msg.forInitiliaztion = false;

                    pubRequestLocalization.publish(req_msg);                
                    std::cout<<"Requested key: "<<req_msg.nodeNumber<<std::endl;
                // }


            }
            lastKeyframeTime = subscribedTime;

            if (!imuQueImu.empty())
            {
                imuIntegratorImu_->resetIntegrationAndSetBias(prevBiasOdom);

                double lastImuQT = subscribedTime;
                for (int i = 0; i < (int)imuQueImu.size(); ++i)
                {
                    sensor_msgs::Imu *thisImu = &imuQueImu[i];
                    double imuTime = thisImu->header.stamp.toSec();;
                    double dt = (lastImuQT < 0) ? (1.0 / 100.0) :(imuTime - lastImuQT);
                    if(dt <= 0) {
                        dt = 0.01;
                    }

                    imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                                                            gtsam::Vector3(thisImu->angular_velocity.x,    thisImu->angular_velocity.y,    thisImu->angular_velocity.z), dt);
                    lastImuQT = imuTime;
                }
            }

            ++key;
        }
    }
    printf("Shitman: %f\n", ProcessTime.toc());
}

void OnReceiveIMUMessage(const sensor_msgs::Imu::ConstPtr & msg) {
    std::lock_guard<std::mutex> lock(mtx);
    imuQueOpt.push_back(*msg);
    imuQueImu.push_back(*msg);

    if(!doneFirstOpt) {
        return;
    }

    double imuTime = msg->header.stamp.toSec();
    double dt = (lastImuMsgTime < 0) ? (1.0/100.0) : (imuTime - lastImuMsgTime);
    lastImuMsgTime = imuTime;
    imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z),
                                            gtsam::Vector3(msg->angular_velocity.x,    msg->angular_velocity.y,    msg->angular_velocity.z), dt);

    gtsam::NavState currentState = imuIntegratorImu_->predict(prevStateOdom, prevBiasOdom);
    nav_msgs::Odometry odometry;
    odometry.header.stamp = msg->header.stamp;
    odometry.header.frame_id = "map";
    odometry.child_frame_id = "lidar_front/os_sensor";
    gtsam::Pose3 imuPose = gtsam::Pose3(currentState.quaternion(), currentState.position());
    
    odometry.pose.pose.position.x = imuPose.translation().x();
    odometry.pose.pose.position.y = imuPose.translation().y();
    odometry.pose.pose.position.z = imuPose.translation().z();
    odometry.pose.pose.orientation.x = imuPose.rotation().toQuaternion().x();
    odometry.pose.pose.orientation.y = imuPose.rotation().toQuaternion().y();
    odometry.pose.pose.orientation.z = imuPose.rotation().toQuaternion().z();
    odometry.pose.pose.orientation.w = imuPose.rotation().toQuaternion().w();
    
    odometry.twist.twist.linear.x = currentState.velocity().x();
    odometry.twist.twist.linear.y = currentState.velocity().y();
    odometry.twist.twist.linear.z = currentState.velocity().z();
    odometry.twist.twist.angular.x = msg->angular_velocity.x + prevBiasOdom.gyroscope().x();
    odometry.twist.twist.angular.y = msg->angular_velocity.y + prevBiasOdom.gyroscope().y();
    odometry.twist.twist.angular.z = msg->angular_velocity.z + prevBiasOdom.gyroscope().z();
    pubImuOdometry.publish(odometry);

    static tf::TransformBroadcaster tfbr;
    tf::Transform tfT;

    tf::poseMsgToTF(odometry.pose.pose, tfT);

    tfbr.sendTransform(tf::StampedTransform(tfT, msg->header.stamp, "map", "/lidar_front/os_sensor"));
}


void roskillHandler(int sig) {
    if(PointNodes.size()==0) {
        // std::cout<<"No Data To Save"<<std::endl;
        // ros::waitForShutdown();
        ros::shutdown();
    } else {
        // std::cout<<"Saving poses Maps"<<std::endl;

        // std::string directory = "/home/dongha/";
        // std::string pose_dir = directory + "poses.txt";


        // FILE * pose_file = fopen(pose_dir.c_str(), "w");


        // for(int i = 0; i < PointNodes.size(); ++i) {
        //     char cloud_no[256];
        //     char pose[1000];
        //     sprintf(cloud_no, "%06d", i);


        //     sprintf(pose, "%06d\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\n",
        //                    i,
        //                    PointNodes[i].pose.x(),
        //                    PointNodes[i].pose.y(),
        //                    PointNodes[i].pose.z(),
        //                    PointNodes[i].pose.rotation().toQuaternion().x(),
        //                    PointNodes[i].pose.rotation().toQuaternion().y(),
        //                    PointNodes[i].pose.rotation().toQuaternion().z(),
        //                    PointNodes[i].pose.rotation().toQuaternion().w()
        //                    );
        //     fwrite(pose, 1, strlen(pose), pose_file);
        // }

        // fclose(pose_file);

        ros::shutdown();
    }
}


int main(int argc, char** argv) {
    ros::init(argc, argv, "optimization_module", ros::init_options::NoSigintHandler);
    
    ros::NodeHandle nh;
    signal(SIGINT, roskillHandler);

    resetGraph();
    GraphParamInitialization();

    ROS_INFO("Optimization Module");
    doneFirstOpt = false;
    lastImuMsgTime = -1;
    lastKeyframeTime = -1;
    lastImuOptTime = -1;
    lastPublishMapTime = -1.0;

    voxelGridFilter.setLeafSize(0.4f, 0.4f, 0.4f);

    mapPoints.reset(new pcl::PointCloud<pcl::PointXYZI>());
    initialPointNode = 0;

    ros::Subscriber subFeaturePoints = nh.subscribe<mobinn_nav_msgs::FeaturePoints>("/feature_clouds", 1, OnSubscribeFeaturePoints);
    ros::Subscriber subIMU = nh.subscribe<sensor_msgs::Imu>("/gx5/imu/data", 1, OnReceiveIMUMessage);

    pubImuOdometry   = nh.advertise<nav_msgs::Odometry>("/Odometry", 2000);

    pubRequestLocalization = nh.advertise<mobinn_nav_msgs::RequestLocalization>("/request_localization", 1);
    ros::Subscriber subAnswerLocalization = nh.subscribe<mobinn_nav_msgs::AnswerLocalization>("/answer_localization", 1, OnSubscribeLocalizationResult);


    ros::MultiThreadedSpinner spinner(4);
    spinner.spin();
}