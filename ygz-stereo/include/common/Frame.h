#ifndef YGZ_FRAME_H
#define YGZ_FRAME_H

#include <opencv2/core/core.hpp>
#include <vector>

#include "common/NumTypes.h"
#include "common/Settings.h"
#include "common/Camera.h"

using cv::Mat;
using namespace std;

namespace ygz {

    struct Feature;
    struct MapPoint;

    struct Frame {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        Frame(const Frame &frame);

        Frame(
                const cv::Mat &left, const cv::Mat &right,
                const double &timestamp,
                shared_ptr<CameraParam> cam);

        Frame() {
            mnId = nNextId++;
        }

        virtual ~Frame();

        void ComputeImagePyramid();

        double ComputeSceneMedianDepth(const int &q);

        void SetPose(const SE3d &Twb);

        // 从TCW来实现SetPose
//        void SetPoseTCW(const SE3d &Tcw);

        // 获得位姿 Twb
        inline SE3d GetPose() {
            unique_lock<mutex> lock(mMutexPose);
            return SE3d(mRwb, mTwb);
        }

        // get Tcw
        SE3d GetTCW() {
            unique_lock<mutex> lock(mMutexPose);
            return SE3d(mRcw, mtcw);
        }

        bool isInFrustum(shared_ptr<MapPoint> pMP, float viewingCosLimit, int boarder = 20);

        bool PosInGrid(const shared_ptr<Feature> feature, int &posX, int &posY);

        vector<size_t> GetFeaturesInArea(
                const float &x, const float &y, const float &r, const int minLevel = -1,
                const int maxLevel = -1);

//        Vector3d UnprojectStereo(const int &i);

        void AssignFeaturesToGrid();

        bool IsKeyFrame() const { return mbIsKeyFrame; }

        bool SetThisAsKeyFrame();

        int TrackedMapPoints(const int &minObs);

        inline Vector3d World2Camera(const Vector3d &p_w, const SE3d &T_c_w) const {
            return T_c_w * p_w;
        }

        inline Vector2d World2Camera2(const Vector3d &p_w, const SE3d &T_c_w) const {
            Vector3d v = T_c_w * p_w;
            return Vector2d(v[0] / v[2], v[1] / v[2]);
        }

        inline Vector3d Camera2World(const Vector3d &p_c, const SE3d &T_c_w) const {
            return T_c_w.inverse() * p_c;
        }

        inline Vector2d Camera2Pixel(const Vector3d &p_c) const {
            return Vector2d(
                    mpCam->fx * p_c(0, 0) / p_c(2, 0) + mpCam->cx,
                    mpCam->fy * p_c(1, 0) / p_c(2, 0) + mpCam->cy
            );
        }

        inline Vector3d Pixel2Camera(const Vector2d &p_p, float depth = 1) const {
            return Vector3d(
                    (p_p(0, 0) - mpCam->cx) * depth / mpCam->fx,
                    (p_p(1, 0) - mpCam->cy) * depth / mpCam->fy,
                    depth
            );
        }

        inline Vector2d Pixel2Camera2(const Vector2f &p_p, float depth = 1) const {
            return Vector2d(
                    (p_p[0] - mpCam->cx) * depth / mpCam->fx,
                    (p_p[1] - mpCam->cy) * depth / mpCam->fy
            );
        }

        inline Vector3d Pixel2World(const Vector2d &p_p, const SE3d &TCW, float depth = 1) const {
            return Camera2World(Pixel2Camera(p_p, depth), TCW);
        }

        inline Vector2d World2Pixel(const Vector3d &p_w, const SE3d &TCW) const {
            return Camera2Pixel(World2Camera(p_w, TCW));
        }

        inline Matrix3d Rwb() {
            unique_lock<mutex> lock(mMutexPose);
            return mRwb.matrix();
        }

        inline Vector3d Twb() {
            unique_lock<mutex> lock(mMutexPose);
            return mTwb;
        }

        inline Matrix3d Rwc() {
            unique_lock<mutex> lock(mMutexPose);
            return mRwc;
        }

        inline Matrix3d Rcw() {
            unique_lock<mutex> lock(mMutexPose);
            return mRcw;
        }

        inline Vector3d Tcw() {
            unique_lock<mutex> lock(mMutexPose);
            return mtcw;
        }

        inline Vector3d Ow() {
            unique_lock<mutex> lock(mMutexPose);
            return mOw;
        }

        inline Vector6d PR() {
            unique_lock<mutex> lock(mMutexPose);
            Vector6d pr;
            pr.head<3>() = mTwb;
            pr.tail<3>() = mRwb.log();
            return pr;
        }

        // ---------------------------------------------------------------

        double mTimeStamp = 0;

        Mat mImLeft, mImRight;    // 左/右图像，显示用
        shared_ptr<CameraParam> mpCam = nullptr; // 相机参数

        std::mutex mMutexFeature;
        std::vector<shared_ptr<Feature>> mFeaturesLeft;
        std::vector<shared_ptr<Feature>> mFeaturesRight;

        std::vector<cv::Mat> mPyramidLeft;
        std::vector<cv::Mat> mPyramidRight;

        long unsigned int mnId = 0;             // id
        static long unsigned int nNextId;       // next id
        long unsigned int mnKFId = 0;           // keyframe id
        static long unsigned int nNextKFId;     // next keyframe id

        // pose, speed and bias
        std::mutex mMutexPose;            // lock the pose related variables
        SO3d mRwb;  // body rotation
        Vector3d mTwb = Vector3d(0, 0, 0); // body translation

        Matrix3d mRcw = Matrix3d::Identity();   ///< Rotation from world to camera
        Vector3d mtcw = Vector3d::Zero();       ///< Translation from world to camera
        Matrix3d mRwc = Matrix3d::Identity();   ///< Rotation from camera to world
        Vector3d mOw = Vector3d::Zero();        ///< =mtwc,Translation from camera to world

        weak_ptr<Frame> mpReferenceKF;    // the reference keyframe

        bool mbIsKeyFrame = false;

        std::vector<std::vector<std::size_t>> mGrid;

    };

}

#endif
