#ifndef YGZ_TRACKER_H
#define YGZ_TRACKER_H

#include "common/Settings.h"
#include "common/NumTypes.h"
#include "common/IMUData.h"
#include "common/Camera.h"
#include "common/Frame.h"
#include "cv/ORBMatcher.h"

#include <deque>

namespace ygz {

    // forward declare
    class BackendSlidingWindowG2O;

    class System;

    class ORBExtractor;

    class Viewer;

    class TrackerLK {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        TrackerLK(const string &settingFile);

        TrackerLK();

        virtual ~TrackerLK() {}

        SE3d InsertStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp, const VecIMU &vimu);

        enum eTrackingState {
            SYSTEM_NOT_READY = -1,
            NO_IMAGES_YET = 0,
            NOT_INITIALIZED = 1,
            OK = 2,
            WEAK = 3
        };

        eTrackingState GetState() {
            unique_lock<mutex> lock(mMutexState);
            return mState;
        }

        void Reset();

        IMUPreIntegration GetIMUFromLastKF();

        void SetBackEnd(shared_ptr<BackendSlidingWindowG2O> backend) {
            mpBackEnd = backend;
        }

        Vector3d g() const { return mgWorld; }

        void SetGravity(const Vector3d &g) { mgWorld = g; }

        void SetViewer(shared_ptr<Viewer> viewer) {
            mpViewer = viewer;
        }

        void SetCamera(shared_ptr<CameraParam> camera) {
            mpCam = camera;
        }

        void SetPureVisionMode(bool pureVision = true) {
            mbVisionOnlyMode = true;
        }

    protected:

        void Track();

        bool TrackLastFrame(bool usePoseInfo = false);

        bool TrackLocalMap(int &inliers);

        // Create stereo matched map points 
        void CreateStereoMapPoints();

        // Clean the old features
        void CleanOldFeatures();

        SE3d mSpeed;       // speed, used to predict the currrent pose

        bool NeedNewKeyFrame(const int &trackinliers);

        void InsertKeyFrame();

        void PredictCurrentPose();

        int OptimizeCurrentPose();

        int OptimizeCurrentPoseWithIMU();

        int OptimizeCurrentPoseWithoutIMU();

        void UpdateLastFrame();

        bool StereoInitialization();

        bool IMUInitialization();

        Vector3d IMUInitEstBg(const std::deque<shared_ptr<Frame>> &vpKFs);

    protected:

        mutex mMutexState;
        eTrackingState mState = SYSTEM_NOT_READY;

        VecIMU mvIMUSinceLastKF;

        shared_ptr<Frame> mpCurrentFrame = nullptr;
        shared_ptr<Frame> mpLastFrame = nullptr;
        shared_ptr<Frame> mpLastKeyFrame = nullptr;
        shared_ptr<CameraParam> mpCam = nullptr;

        // System
        shared_ptr<System> mpSystem = nullptr;

        // BackEnd
        shared_ptr<BackendSlidingWindowG2O> mpBackEnd = nullptr;

        Vector3d mgWorld = Vector3d(0, 0, setting::gravity);

        // ORB Extractor
        shared_ptr<ORBExtractor> mpExtractor = nullptr;

        // ORB Matcher
        shared_ptr<ORBMatcher> mpMatcher = nullptr;

        // Viewer
        shared_ptr<Viewer> mpViewer = nullptr;

        int mTrackInliersCnt = 0;

    public:

        bool mbVisionOnlyMode = false;

    };

}

#endif
