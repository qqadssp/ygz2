#ifndef YGZ_TRACKER_LK_H
#define YGZ_TRACKER_LK_H

#include "common/Settings.h"
#include "common/NumTypes.h"
#include "common/Camera.h"
#include "common/Frame.h"
#include "cv/ORBMatcher.h"

#include <deque>
/**
 * Tracked implemented by LK flow (like VINS)
 */

namespace ygz {

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

        SE3d InsertStereo(
                const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp);

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

        void SetBackEnd(shared_ptr<BackendSlidingWindowG2O> backend) {
            mpBackEnd = backend;
        }

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

//        bool TrackReferenceKF(bool usePoseInfomation = true);

        bool TrackLocalMap(int &inliers);

        bool NeedNewKeyFrame(const int &trackinliers);

        void InsertKeyFrame();

        int OptimizeCurrentPose();

        int OptimizeCurrentPoseWithoutIMU();

        void UpdateLastFrame();

        bool StereoInitialization();

    protected:

        void CreateStereoMapPoints();

        void CleanOldFeatures();

        SE3d mSpeed;       // speed, used to predict the currrent pose

        mutex mMutexState;
        eTrackingState mState = SYSTEM_NOT_READY;

        shared_ptr<Frame> mpCurrentFrame = nullptr;  // 当前帧
        shared_ptr<Frame> mpLastFrame = nullptr;     // 上一个帧
        shared_ptr<Frame> mpLastKeyFrame = nullptr;  // 上一个关键帧
        shared_ptr<CameraParam> mpCam = nullptr;     // 相机内参

        shared_ptr<System> mpSystem = nullptr;

        shared_ptr<BackendSlidingWindowG2O> mpBackEnd = nullptr;

        shared_ptr<ORBExtractor> mpExtractor = nullptr;

        shared_ptr<ORBMatcher> mpMatcher = nullptr;

        shared_ptr<Viewer> mpViewer = nullptr;

        int mTrackInliersCnt = 0;

    public:
        bool mbVisionOnlyMode = false;  // 仅视觉模式？
    };
}

#endif
