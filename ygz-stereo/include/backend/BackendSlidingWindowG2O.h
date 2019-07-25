#ifndef YGZ_BACKEND_SLIDING_WINDOW_G2O_H
#define YGZ_BACKEND_SLIDING_WINDOW_G2O_H

#include "common/NumTypes.h"
#include "common/Settings.h"

#include <deque>
#include <set>

namespace ygz {

    // forward declare
    struct Frame;
    struct MapPoint;

    class TrackerLK;

    class BackendSlidingWindowG2O {

    public:
        BackendSlidingWindowG2O(shared_ptr<TrackerLK> tracker) : mpTracker(tracker) {
            mtBackendMainLoop = thread(&BackendSlidingWindowG2O::MainLoop, this);
        }

        virtual ~BackendSlidingWindowG2O() {}

        void MainLoop();

        int InsertKeyFrame(shared_ptr<Frame> newKF);

        bool IsBusy();

        void Shutdown();

        std::set<shared_ptr<MapPoint>> GetLocalMap();

        std::deque<shared_ptr<Frame>> GetAllKF() {
            unique_lock<mutex> lock(mMutexKFs);
            return mpKFs;
        }

        void Reset();

        void CallLocalBA();

        void OptimizeCurrent(shared_ptr<Frame> current);

        bool Stop() {
            unique_lock<mutex> lock(mMutexStop);
            if (mbStopRequested && !mbNotStop) {
                mbStopped = true;
                LOG(INFO) << "Local Mapping STOP" << endl;
                return true;
            }
            return false;
        }

        void RequestStop() {
            unique_lock<mutex> lock(mMutexStop);
            mbStopRequested = true;
            unique_lock<mutex> lock2(mMutexNewKFs);
            mbAbortBA = true;
        }

        bool isStopped() {
            unique_lock<mutex> lock(mMutexStop);
            return mbStopped;
        }

        bool StopRequested() {
            unique_lock<mutex> lock(mMutexStop);
            return mbStopRequested;
        }

        void Release() {
            unique_lock<mutex> lock(mMutexStop);
            unique_lock<mutex> lock2(mMutexFinish);
            if (mbFinished)
                return;
            mbStopped = false;
            mbStopRequested = false;
            mpNewKFs.clear();
            mpKFs.clear();
            mpPoints.clear();
            mpCurrent = nullptr;
            LOG(INFO) << "Local Mapping RELEASE" << endl;
        }

        bool SetNotStop(bool flag) {
            unique_lock<mutex> lock(mMutexStop);
            if (flag && mbStopped)
                return false;
            mbNotStop = flag;
            return true;
        }

        void InterruptBA() {
            mbAbortBA = true;
        }

        void RequestReset() {
            {
                unique_lock<mutex> lock(mMutexReset);
                mbResetRequested = true;
                mbAbortBA = true;
            }

            while (1) {
                {
                    unique_lock<mutex> lock2(mMutexReset);
                    if (!mbResetRequested)
                        break;
                }
                usleep(3000);
            }
            mbAbortBA = false;
        }

        void ResetIfRequested() {
            unique_lock<mutex> lock(mMutexReset);
            if (mbResetRequested) {
                mpNewKFs.clear();
                mbResetRequested = false;
            }
        }

        void RequestFinish() {
            unique_lock<mutex> lock(mMutexFinish);
            mbFinishRequested = true;
        }

        bool CheckFinish() {
            unique_lock<mutex> lock(mMutexFinish);
            return mbFinishRequested;
        }

        void SetFinish() {
            mbAbortBA = true;
            unique_lock<mutex> lock(mMutexFinish);
            mbFinished = true;
            unique_lock<mutex> lock2(mMutexStop);
            mbStopped = true;

            mtBackendMainLoop.join();
        }

        bool isFinished() {
            unique_lock<mutex> lock(mMutexFinish);
            return mbFinished;
        }

        void SetAcceptKeyFrames(bool flag) {
            unique_lock<mutex> lock(mMutexAccept);
            mbAcceptKeyFrames = flag;
        }

        bool CheckNewKeyFrames() {
            unique_lock<mutex> lock(mMutexNewKFs);
            return !mpNewKFs.empty();
        }


    private: //

        void ProcessNewKeyFrame();

        void DeleteKF(int idx);

        int CleanMapPoint();

        void LocalBAXYZWithoutIMU(bool verbose = false);

    private:
        shared_ptr<TrackerLK> mpTracker = nullptr;
        shared_ptr<Frame> mpCurrent = nullptr;

        bool mbFirstCall = true;

        std::deque<shared_ptr<Frame>> mpKFs;
        std::deque<shared_ptr<Frame>> mpNewKFs;
        std::set<shared_ptr<MapPoint>> mpPoints;

        // mutex
        std::mutex mMutexReset;
        std::mutex mMutexFinish;
        std::mutex mMutexNewKFs;
        std::mutex mMutexStop;
        std::mutex mMutexAccept;

        std::mutex mMutexKFs;
        std::mutex mMutexPoints;


        // state variables
        bool mbAbortBA = false;
        bool mbStopped = false;
        bool mbStopRequested = false;
        bool mbNotStop = false;
        bool mbAcceptKeyFrames = false;
        bool mbFinishRequested = false;
        bool mbFinished = false;
        bool mbResetRequested = false;

        // main thread
        std::thread mtBackendMainLoop;

    };
}

#endif
