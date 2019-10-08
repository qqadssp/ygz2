#ifndef YGZ_SYSTEM_H
#define YGZ_SYSTEM_H

#include "common/NumTypes.h"
#include "cv/TrackerLK.h"
#include "backend/BackendSlidingWindowG2O.h"
#include "util/Viewer.h"

namespace ygz {

    class System {

    public:
        System(const string &configPath);

        ~System();

        SE3d AddStereoIMU(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp,
                          const VecIMU &vimu);

        void Shutdown();

    public:

        shared_ptr<TrackerLK> mpTracker = nullptr;
        shared_ptr<BackendSlidingWindowG2O> mpBackend = nullptr;
        shared_ptr<Viewer> mpViewer = nullptr;

    };

}

#endif
