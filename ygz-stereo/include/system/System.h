#ifndef YGZ_SYSTEM_H
#define YGZ_SYSTEM_H

#include "common/NumTypes.h"
#include "cv/TrackerLK.h"
#include "backend/BackendSlidingWindowG2O.h"
#include "util/Viewer.h"

// the interface of full system

namespace ygz {

    class System {

    public:
        System(const string &configPath);

        ~System();

        SE3d AddStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectright, const double &timestamp);

        void Shutdown();

    public:

        shared_ptr<TrackerLK> mpTracker = nullptr;
        shared_ptr<BackendSlidingWindowG2O> mpBackend = nullptr;
        shared_ptr<Viewer> mpViewer = nullptr;

    };

}

#endif
