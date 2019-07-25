#include "system/System.h"
#include "cv/TrackerLK.h"
#include "backend/BackendSlidingWindowG2O.h"

namespace ygz {

    System::System(const string &configPath) {

        // Read rectification parameters
        cv::FileStorage fsSettings(configPath, cv::FileStorage::READ);

        if (!fsSettings.isOpened()) {
            LOG(FATAL) << "ERROR: Wrong path to settings" << endl;
            return;
        }

        int rows_l = fsSettings["Camera.height"];
        int cols_l = fsSettings["Camera.width"];

        setting::imageHeight = rows_l;
        setting::imageWidth = cols_l;

        // Create camera object
        setting::initSettings();
        float fx = fsSettings["Camera.fx"];
        float fy = fsSettings["Camera.fy"];
        float cx = fsSettings["Camera.cx"];
        float cy = fsSettings["Camera.cy"];
        float bf = fsSettings["Camera.bf"];

        shared_ptr<CameraParam> camera(new CameraParam(fx, fy, cx, cy, bf));

        // create a tracker
        mpTracker = shared_ptr<TrackerLK>(new TrackerLK(configPath));
        mpTracker->SetCamera(camera);

        bool pureVisionMode = string(fsSettings["PureVisionMode"]) == "true";
        if (pureVisionMode) {
            LOG(INFO) << "Running in pure stereo vision mode!" << endl;
            mpTracker->SetPureVisionMode(true);
        }

        // create a backend
        mpBackend = shared_ptr<BackendSlidingWindowG2O>(new BackendSlidingWindowG2O(mpTracker));
        mpTracker->SetBackEnd(mpBackend);

        // create a viewer
        bool useViewer = string(fsSettings["UseViewer"]) == "true";
        if (useViewer) {
            mpViewer = shared_ptr<Viewer>(new Viewer());
            mpTracker->SetViewer(mpViewer);
        }

        LOG(INFO) << "YGZ system all ready, waiting for images ..." << endl;
    }

    System::~System() {
        setting::destroySettings();
    }

    SE3d System::AddStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectright, const double &timestamp) {
        return mpTracker->InsertStereo(imRectLeft, imRectright, timestamp);
    }

    void System::Shutdown() {
        LOG(INFO) << "System shutdown" << endl;
        mpBackend->Shutdown();
        if (mpViewer) {
            LOG(INFO) << "Please close the GUI to shutdown all the system" << endl;
            mpViewer->WaitToFinish();
        }
    }

}
