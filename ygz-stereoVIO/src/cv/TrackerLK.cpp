#include "cv/TrackerLK.h"
#include "cv/ORBExtractor.h"
#include "cv/ORBMatcher.h"
#include "cv/LKFlow.h"
#include "common/Feature.h"
#include "backend/BackendSlidingWindowG2O.h"
#include "common/MapPoint.h"
#include "util/Viewer.h"
#include "common/IMUPreIntegration.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "common/G2OTypes.h"
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/core/robust_kernel_impl.h>

namespace ygz {

    TrackerLK::TrackerLK(const string &settingFile) {

        cv::FileStorage fSettings(settingFile, cv::FileStorage::READ);
        if (fSettings.isOpened() == false) {
            cerr << "Setting file not found." << endl;
            return;
        }

        // create camera object
        float fx = fSettings["Camera.fx"];
        float fy = fSettings["Camera.fy"];
        float cx = fSettings["Camera.cx"];
        float cy = fSettings["Camera.cy"];
        float bf = fSettings["Camera.bf"];

        mpCam = shared_ptr<CameraParam>(new CameraParam(fx, fy, cx, cy, bf));
        mpExtractor = shared_ptr<ORBExtractor>(new ORBExtractor(ORBExtractor::FAST_SINGLE_LEVEL));
        mpMatcher = shared_ptr<ORBMatcher>(new ORBMatcher);
        mState = NO_IMAGES_YET;
    }

    TrackerLK::TrackerLK() {
        mState = NO_IMAGES_YET;
    }

    SE3d TrackerLK::InsertStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp,
                                 const VecIMU &vimu) {

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        if (setting::trackerUseHistBalance) {
            // perform a histogram equalization
            cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
            Mat imgLeftAfter, imgRightAfter;
            clahe->apply(imRectLeft, imgLeftAfter);
            clahe->apply(imRectRight, imgRightAfter);
            mpCurrentFrame = shared_ptr<Frame>(new Frame(imgLeftAfter, imgRightAfter, timestamp, mpCam, vimu));
        } else {
            mpCurrentFrame = shared_ptr<Frame>(new Frame(imRectLeft, imRectRight, timestamp, mpCam, vimu));
        }

        if (this->mbVisionOnlyMode == false)
            mvIMUSinceLastKF.insert(mvIMUSinceLastKF.end(), vimu.begin(), vimu.end());

        if (mpLastKeyFrame)
            mpCurrentFrame->mpReferenceKF = mpLastKeyFrame;

        // DO TRACKING !!
        LOG(INFO) << "\n\n********* Tracking frame " << mpCurrentFrame->mnId << " **********" << endl;
        Track();
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        double timeCost = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        LOG(INFO) << "Insert stereo cost time: " << timeCost << endl;

        if (mpViewer) {
            if (mpCurrentFrame->IsKeyFrame()) {
                mpViewer->AddFrame(mpCurrentFrame);
            } else {
                mpViewer->SetCurrentFrame(mpCurrentFrame);
            }
            mpViewer->SetTrackStatus(static_cast<int>(mState), mTrackInliersCnt);
        }

        LOG(INFO) << "Tracker returns, pose = \n" << mpCurrentFrame->GetPose().matrix() << endl;
        if (mbVisionOnlyMode == false)
            LOG(INFO) << "speed and bias = \n" << mpCurrentFrame->mSpeedAndBias.transpose() << endl;
        return mpCurrentFrame->GetPose();
    }

    void TrackerLK::Track() {

        mTrackInliersCnt = 0;

        if (mState == NO_IMAGES_YET) {

            // first we build the pyramid and compute the features
            LOG(INFO) << "Detecting features" << endl;
            mpExtractor->Detect(mpCurrentFrame, true, false);    // extract the keypoint in left eye
            LOG(INFO) << "Compute stereo matches" << endl;
            mpMatcher->ComputeStereoMatches(mpCurrentFrame, ORBMatcher::OPTIFLOW_CV);

            if (StereoInitialization() == false) {
                LOG(INFO) << "Stereo init failed." << endl;
                return;
            }

            InsertKeyFrame();

            mpLastFrame = mpCurrentFrame;
            mpLastKeyFrame = mpCurrentFrame;

            mState = NOT_INITIALIZED;

            return;

        } else if (mState == NOT_INITIALIZED) {

            bool bOK = false;
            mpCurrentFrame->SetPose(mpLastFrame->GetPose() * mSpeed);  // assume the speed is constant

            bOK = TrackLastFrame(false);

            if (bOK) {
                bOK = TrackLocalMap(mTrackInliersCnt);
            }

            if (bOK == false) {
                Reset();
                return;
            }

            CleanOldFeatures();

            if (NeedNewKeyFrame(mTrackInliersCnt)) {

                std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
                CreateStereoMapPoints();
                InsertKeyFrame();
                std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
                double timeCost = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
                LOG(INFO) << "Insert KF cost time: " << timeCost << endl;
            } else {

            }

            if (mbVisionOnlyMode == false) {
                if (IMUInitialization() == true) {
                    mState = OK;
                }
            }

            mSpeed = mpLastFrame->GetPose().inverse() * mpCurrentFrame->GetPose();
            mpLastFrame = mpCurrentFrame;
            return;

        } else if (mState == OK) {

            bool bOK = false;

            PredictCurrentPose();
            bOK = TrackLastFrame(false);

            if (bOK) {
                bOK = TrackLocalMap(mTrackInliersCnt);
            }

            if (bOK) {

                CleanOldFeatures();

                if (NeedNewKeyFrame(mTrackInliersCnt)) {
                    CreateStereoMapPoints();
                    InsertKeyFrame();
                }

                mpLastFrame = mpCurrentFrame;
                mSpeed = mpLastFrame->GetPose().inverse() * mpCurrentFrame->GetPose();

            } else {
                mState = WEAK;
                mpLastFrame = mpCurrentFrame;
                LOG(INFO) << "Set into WEAK mode" << endl;
                // in this case, don't save current as last frame
            }
        } else if (mState == WEAK) {
/*
            LOG(WARNING) << "Running WEAK mode" << endl;
            // use imu only to propagate the pose and try to initialize stereo vision
            PredictCurrentPose();

            // first let's try to track the last (may be bit longer) frame
            bool bOK = false;
            bOK = TrackLastFrame(false);
            bOK = TrackLocalMap(mTrackInliersCnt);
            if (bOK) {
                // we successfully tracked the last frame and local map, set state back to OK
                mState = OK;
                if (NeedNewKeyFrame(mTrackInliersCnt))
                    InsertKeyFrame();
                mpLastFrame = mpCurrentFrame;
            } else {
                // track failed, try use current frame to init the stereo vision
                // grab the features and do stereo matching
                mpExtractor->Detect(mpCurrentFrame, true, false);
                mpMatcher->ComputeStereoMatches(mpCurrentFrame, ORBMatcher::OPTIFLOW_CV);
                mpBackEnd->Reset();
                CreateStereoMapPoints();
                CleanOldFeatures();
                InsertKeyFrame();
                mState = OK;
                mpLastFrame = mpCurrentFrame;
                LOG(INFO) << "Recovered from WEAK into NOT_INITIALIZAED." << endl;
            }
*/
        }
    }

    bool TrackerLK::TrackLastFrame(bool usePoseInfo) {

        // Track the points in last frame and create new features in current
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        VecVector2f trackedPts, refPts;
        vector<size_t> idxRef;
        SE3d TCW = mpCurrentFrame->GetTCW();
        unique_lock<mutex> lock(mpLastFrame->mMutexFeature);

        for (size_t i = 0; i < mpLastFrame->mFeaturesLeft.size(); i++) {
            shared_ptr<Feature> feat = mpLastFrame->mFeaturesLeft[i];
            if (feat == nullptr) {
                continue;
            }

            idxRef.push_back(i);
            if (feat->mpPoint && feat->mpPoint->isBad() == false) {
                // associated with a good map point, predict the projected pixel in current
                refPts.push_back(feat->mPixel);
                // trackedPts.push_back(feat->mPixel);
                Vector2d px = mpCurrentFrame->World2Pixel(feat->mpPoint->GetWorldPos(), TCW);
                trackedPts.push_back(Vector2f(px[0], px[1]));
            } else {
                refPts.push_back(feat->mPixel);
                trackedPts.push_back(feat->mPixel);
            }
        }

        int cntMatches = LKFlowCV(mpLastFrame, mpCurrentFrame, refPts, trackedPts);

        int validMatches = 0;

        for (size_t i = 0; i < trackedPts.size(); i++) {
            if (trackedPts[i][0] < 0 || trackedPts[i][1] < 0)
                continue;

            // create a feature assigned with this map point
            shared_ptr<Feature> feat(new Feature);
            feat->mPixel = trackedPts[i];
            feat->mpPoint = mpLastFrame->mFeaturesLeft[idxRef[i]]->mpPoint;
            mpCurrentFrame->mFeaturesLeft.push_back(feat);
            if (feat->mpPoint->Status() == MapPoint::GOOD) {
                validMatches++;
            }

        }

        LOG(INFO) << "Current features: " << mpCurrentFrame->mFeaturesLeft.size() << endl;

        if (validMatches <= setting::minTrackLastFrameFeatures) {
            LOG(WARNING) << "Track last frame not enough valid matches: " << validMatches << ", I will abort this frame"
                         << endl;
            return false;
        }

        // do pose optimization
        LOG(INFO) << "LK tracked points: " << cntMatches << ", valid: " << validMatches << ", last frame features: "
                  << mpLastFrame->mFeaturesLeft.size() << endl;

        std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
        double timeCost = std::chrono::duration_cast<std::chrono::duration<double> >(t3 - t1).count();
        LOG(INFO) << "LK cost time: " << timeCost << ", pts: " << refPts.size() << endl;

        mTrackInliersCnt = OptimizeCurrentPose();
        // LOG(INFO) << "Track last frame inliers: " << inliers << endl;

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        timeCost = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        LOG(INFO) << "Track last frame cost time: " << timeCost << endl;

        if (mTrackInliersCnt >= setting::minTrackLastFrameFeatures) {
            return true;
        } else {
            return false;
        }
    }

    bool TrackerLK::TrackLocalMap(int &inliers) {

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        set<shared_ptr<MapPoint>> localmap = mpBackEnd->GetLocalMap();

        std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
        double timeCost = std::chrono::duration_cast<std::chrono::duration<double> >(t4 - t1).count();
        LOG(INFO) << "get local map points cost time: " << timeCost << endl;

        for (auto mp: localmap)
            if (mp)
                mp->mbTrackInView = false;

        for (auto feat: mpCurrentFrame->mFeaturesLeft)
            if (feat && feat->mpPoint && feat->mbOutlier == false)
                feat->mpPoint->mbTrackInView = true;

        set<shared_ptr<MapPoint> > mpsInView;
        for (auto &mp: localmap) {
            if (mp && mp->isBad() == false && mp->mbTrackInView == false && mpCurrentFrame->isInFrustum(mp, 0.5)) {
                mpsInView.insert(mp);
            }
        }

        if (mpsInView.empty())
            return inliers >= setting::minTrackLocalMapInliers;

        LOG(INFO) << "Call Search by direct projection" << endl;
        int cntMatches = mpMatcher->SearchByDirectProjection(mpCurrentFrame, mpsInView);
        LOG(INFO) << "Track local map matches: " << cntMatches << ", current features: "
                  << mpCurrentFrame->mFeaturesLeft.size() << endl;

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        timeCost = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        LOG(INFO) << "Search local map points cost time: " << timeCost << endl;

        // Optimize Pose
        int optinliers = OptimizeCurrentPose();

        // Update MapPoints Statistics
        inliers = 0;
        for (shared_ptr<Feature> feat : mpCurrentFrame->mFeaturesLeft) {
            if (feat->mpPoint) {
                if (!feat->mbOutlier) {
                    feat->mpPoint->IncreaseFound();
                    if (feat->mpPoint->Status() == MapPoint::GOOD)
                        inliers++;
                } else {
                }
            }
        }

        std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
        timeCost = std::chrono::duration_cast<std::chrono::duration<double> >(t3 - t1).count();
        LOG(INFO) << "Track local map cost time: " << timeCost << endl;

        LOG(INFO) << "Track Local map inliers: " << inliers << endl;

        // Decide if the tracking was succesful
        if (inliers < setting::minTrackLocalMapInliers)
            return false;
        else
            return true;
    }

    void TrackerLK::Reset() {

        LOG(INFO) << "Tracker is reseted" << endl;
        mpCurrentFrame->SetPose(mpLastFrame->GetPose());
        LOG(INFO) << "Current pose = \n" << mpCurrentFrame->GetPose().matrix() << endl;
        mpBackEnd->Reset();

        // test if we can just recover from stereo
        mpCurrentFrame->mFeaturesLeft.clear();

        LOG(INFO) << "Try init stereo" << endl;
        mpExtractor->Detect(mpCurrentFrame, true, false);    // extract the keypoint in left eye
        mpMatcher->ComputeStereoMatches(mpCurrentFrame, ORBMatcher::OPTIFLOW_BASED);

        if (StereoInitialization() == false) {
            LOG(INFO) << "Stereo init failed." << endl;
            mState = NOT_INITIALIZED;
            return;
        } else {
            LOG(INFO) << "Stereo init succeed." << endl;
            // set the current as a new kf and track it
            InsertKeyFrame();

            mpLastFrame = mpCurrentFrame;
            mpLastKeyFrame = mpCurrentFrame;
        }
        return;
    }

    void TrackerLK::CreateStereoMapPoints() {

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        mpCurrentFrame->AssignFeaturesToGrid();
        mpExtractor->Detect(mpCurrentFrame, true, false);    // extract new keypoints in left eye

        std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
        double timeCost2 = std::chrono::duration_cast<std::chrono::duration<double> >(t3 - t1).count();
        LOG(INFO) << "Detect new feature cost time: " << timeCost2 << endl;

        mpMatcher->ComputeStereoMatchesOptiFlow(mpCurrentFrame, true);

        float meanInvDepth = 1.0 / (mpCurrentFrame->ComputeSceneMedianDepth(2) + 1e-9);

        int cntMono = 0, cntStereo = 0, cntUpdate = 0;

        for (size_t i = 0; i < mpCurrentFrame->mFeaturesLeft.size(); i++) {
            shared_ptr<Feature> feat = mpCurrentFrame->mFeaturesLeft[i];
            if (feat == nullptr)
                continue;
            if (feat->mpPoint == nullptr) {

                if (feat->mfInvDepth > setting::minNewMapPointInvD && feat->mfInvDepth < setting::maxNewMapPointInvD) {
                    // we create a map point here
                    shared_ptr<MapPoint> mp(new MapPoint(mpCurrentFrame, i));
                    feat->mpPoint = mp;
                    cntStereo++;
                } else {
                    feat->mfInvDepth = meanInvDepth;
                    shared_ptr<MapPoint> mp(new MapPoint(mpCurrentFrame, i));
                    feat->mpPoint = mp;
                    mp->SetStatus(MapPoint::IMMATURE);
                    cntMono++;
                }
            } else {
                if (feat->mpPoint->Status() == MapPoint::IMMATURE) {
                    if (feat->mpPoint->mpRefKF.expired() ||
                        (feat->mpPoint->mpRefKF.expired() == false &&
                         feat->mpPoint->mpRefKF.lock()->mbIsKeyFrame == false))
                        feat->mpPoint->mpRefKF = mpCurrentFrame;    // change its reference

                    if (feat->mfInvDepth > setting::minNewMapPointInvD &&
                        feat->mfInvDepth < setting::maxNewMapPointInvD) {
                        feat->mpPoint->SetStatus(MapPoint::GOOD);
                        Vector3d ptFrame =
                                mpCurrentFrame->mpCam->Img2Cam(feat->mPixel) * (1.0 / double(feat->mfInvDepth));
                        feat->mpPoint->SetWorldPos(mpCurrentFrame->mRwc * ptFrame + mpCurrentFrame->mOw);
                        cntUpdate++;
                    } else {
                        Vector3d pw = feat->mpPoint->mWorldPos;
                        feat->mfInvDepth = 1.0 / (mpCurrentFrame->mRcw * pw + mpCurrentFrame->mtcw)[2];
                    }
                }
            }
        }

        LOG(INFO) << "new stereo: " << cntStereo << ", new Mono: " << cntMono << ", update immature: " << cntUpdate
                  << ", total features: " << mpCurrentFrame->mFeaturesLeft.size() << endl;

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        double timeCost = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        LOG(INFO) << "Create map point cost time: " << timeCost << endl;
    }

    void TrackerLK::CleanOldFeatures() {

        LOG(INFO) << "Cleaning old features" << endl;
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        int cntPassed = 0;
        int cntOutlier = 0;
        for (shared_ptr<Feature> &feat: mpCurrentFrame->mFeaturesLeft) {
            if (feat->mbOutlier) {
                feat = nullptr;
                cntOutlier++;
                continue;
            }

            shared_ptr<MapPoint> mp = feat->mpPoint;
            auto status = mp->Status();
            if (status == MapPoint::BAD) {
                feat->mpPoint = nullptr;
                feat = nullptr;
            } else if (status == MapPoint::GOOD) {
                if (mp->mpRefKF.expired() || mp->mpRefKF.lock()->IsKeyFrame() == false) {
                    mp->mpRefKF = mpCurrentFrame;
                    cntPassed++;
                }
            } else {
                if (mp->mpRefKF.expired() || mp->mpRefKF.lock()->IsKeyFrame() == false) {
                    mp->mpRefKF = mpCurrentFrame;
                    cntPassed++;
                }
            }
        }
        LOG(INFO) << "passed " << cntPassed << " features into current, delete "
                  << cntOutlier << " outliers." << endl;

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        double timeCost = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        LOG(INFO) << "Clean old features cost time: " << timeCost << endl;

    }

    void TrackerLK::PredictCurrentPose() {

        // step 0. get initial state from last KF
        mpCurrentFrame->SetPose(mpLastKeyFrame->GetPose());
        mpCurrentFrame->SetSpeedBias(mpLastKeyFrame->Speed(), mpLastKeyFrame->BiasG(), mpLastKeyFrame->BiasA());

        // step 1. calc IMUPreIntegration from last key frame to current frame
        IMUPreIntegration imuPreInt;

        Vector3d bg = mpLastKeyFrame->BiasG();
        Vector3d ba = mpLastKeyFrame->BiasA();

        // remember to consider the gap between the last KF and the first IMU
        const IMUData &imu = mvIMUSinceLastKF.front();
        double dt = std::max(0.0, imu.mfTimeStamp - mpLastKeyFrame->mTimeStamp);
        imuPreInt.update(imu.mfGyro - bg, imu.mfAcce - ba, dt);

        // integrate each imu
        for (size_t i = 0; i < mvIMUSinceLastKF.size(); i++) {
            const IMUData &imu = mvIMUSinceLastKF[i];
            double nextt = 0;

            if (i == (mvIMUSinceLastKF.size() - 1))
                nextt = mpCurrentFrame->mTimeStamp;    // last IMU, next is this KeyFrame
            else
                nextt = mvIMUSinceLastKF[i + 1].mfTimeStamp;  // regular condition, next is imu data

            // delta time
            double dt = std::max(0.0, nextt - imu.mfTimeStamp);
            // update pre-integrator
            imuPreInt.update(imu.mfGyro - bg, imu.mfAcce - ba, dt);
        }

        // step 2. predict current pose
        mpCurrentFrame->UpdatePoseFromPreintegration(imuPreInt, mgWorld);

        LOG(INFO) << "Predicted pose: \n" << mpCurrentFrame->GetPose().matrix() << "\nspeed and bias = "
                  << mpCurrentFrame->mSpeedAndBias.transpose() << endl;
    }

    bool TrackerLK::NeedNewKeyFrame(const int &trackinliers) {

        int nKFs = mpBackEnd->GetAllKF().size();

        // matches in reference KeyFrame
        int nRefMatches = mpLastKeyFrame->TrackedMapPoints(2);

        // Thresholds
        float thRefRatio = 0.75f;
        if (nKFs < 2)
            thRefRatio = 0.4f;

        // Condition 1c: tracking is weak
//        bool c1 = trackinliers < 40;
        bool c1 = trackinliers < 90;

        // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
        bool c2 = ((trackinliers < nRefMatches * thRefRatio) && trackinliers > 15);

        double timegap = setting::keyframeTimeGapTracking;

        if (mState != OK || mbVisionOnlyMode)
            timegap = setting::keyframeTimeGapInit;
        else
            timegap = setting::keyframeTimeGapTracking;

        bool cTimeGap = (mpCurrentFrame->mTimeStamp - mpLastKeyFrame->mTimeStamp) >= timegap;

        bool isBackendBusy = mpBackEnd->IsBusy();

//        if ((c1 || c2 || cTimeGap) && !isBackendBusy) {
//        if ((c1 || c2 ) && !isBackendBusy) {
        if ((c1) && !isBackendBusy) {
            return true;
        } else {
            return false;
        }

    }

    void TrackerLK::UpdateLastFrame() {
        // TODO: add update last frame PRVB from mpRefKF
    }

    void TrackerLK::InsertKeyFrame() {

        mpCurrentFrame->SetThisAsKeyFrame();

        mpCurrentFrame->mvIMUDataSinceLastFrame = mvIMUSinceLastKF;

        mvIMUSinceLastKF.clear();

        mpCurrentFrame->mpReferenceKF = mpLastKeyFrame;

        if (mbVisionOnlyMode == false && mpLastKeyFrame)
            mpCurrentFrame->ComputeIMUPreIntSinceLastFrame(mpLastKeyFrame, mpCurrentFrame->mIMUPreInt);

        mpLastKeyFrame = mpCurrentFrame;

        mpBackEnd->InsertKeyFrame(mpCurrentFrame);
        LOG(INFO) << "Insert keyframe done." << endl;
    }

    bool TrackerLK::StereoInitialization() {

        int cntValidFeat = 0;
        for (auto feat: mpCurrentFrame->mFeaturesLeft) {
            if (feat->mfInvDepth > 0)
                cntValidFeat++;
        }
        if (cntValidFeat < setting::minValidInitFeatures) {
            LOG(INFO) << "Valid feature is not enough! " << cntValidFeat << endl;
            return false;
        }

        LOG(INFO) << "stereo init created " << cntValidFeat << " features." << endl;
        double medianDepth = mpCurrentFrame->ComputeSceneMedianDepth(2);
        float medianInvDepth = 1.0 / medianDepth;

        for (size_t i = 0; i < mpCurrentFrame->mFeaturesLeft.size(); i++) {
            auto feat = mpCurrentFrame->mFeaturesLeft[i];
            if (feat->mfInvDepth > 0) {
                shared_ptr<MapPoint> mp(new MapPoint(mpCurrentFrame, i));
                feat->mpPoint = mp;
            } else {
                feat->mfInvDepth = medianInvDepth;
                shared_ptr<MapPoint> mp(new MapPoint(mpCurrentFrame, i));
                feat->mpPoint = mp;
                mp->SetStatus(MapPoint::IMMATURE);
            }
        }

        return true;
    }

    // Input: KeyFrame rotation Rwb
    Vector3d TrackerLK::IMUInitEstBg(const std::deque<shared_ptr<Frame>> &vpKFs) {

        // Setup optimizer
        typedef g2o::BlockSolverX Block;
        std::unique_ptr<Block::LinearSolverType> linearSolver (new g2o::LinearSolverEigen<Block::PoseMatrixType>());
        std::unique_ptr<Block> solver_ptr (new Block(std::move(linearSolver)));
        g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(std::move(solver_ptr));
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm(solver);

        // Add vertex of gyro bias, to optimizer graph
        ygz::VertexGyrBias *vBiasg = new ygz::VertexGyrBias();
        vBiasg->setEstimate(Eigen::Vector3d::Zero());
        vBiasg->setId(0);
        optimizer.addVertex(vBiasg);

        // Add unary edges for gyro bias vertex
        shared_ptr<Frame> pPrevKF0 = vpKFs.front();
        for (auto pKF : vpKFs) {
            // Ignore the first KF
            if (pKF == vpKFs.front())
                continue;

            shared_ptr<Frame> pPrevKF = pKF->mpReferenceKF.lock();

            const IMUPreIntegration &imupreint = pKF->GetIMUPreInt();
            EdgeGyrBias *eBiasg = new EdgeGyrBias();
            eBiasg->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));

            // measurement is not used in EdgeGyrBias
            eBiasg->dRbij = imupreint.getDeltaR();
            eBiasg->J_dR_bg = imupreint.getJRBiasg();
            eBiasg->Rwbi = pPrevKF->mRwb.matrix();
            eBiasg->Rwbj = pKF->mRwb.matrix();
            eBiasg->setInformation(imupreint.getCovPVPhi().bottomRightCorner(3, 3).inverse());
            optimizer.addEdge(eBiasg);

            pPrevKF0 = pKF;
        }

        // It's actualy a linear estimator, so 1 iteration is enough.
        optimizer.initializeOptimization();
        optimizer.optimize(1);

        // update bias G
        VertexGyrBias *vBgEst = static_cast<VertexGyrBias *>(optimizer.vertex(0));

        return vBgEst->estimate();
    }

    bool TrackerLK::IMUInitialization() {

        std::deque<shared_ptr<Frame>> vpKFs = mpBackEnd->GetAllKF();
        int N = vpKFs.size();
        if (N < setting::minInitKFs)
            return false;

        // Note 1.
        // Input : N (N>=4) KeyFrame/Frame poses of stereo vslam
        //         Assume IMU bias are identical for N KeyFrame/Frame
        // Compute :
        //          bg: gyroscope bias
        //          ba: accelerometer bias
        //          gv: gravity in vslam frame
        //          Vwbi: velocity of N KeyFrame/Frame
        // (Absolute scale is available for stereo vslam)

        // Note 2.
        // Convention by wangjing:
        // W: world coordinate frame (z-axis aligned with gravity, gW=[0;0;~9.8])
        // B: body coordinate frame (IMU)
        // V: camera frame of first camera (vslam's coordinate frame)
        // TWB/T : 6dof pose of frame, TWB = [RWB, PWB; 0, 1], XW = RWB*XW + PWB
        // RWB/R : 3dof rotation of frame's body(IMU)
        // PWB/P : 3dof translation of frame's body(IMU) in world
        // XW/XB : 3dof point coordinate in world/body

        // Step0. get all keyframes in map
        //        reset v/bg/ba to 0
        //        re-compute pre-integration

        Vector3d v3zero = Vector3d::Zero();
        for (auto pKF: vpKFs) {
            pKF->SetBiasG(v3zero);
            pKF->SetBiasA(v3zero);
        }
        for (int i = 1; i < N; i++) {
            vpKFs[i]->ComputeIMUPreInt();
        }

        // Step1. gyroscope bias estimation
        //        update bg and re-compute pre-integration
        Vector3d bgest = IMUInitEstBg(vpKFs);

        for (auto pKF: vpKFs) {
            pKF->SetBiasG(bgest);
        }
        for (int i = 1; i < N; i++) {
            vpKFs[i]->ComputeIMUPreInt();
        }

        // Step2. accelerometer bias and gravity estimation (gv = Rvw*gw)
        // let's first assume ba is given by prior and solve the gw
        // Step 2.1 gravity estimation

        // Solve C*x=D for x=[gw] (3+3)x1 vector
        // \see section IV in "Visual Inertial Monocular SLAM with Map Reuse"
        Vector3d baPrior = setting::biasAccePrior;

        MatrixXd C(3 * (N - 2), 3);
        C.setZero();

        VectorXd D(3 * (N - 2));
        D.setZero();

        Matrix3d I3 = Matrix3d::Identity();
        for (int i = 0; i < N - 2; i++) {

            shared_ptr<Frame> pKF1 = vpKFs[i];
            shared_ptr<Frame> pKF2 = vpKFs[i + 1];
            shared_ptr<Frame> pKF3 = vpKFs[i + 2];

            // Poses
            Matrix3d R1 = pKF1->mRwb.matrix();
            Matrix3d R2 = pKF2->mRwb.matrix();
            Vector3d p1 = pKF1->mTwb;
            Vector3d p2 = pKF2->mTwb;
            Vector3d p3 = pKF3->mTwb;

            // Delta time between frames
            double dt12 = pKF2->mIMUPreInt.getDeltaTime();
            double dt23 = pKF3->mIMUPreInt.getDeltaTime();
            // Pre-integrated measurements
            Vector3d dp12 = pKF2->mIMUPreInt.getDeltaP();
            Vector3d dv12 = pKF2->mIMUPreInt.getDeltaV();
            Vector3d dp23 = pKF3->mIMUPreInt.getDeltaP();

            Matrix3d Jpba12 = pKF2->mIMUPreInt.getJPBiasa();
            Matrix3d Jvba12 = pKF2->mIMUPreInt.getJVBiasa();
            Matrix3d Jpba23 = pKF3->mIMUPreInt.getJPBiasa();

            Matrix3d lambda = 0.5 * (dt12 * dt12 * dt23 + dt12 * dt23 * dt23) * I3;
            Vector3d phi = R2 * Jpba23 * baPrior * dt12 -
                           R1 * Jpba12 * baPrior * dt23 +
                           R1 * Jvba12 * baPrior * dt12 * dt23;
            Vector3d gamma = p3 * dt12 + p1 * dt23 + R1 * dp12 * dt23 - p2 * (dt12 + dt23)
                             - R2 * dp23 * dt12 - R1 * dv12 * dt12 * dt23;

            C.block<3, 3>(3 * i, 0) = lambda;
            D.segment<3>(3 * i) = gamma - phi;

        }

        // Use svd to compute C*x=D, x=[ba] 6x1 vector
        // Solve Ax = b where x is ba
        JacobiSVD<MatrixXd> svd2(C, ComputeThinU | ComputeThinV);
        VectorXd y = svd2.solve(D);
        Vector3d gpre = y.head(3);
        // normalize g
        Vector3d g0 = gpre / gpre.norm() * setting::gravity;

        // Step2.2
        // estimate the bias from g
        MatrixXd A(3 * (N - 2), 3);
        A.setZero();
        VectorXd B(3 * (N - 2));
        B.setZero();

        for (int i = 0; i < N - 2; i++) {

            shared_ptr<Frame> pKF1 = vpKFs[i];
            shared_ptr<Frame> pKF2 = vpKFs[i + 1];
            shared_ptr<Frame> pKF3 = vpKFs[i + 2];

            // Poses
            Matrix3d R1 = pKF1->mRwb.matrix();
            Matrix3d R2 = pKF2->mRwb.matrix();
            Vector3d p1 = pKF1->mTwb;
            Vector3d p2 = pKF2->mTwb;
            Vector3d p3 = pKF3->mTwb;

            // Delta time between frames
            double dt12 = pKF2->mIMUPreInt.getDeltaTime();
            double dt23 = pKF3->mIMUPreInt.getDeltaTime();
            // Pre-integrated measurements
            Vector3d dp12 = pKF2->mIMUPreInt.getDeltaP();
            Vector3d dv12 = pKF2->mIMUPreInt.getDeltaV();
            Vector3d dp23 = pKF3->mIMUPreInt.getDeltaP();

            Matrix3d Jpba12 = pKF2->mIMUPreInt.getJPBiasa();
            Matrix3d Jvba12 = pKF2->mIMUPreInt.getJVBiasa();
            Matrix3d Jpba23 = pKF3->mIMUPreInt.getJPBiasa();

            Vector3d lambda = 0.5 * (dt12 * dt12 * dt23 + dt12 * dt23 * dt23) * I3 * g0;
            Matrix3d phi = R2 * Jpba23 * dt12 -
                           R1 * Jpba12 * dt23 +
                           R1 * Jvba12 * dt12 * dt23;
            Vector3d gamma = p3 * dt12 + p1 * dt23 + R1 * dp12 * dt23 - p2 * (dt12 + dt23)
                             - R2 * dp23 * dt12 - R1 * dv12 * dt12 * dt23;

            A.block<3, 3>(3 * i, 0) = phi;
            B.segment<3>(3 * i) = gamma - lambda;
        }

        JacobiSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);
        VectorXd y2 = svd.solve(B);
        Vector3d baest = y2;

        // update ba and re-compute pre-integration
        for (auto pkf : vpKFs) {
            pkf->SetBiasA(baest);
        }
        for (int i = 1; i < N; i++) {
            vpKFs[i]->ComputeIMUPreInt();
        }

        // Step3. velocity estimation
        for (int i = 0; i < N; i++) {
            auto pKF = vpKFs[i];
            if (i != N - 1) {
                // not last KeyFrame, R1*dp12 = p2 - p1 -v1*dt12 - 0.5*gw*dt12*dt12
                //  ==>> v1 = 1/dt12 * (p2 - p1 - 0.5*gw*dt12*dt12 - R1*dp12)

                auto pKF2 = vpKFs[i + 1];
                const Vector3d p2 = pKF2->mTwb;
                const Vector3d p1 = pKF->mTwb;
                const Matrix3d R1 = pKF->mRwb.matrix();
                const double dt12 = pKF2->mIMUPreInt.getDeltaTime();
                const Vector3d dp12 = pKF2->mIMUPreInt.getDeltaP();

                Vector3d v1 = (p2 - p1 - 0.5 * g0 * dt12 * dt12 - R1 * dp12) / dt12;
                pKF->SetSpeed(v1);
            } else {
                // last KeyFrame, R0*dv01 = v1 - v0 - gw*dt01 ==>> v1 = v0 + gw*dt01 + R0*dv01
                auto pKF0 = vpKFs[i - 1];
                const Matrix3d R0 = pKF0->mRwb.matrix();
                const Vector3d v0 = pKF0->mSpeedAndBias.segment<3>(0);
                const double dt01 = pKF->mIMUPreInt.getDeltaTime();
                const Vector3d dv01 = pKF->mIMUPreInt.getDeltaV();

                Vector3d v1 = v0 + g0 * dt01 + R0 * dv01;
                pKF->SetSpeed(v1);
            }
        }

        double gprenorm = gpre.norm();
        // double baestdif = (baest0 - baest).norm();

        LOG(INFO) << "Estimated gravity before: " << gpre.transpose() << ", |gw| = " << gprenorm << endl;
        LOG(INFO) << "Estimated acc bias after: " << baest.transpose() << endl;
        LOG(INFO) << "Estimated gyr bias: " << bgest.transpose() << endl;

        bool initflag = false;
        if (gprenorm > 9.7 && gprenorm < 9.9 && /* baestdif < 0.2  && */
            baest.norm() < 1) {
            LOG(INFO) << "IMU init ok!" << endl;
            initflag = true;
        } else {
            // goodcnt = 0;
        }

        // align 'world frame' to gravity vector, making mgWorld = [0,0,9.8]
        if (initflag) {
            /*
            // compute Rvw
            Vector3d gw1(0, 0, 1);
            Vector3d gv1 = g0 / g0.norm();
            Vector3d gw1xgv1 = gw1.cross(gv1);
            Vector3d vhat = gw1xgv1 / gw1xgv1.norm();
            double theta = std::atan2(gw1xgv1.norm(), gw1.dot(gv1));
            Matrix3d Rvw = Sophus::SO3d::exp(vhat * theta).matrix();
            Matrix3d Rwv = Rvw.transpose();
            Sophus::SE3d Twv(Rwv, Vector3d::Zero());
            // 设置重力
            Vector3d gw = Rwv * g0;
            mgWorld = gw;

            // rotate pose/rotation/velocity to align with 'world' frame
            for (int i = 0; i < N; i++) {
                auto pKF = vpKFs[i];
                Sophus::SE3d Tvb = pKF->GetPose();
                Vector3d Vvb = pKF->Speed();
                // set pose/speed/biasg/biasa
                pKF->SetPose(Twv * Tvb);
                pKF->SetSpeed(Rwv * Vvb);
                pKF->SetBiasG(bgest);
                pKF->SetBiasA(baest);
            }

            if (mpCurrentFrame->IsKeyFrame() == false) {
                mpCurrentFrame->SetPose(Twv * mpCurrentFrame->GetPose());
                mpCurrentFrame->SetSpeed(Rwv * mpCurrentFrame->Speed());
                mpCurrentFrame->SetBiasG(bgest);
                mpCurrentFrame->SetBiasA(baest);
            }

            // re-compute pre-integration for KeyFrame (except for the first KeyFrame)
            for (int i = 1; i < N; i++) {
                vpKFs[i]->ComputeIMUPreInt();
            }

            // MapPoints
            auto vsMPs = mpBackEnd->GetLocalMap();
            for (auto mp : vsMPs) {
                Vector3d Pv = mp->GetWorldPos();
                Vector3d Pw = Rwv * Pv;
                mp->SetWorldPos(Pw);
            }
             */
            mgWorld = g0;
        }
        return initflag;
    }

    int TrackerLK::OptimizeCurrentPose() {

        assert(mState == OK || mState == WEAK || mState == NOT_INITIALIZED);

        if (mState == OK) {
            return OptimizeCurrentPoseWithIMU();
        } else {
            return OptimizeCurrentPoseWithoutIMU();
        }
    }

    int TrackerLK::OptimizeCurrentPoseWithIMU() {

        assert(mpCurrentFrame != nullptr);
        assert(mpLastKeyFrame != nullptr);
        assert(mvIMUSinceLastKF.size() != 0);

        LOG(INFO) << "Calling optimization with imu" << endl;

        IMUPreIntegration imupreint = GetIMUFromLastKF();

        // setup g2o
        typedef g2o::BlockSolverX Block;
        std::unique_ptr<Block::LinearSolverType> linearSolver (new g2o::LinearSolverEigen<Block::PoseMatrixType>());
        std::unique_ptr<Block> solver_ptr (new Block(std::move(linearSolver)));
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr));
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm(solver);

        // Set current frame vertex PVR/Bias
        // P+R
        VertexPR *vPR = new VertexPR();
        vPR->setEstimate(mpCurrentFrame->PR());
        vPR->setId(0);
        optimizer.addVertex(vPR);

        // Speed
        VertexSpeed *vSpeed = new VertexSpeed();
        vSpeed->setEstimate(mpCurrentFrame->Speed());
        vSpeed->setId(1);
        optimizer.addVertex(vSpeed);

        // Bg, Ba
        VertexGyrBias *vBiasG = new VertexGyrBias();
//        vBiasG->setEstimate(mpCurrentFrame->BiasG());
        vBiasG->setEstimate(Vector3d::Zero());
        vBiasG->setId(2);
        optimizer.addVertex(vBiasG);

        VertexAcceBias *vBiasA = new VertexAcceBias();
//        vBiasA->setEstimate(mpCurrentFrame->BiasA());
        vBiasA->setEstimate(Vector3d::Zero());
        vBiasA->setId(3);
        optimizer.addVertex(vBiasA);

        // P+R
        VertexPR *vPRL = new VertexPR();
        vPRL->setEstimate(mpLastKeyFrame->PR());
        vPRL->setId(4);
        vPRL->setFixed(true);
        optimizer.addVertex(vPRL);

        // Speed
        VertexSpeed *vSpeedL = new VertexSpeed();
        vSpeedL->setEstimate(mpLastKeyFrame->Speed());
        vSpeedL->setId(5);
        vSpeedL->setFixed(true);
        optimizer.addVertex(vSpeedL);

        // Bg
        VertexGyrBias *vBiasGL = new VertexGyrBias();
//        vBiasGL->setEstimate(mpLastKeyFrame->BiasG());
        vBiasGL->setEstimate(Vector3d::Zero());
        vBiasGL->setId(6);
        vBiasGL->setFixed(true);
        optimizer.addVertex(vBiasGL);

        VertexAcceBias *vBiasAL = new VertexAcceBias();
//        vBiasAL->setEstimate(mpLastKeyFrame->BiasA());
        vBiasAL->setEstimate(Vector3d::Zero());
        vBiasAL->setId(7);
        vBiasAL->setFixed(true);
        optimizer.addVertex(vBiasAL);

        // Edges
        // Set PVR edge between LastKF-Frame
        EdgePRV *ePRV = new EdgePRV(mgWorld);
        ePRV->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vPRL));
        ePRV->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vPR));
        ePRV->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vSpeedL));
        ePRV->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vSpeed));
        ePRV->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vBiasGL));
        ePRV->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vBiasAL));
        ePRV->setMeasurement(imupreint);

        // set Covariance
        Matrix9d CovPRV = imupreint.getCovPVPhi();

        CovPRV.col(3).swap(CovPRV.col(6));
        CovPRV.col(4).swap(CovPRV.col(7));
        CovPRV.col(5).swap(CovPRV.col(8));
        CovPRV.row(3).swap(CovPRV.row(6));
        CovPRV.row(4).swap(CovPRV.row(7));
        CovPRV.row(5).swap(CovPRV.row(8));

        // information matrix
        ePRV->setInformation(CovPRV.inverse());

        // robust kernel
        const float thHuberNavStatePVR = sqrt(21.666 * 1000); // this usually get a large error
        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
        ePRV->setRobustKernel(rk);
        rk->setDelta(thHuberNavStatePVR);
        optimizer.addEdge(ePRV);

        EdgeBiasG *eBG = new EdgeBiasG();
        eBG->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vBiasGL));
        eBG->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vBiasG));
        eBG->setMeasurement(Vector3d::Zero());
        Matrix3d infoBG = Matrix3d::Identity() * setting::gyrBiasRw2;
        eBG->setInformation(infoBG / imupreint.getDeltaTime());

        float thHuberNavStateBias = sqrt(16.812);
        g2o::RobustKernelHuber *rkb = new g2o::RobustKernelHuber;
        eBG->setRobustKernel(rkb);
        rkb->setDelta(thHuberNavStateBias);
        optimizer.addEdge(eBG);

        EdgeBiasA *eBA = new EdgeBiasA();
        eBA->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vBiasAL));
        eBA->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vBiasA));
        eBA->setMeasurement(Vector3d::Zero());
        Matrix3d infoBA = Matrix3d::Identity() * setting::accBiasRw2;
        eBA->setInformation(infoBA / imupreint.getDeltaTime());
        g2o::RobustKernelHuber *rkba = new g2o::RobustKernelHuber;
        eBA->setRobustKernel(rkba);
        rkba->setDelta(thHuberNavStateBias);
        optimizer.addEdge(eBA);

        // Set MapPoint vertices
        const int N = mpCurrentFrame->mFeaturesLeft.size();

        vector<EdgeProjectPoseOnly *> vpEdgesMono;
        vector<size_t> vnIndexEdgeMono;
        vpEdgesMono.reserve(N);
        vnIndexEdgeMono.reserve(N);

        const float delta = sqrt(5.991);

        int nInitialCorrespondences = 0;
        for (size_t i = 0; i < N; i++) {
            shared_ptr<Feature> feat = mpCurrentFrame->mFeaturesLeft[i];
            if (feat == nullptr)
                continue;
            shared_ptr<MapPoint> pMP = feat->mpPoint;

            if (pMP && pMP->Status() == MapPoint::GOOD) {
                // The points with only one observation are useless.
                if (pMP->Observations() < 1)
                    continue;

                nInitialCorrespondences++;
                feat->mbOutlier = false;

                EdgeProjectPoseOnly *eProj = new EdgeProjectPoseOnly(mpCam.get(), pMP->GetWorldPos());
                eProj->setVertex(0, vPR);
                eProj->setInformation(Matrix2d::Identity() * setting::invLevelSigma2[feat->mLevel]);
                eProj->setMeasurement(feat->mPixel.cast<double>());
                g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                eProj->setRobustKernel(rk);
                rk->setDelta(delta);
                optimizer.addEdge(eProj);

                vpEdgesMono.push_back(eProj);
                vnIndexEdgeMono.push_back(i);
            }
        }

        const float chi2th[4] = {5.991, 5.991, 5.991, 5.991};
        const int its[4] = {10, 10, 10, 10};

        int nBad = 0;
        for (size_t it = 0; it < 4; it++) {
            // Reset estimate for vertex
            vPR->setEstimate(mpCurrentFrame->PR());
            vSpeed->setEstimate(mpCurrentFrame->Speed());
            vBiasG->setEstimate(mpCurrentFrame->BiasG());
            vBiasA->setEstimate(mpCurrentFrame->BiasA());

            optimizer.initializeOptimization(0);
            optimizer.optimize(its[it]);

            nBad = 0;
            for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
                EdgeProjectPoseOnly *e = vpEdgesMono[i];
                const size_t idx = vnIndexEdgeMono[i];
                shared_ptr<Feature> feat = mpCurrentFrame->mFeaturesLeft[idx];

                if (feat->mbOutlier == true) {
                    e->computeError();
                }

                const float chi2 = e->chi2();

                if (chi2 > chi2th[it]) {
                    feat->mbOutlier = true;
                    e->setLevel(1);
                    nBad++;
                } else {
                    feat->mbOutlier = false;
                    e->setLevel(0);
                }

                if (it == 2) {
                    e->setRobustKernel(0);
                }
            }

            if (optimizer.edges().size() < 10)
                break;
        }

        // Recover optimized pose and return number of inliers
        int inliers = nInitialCorrespondences - nBad;
        if (inliers < setting::minPoseOptimizationInliers) {
            LOG(WARNING) << "inliers is small, pose may by unreliable: " << inliers << endl;
        }

        vPR->setFixed(true);
        vPRL->setFixed(true);
        vBiasAL->setFixed(false);
        vBiasGL->setFixed(false);
        vSpeedL->setFixed(false);

        optimizer.initializeOptimization();
        optimizer.optimize(10);

        // LOG(INFO) << "ePRV chi2 = " << ePRV->chi2() << endl;
        // LOG(INFO) << "eBG chi2 = " << eBG->chi2() << endl;
        // LOG(INFO) << "eBA chi2 = " << eBA->chi2() << endl;
        // LOG(INFO) << "speed = " << vSpeed->estimate().transpose()<<endl;
        // LOG(INFO) << "ba = " << vBiasG->estimate().transpose()<<endl;
        // LOG(INFO) << "bg = " << vBiasA->estimate().transpose()<<endl;

        mpCurrentFrame->SetPose(SE3d(vPR->R(), vPR->t()));
//        mpCurrentFrame->SetSpeedBias(vSpeed->estimate(), vBiasG->estimate(), vBiasA->estimate());
//        mpLastKeyFrame->SetSpeedBias(vSpeedL->estimate(), vBiasGL->estimate(), vBiasAL->estimate());
        mpCurrentFrame->SetSpeedBias(vSpeed->estimate(), mpCurrentFrame->BiasG()+vBiasGL->estimate(), mpCurrentFrame->BiasA()+vBiasAL->estimate());
        mpLastKeyFrame->SetSpeedBias(vSpeedL->estimate(), mpLastKeyFrame->BiasG(), mpLastKeyFrame->BiasA());

        for (shared_ptr<Feature> feat: mpCurrentFrame->mFeaturesLeft) {
            if (feat->mpPoint && feat->mbOutlier == false &&
                feat->mpPoint->Status() == MapPoint::GOOD &&
                feat->mfInvDepth < 0) {
                Vector3d pw = feat->mpPoint->mWorldPos;
                feat->mfInvDepth = 1.0 / (mpCurrentFrame->mRcw * pw + mpCurrentFrame->mtcw)[2];
            }
        }

        return inliers;
    }

    int TrackerLK::OptimizeCurrentPoseWithoutIMU() {

        assert(mpCurrentFrame != nullptr);

        // setup g2o
        typedef g2o::BlockSolverX Block;
        std::unique_ptr<Block::LinearSolverType> linearSolver (new g2o::LinearSolverEigen<Block::PoseMatrixType>());
        std::unique_ptr<Block> solver_ptr (new Block(std::move(linearSolver)));
        g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(std::move(solver_ptr));
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm(solver);

        VertexPR *vPR = new VertexPR();
        vPR->setEstimate(mpCurrentFrame->PR());
        vPR->setId(0);
        optimizer.addVertex(vPR);

        // Set MapPoint vertices
        const int N = mpCurrentFrame->mFeaturesLeft.size();

        vector<EdgeProjectPoseOnly *> vpEdgesProj;
        vector<size_t> vnIndexEdges;
        vpEdgesProj.reserve(N);
        vnIndexEdges.reserve(N);

        const float delta = sqrt(5.991);
        int nInitialCorrespondences = 0;

        for (size_t i = 0; i < N; i++) {
            shared_ptr<Feature> feat = mpCurrentFrame->mFeaturesLeft[i];
            shared_ptr<MapPoint> pMP = feat->mpPoint;

            if (pMP && pMP->Status() == MapPoint::GOOD) {
                nInitialCorrespondences++;
                feat->mbOutlier = false;

                EdgeProjectPoseOnly *eProj = new EdgeProjectPoseOnly(mpCam.get(), pMP->GetWorldPos());
                eProj->setVertex(0, vPR);
                eProj->setInformation(Matrix2d::Identity() * setting::invLevelSigma2[feat->mLevel]);
                eProj->setMeasurement(feat->mPixel.cast<double>());

                g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                eProj->setRobustKernel(rk);
                rk->setDelta(delta);
                optimizer.addEdge(eProj);
                vpEdgesProj.push_back(eProj);
                vnIndexEdges.push_back(i);
            }
        }

        const float chi2Mono[4] = {5.991, 5.991, 5.991, 5.991};
        const int its[4] = {10, 10, 10, 10};

        int nBad = 0;
        for (size_t it = 0; it < 4; it++) {
            // Reset estimate for vertex
            vPR->setEstimate(mpCurrentFrame->PR());

            optimizer.initializeOptimization(0);
            optimizer.optimize(its[it]);

            nBad = 0;
            for (size_t i = 0, iend = vpEdgesProj.size(); i < iend; i++) {
                EdgeProjectPoseOnly *e = vpEdgesProj[i];
                const size_t idx = vnIndexEdges[i];
                shared_ptr<Feature> feat = mpCurrentFrame->mFeaturesLeft[idx];
                if (feat->mbOutlier == true) {
                    e->computeError();
                }

                const float chi2 = e->chi2();

                if (chi2 > chi2Mono[it]) {
                    feat->mbOutlier = true;
                    e->setLevel(1);
                    nBad++;
                } else {
                    feat->mbOutlier = false;
                    e->setLevel(0);
                }

                if (it == 2) {
                    e->setRobustKernel(0);
                }
            }

            if (optimizer.edges().size() < 10)
                break;
        }

        // Recover optimized pose and return number of inliers
        int inliers = nInitialCorrespondences - nBad;
        LOG(INFO) << "bad/total = " << nBad << "/" << nInitialCorrespondences << endl;
        if (inliers < setting::minPoseOptimizationInliers) {
            LOG(WARNING) << "inliers is small, pose may by unreliable: " << inliers << endl;
        } else {
            // recover current pose
            mpCurrentFrame->SetPose(SE3d(vPR->R(), vPR->t()));
            LOG(INFO) << "Estimated Twb = \n" << mpCurrentFrame->GetPose().matrix() << endl;
        }

        for (shared_ptr<Feature> feat: mpCurrentFrame->mFeaturesLeft) {
            if (feat->mpPoint && feat->mbOutlier == false &&
                feat->mpPoint->Status() == MapPoint::GOOD &&
                feat->mfInvDepth < 0) {
                Vector3d pw = feat->mpPoint->mWorldPos;
                feat->mfInvDepth = 1.0 / (mpCurrentFrame->mRcw * pw + mpCurrentFrame->mtcw)[2];
            }
        }

        return inliers;
    }

    IMUPreIntegration TrackerLK::GetIMUFromLastKF() {

        assert(mpLastKeyFrame != nullptr);
        assert(mvIMUSinceLastKF.size() != 0);

        // Reset pre-integrator first
        IMUPreIntegration IMUPreInt;

        Vector3d bg = mpLastKeyFrame->BiasG();
        Vector3d ba = mpLastKeyFrame->BiasA();

        const IMUData &imu = mvIMUSinceLastKF.front();
        double dt = imu.mfTimeStamp - mpLastKeyFrame->mTimeStamp;
        IMUPreInt.update(imu.mfGyro - bg, imu.mfAcce - ba, dt);

        // integrate each imu
        for (size_t i = 0; i < mvIMUSinceLastKF.size(); i++) {
            const IMUData &imu = mvIMUSinceLastKF[i];
            double nextt;
            if (i == mvIMUSinceLastKF.size() - 1)
                nextt = mpCurrentFrame->mTimeStamp;         // last IMU, next is this KeyFrame
            else
                nextt = mvIMUSinceLastKF[i + 1].mfTimeStamp;  // regular condition, next is imu data

            // delta time
            double dt = nextt - imu.mfTimeStamp;
            // update pre-integrator
            IMUPreInt.update(imu.mfGyro - bg, imu.mfAcce - ba, dt);
        }

        return IMUPreInt;
    }


}
