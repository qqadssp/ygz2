#include "cv/TrackerLK.h"
#include "cv/ORBExtractor.h"
#include "cv/ORBMatcher.h"
#include "cv/LKFlow.h"
#include "common/Feature.h"
#include "backend/BackendSlidingWindowG2O.h"
#include "common/MapPoint.h"
#include "util/Viewer.h"

#include "common/G2OTypes.h"
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/core/robust_kernel_impl.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

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

    SE3d TrackerLK::InsertStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp) {

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        if (setting::trackerUseHistBalance) {
            // perform a histogram equalization
            cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
            Mat imgLeftAfter, imgRightAfter;
            clahe->apply(imRectLeft, imgLeftAfter);
            clahe->apply(imRectRight, imgRightAfter);
            mpCurrentFrame = shared_ptr<Frame>(new Frame(imgLeftAfter, imgRightAfter, timestamp, mpCam));
        } else {
            mpCurrentFrame = shared_ptr<Frame>(new Frame(imRectLeft, imRectRight, timestamp, mpCam));
        }

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
        return mpCurrentFrame->GetPose();
    }

    void TrackerLK::Track() {

        mTrackInliersCnt = 0;

        if (mState == NO_IMAGES_YET) {

            LOG(INFO) << "Detecting features" << endl;
            mpExtractor->Detect(mpCurrentFrame, true, false);    // extract the keypoint in left eye
            LOG(INFO) << "Compute stereo matches" << endl;
            mpMatcher->ComputeStereoMatchesOptiFlowCV(mpCurrentFrame);

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

            mSpeed = mpLastFrame->GetPose().inverse() * mpCurrentFrame->GetPose();
            mpLastFrame = mpCurrentFrame;
            return;

        } else if (mState == OK) {

        } else if (mState == WEAK) {

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
            refPts.push_back(feat->mPixel);
            trackedPts.push_back(feat->mPixel);
        }

        int cntMatches = LKFlowCV(mpLastFrame, mpCurrentFrame, refPts, trackedPts);
        // int cntMatches = LKFlow(mpLastFrame, mpCurrentFrame, trackedPts);

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
        mpMatcher->ComputeStereoMatchesOptiFlow(mpCurrentFrame, false);

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

    bool TrackerLK::NeedNewKeyFrame(const int &trackinliers) {

        int nKFs = mpBackEnd->GetAllKF().size();

        int nRefMatches = mpLastKeyFrame->TrackedMapPoints(2);

        // Thresholds
        float thRefRatio = 0.75f;
        if (nKFs < 2)
            thRefRatio = 0.4f;

        // Condition 1c: tracking is weak
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
//        if ((c1 || cTimeGap) && !isBackendBusy) {
        if ((c1) && !isBackendBusy) {
            return true;
        } else {
            return false;
        }

    }

    void TrackerLK::InsertKeyFrame() {

        mpCurrentFrame->SetThisAsKeyFrame();

        mpCurrentFrame->mpReferenceKF = mpLastKeyFrame;

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
    int TrackerLK::OptimizeCurrentPose() {

        assert(mState == OK || mState == WEAK || mState == NOT_INITIALIZED);

        return OptimizeCurrentPoseWithoutIMU();
    }

    int TrackerLK::OptimizeCurrentPoseWithoutIMU() {

        assert(mpCurrentFrame != nullptr);

        // setup g2o
        typedef g2o::BlockSolverX Block;
        std::unique_ptr<Block::LinearSolverType> linearSolver (new g2o::LinearSolverEigen<Block::PoseMatrixType>());
        std::unique_ptr<Block> solver_ptr (new Block(std::move(linearSolver)));

        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr));

        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm(solver);

        VertexPR *vPR = new VertexPR();
        vPR->setEstimate(mpCurrentFrame->PR());
        vPR->setId(0);
        optimizer.addVertex(vPR);

        // Set MapPoint vertices
        const int N = mpCurrentFrame->mFeaturesLeft.size();

        int nInitialCorrespondences = 0;
        int nBad = 0;

        vector<EdgeProjectPoseOnly *> vpEdgesProj;
        vector<size_t> vnIndexEdges;
        vpEdgesProj.reserve(N);
        vnIndexEdges.reserve(N);

        const float delta = sqrt(5.991);

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

        for (size_t it = 0; it < 4; it++) {
            // Reset estimate for vertex
            vPR->setEstimate(mpCurrentFrame->PR());

            optimizer.initializeOptimization(0);
            optimizer.optimize(its[it]);

            // outlier
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
}
