#include "common/Feature.h"
#include "common/MapPoint.h"
#include "common/Frame.h"
#include "backend/BackendSlidingWindowG2O.h"
#include "cv/TrackerLK.h"
#include "cv/ORBMatcher.h"
#include "common/Camera.h"

// g2o related
#include "common/G2OTypes.h"
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/core/robust_kernel_impl.h>

namespace ygz {

    bool BackendSlidingWindowG2O::IsBusy() {
        unique_lock<mutex> lock(mMutexAccept);
        return !mbAcceptKeyFrames;
    }

    void BackendSlidingWindowG2O::MainLoop() {

        mbFinished = false;
        while (1) {

            // Tracking will see that Local Mapping is busy
            SetAcceptKeyFrames(false);

            // Check if there are keyframes in the queue
            if (CheckNewKeyFrames()) {
                // BoW conversion and insertion in Map
                mbAbortBA = false;
                LOG(INFO) << "Process new KF" << endl;
                ProcessNewKeyFrame();
                LOG(INFO) << "Process new KF done." << endl;
            } else if (Stop()) {
                // Safe area to stop
                while (isStopped() && !CheckFinish()) {
                    usleep(3000);
                }
                if (CheckFinish())
                    break;
            }

            ResetIfRequested();

            // Tracking will see that Local Mapping is busy
            SetAcceptKeyFrames(true);

            if (CheckFinish())
                break;

            usleep(3000);
        }

        SetFinish();
    }

    void BackendSlidingWindowG2O::ProcessNewKeyFrame() {
        {
            unique_lock<mutex> lock(mMutexNewKFs);
            mpCurrent = mpNewKFs.front();
            mpNewKFs.pop_front();
        }

        {
            unique_lock<mutex> lockKF(mMutexKFs);
            mpKFs.push_back(mpCurrent);
        }

        mpCurrent->AssignFeaturesToGrid();

        int cntSetGood = 0, cntImmature = 0;

        {
            unique_lock<mutex> lockMps(mMutexPoints);
            unique_lock<mutex> lock(mpCurrent->mMutexFeature);
            for (size_t i = 0; i < mpCurrent->mFeaturesLeft.size(); i++) {
                shared_ptr<Feature> feat = mpCurrent->mFeaturesLeft[i];
                if (feat == nullptr)
                    continue;
                if (feat->mfInvDepth > 0 && feat->mpPoint == nullptr) {

                    if (feat->mfInvDepth < setting::minNewMapPointInvD ||
                        feat->mfInvDepth > setting::maxNewMapPointInvD)
                        continue;

                    // create a new map point
                    shared_ptr<MapPoint> pNewMP(new MapPoint(mpCurrent, i));
                    pNewMP->AddObservation(mpCurrent, i);
                    feat->mpPoint = pNewMP;
                    pNewMP->UpdateNormalAndDepth();
                    mpPoints.insert(pNewMP);

                } else if (feat->mpPoint && feat->mpPoint->Status() == MapPoint::GOOD) {
                    shared_ptr<MapPoint> pMP = feat->mpPoint;
                    if (mpPoints.count(pMP) == 0)
                        mpPoints.insert(pMP);
                    if (pMP->GetObsFromKF(mpCurrent) == -1) {
                        pMP->AddObservation(mpCurrent, i);
                        pMP->UpdateNormalAndDepth();
                    }
                } else if (feat->mpPoint && feat->mpPoint->Status() == MapPoint::IMMATURE) {
                    shared_ptr<MapPoint> pMP = feat->mpPoint;
                    cntImmature++;
                    if (mpPoints.count(pMP) == 0)
                        mpPoints.insert(pMP);
                    // add observation
                    if (pMP->GetObsFromKF(mpCurrent) == -1) {
                        pMP->AddObservation(mpCurrent, i);
                        pMP->UpdateNormalAndDepth();
                    }
                    if (pMP->TestGoodFromImmature()) {
                        pMP->SetStatus(MapPoint::GOOD);
                        cntSetGood++;
                    }
                }
            }
        }

        LOG(INFO) << "new good points: " << cntSetGood << " in total immature points: " << cntImmature << endl;
        if (mpKFs.size() == 1) {
            // don't need BA
            return;
        }

        // do ba
        TrackerLK::eTrackingState trackerState = mpTracker->GetState();

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        LocalBAXYZWithoutIMU();

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        double timeCost = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        LOG(INFO) << "Local BA cost time: " << timeCost << endl;

        if (mpKFs.size() > setting::numBackendKeyframes) {
            DeleteKF(0);
        }

        CleanMapPoint();
        LOG(INFO) << "Backend KF: " << mpKFs.size() << ", map points: " << mpPoints.size() << endl;
    }

    int BackendSlidingWindowG2O::InsertKeyFrame(shared_ptr<Frame> newKF) {
        unique_lock<mutex> lock(mMutexNewKFs);
        mpNewKFs.push_back(newKF);
        mbAbortBA = true;

//            SetAcceptKeyFrames(false);
//                mbAbortBA = false;
//                ProcessNewKeyFrame();
//            SetAcceptKeyFrames(true);

        return 0;
    }

    std::set<shared_ptr<MapPoint>> BackendSlidingWindowG2O::GetLocalMap() {
        unique_lock<mutex> lock(mMutexPoints);
        return mpPoints;
    }

    void BackendSlidingWindowG2O::DeleteKF(int idx) {
        LOG(INFO) << "Deleting KF " << idx << endl;
        if (idx < 0 || idx >= mpKFs.size() - 2) {
            LOG(ERROR) << "index error in DeleteKF" << endl;
            return;
        }

        // change reference KeyFrame
        if ( idx != 0)
            mpKFs[idx + 1]->mpReferenceKF = mpKFs[idx]->mpReferenceKF;

        // erase KeyFrame in Map
        LOG(INFO)<<"Erase KF in mpKFs"<<endl;
        unique_lock<mutex> lock(mMutexKFs);
        if (idx == 0)
            mpKFs.pop_front();
        else
            mpKFs.erase(mpKFs.begin() + idx);
        LOG(INFO)<<"Done"<<endl;
    }

    // 找到当前关键帧在图中邻接的一些关键帧
    // 对于每一个邻接的关键帧，根据当前关键帧和该邻接关键帧的姿态，算出两幅图像的基本矩阵
    // 搜索当前关键帧和邻接关键帧之间未成为3d点的特征点匹配
    //         在搜索特征点匹配时，先获分别获得这两个帧在数据库中所有二级节点对应的特征点集
    //         获取两个帧中所有相等的二级节点对，在每个二级节点对
    //         能够得到一对特征点集合，分别对应当前关键帧和相邻关键帧
    //         在两个集合中利用极线约束(基本矩阵)和描述子距离来寻求特征点匹配对
    //         可以在满足极线约束的条件下搜索描述子距离最小并且不超过一定阈值的匹配对
    // 获得未成为3d点的特征点匹配集合后，通过三角化获得3d坐标
    // 在通过平行、重投影误差、尺度一致性等检查后，则建立一个对应的3d点MapPoint对象
    // 需要标记新的3d点与两个关键帧之间的观测关系，还需要计算3d点的平均观测方向以及最佳的描述子
    // 最后将新产生的3d点放到检测队列中等待检验

    int BackendSlidingWindowG2O::CleanMapPoint() {

        int cnt = 0;
        unique_lock<mutex> lockMps(mMutexPoints);
        for (auto iter = mpPoints.begin(); iter != mpPoints.end();) {
            shared_ptr<MapPoint> mp = *iter;
            if (mp->mState == MapPoint::BAD) {
                iter = mpPoints.erase(iter);
                cnt++;
            } else {
                if (mp->mpRefKF.expired()) {
                    // erase the reference KeyFrame in the observation
                    // unique_lock<mutex> lock(mp->mMutexFeatures);
                    mp->RemoveObservation(mp->mpRefKF);
                    // change the reference KeyFrame
                    if (mp->SetAnotherRef() == false) {
                        iter = mpPoints.erase(iter);
                        cnt++;
                        continue;
                    }
                }
                iter++;
            }
        }

        LOG(INFO) << "Clean map point done, total bad points " << cnt << endl;
        return 0;
    }

    void BackendSlidingWindowG2O::LocalBAXYZWithoutIMU(bool verbose) {

        LOG(INFO) << "Calling Local BA XYZ without IMU" << endl;
        // Setup optimizer
        typedef g2o::BlockSolverX Block;
        std::unique_ptr<Block::LinearSolverType> linearSolver (new g2o::LinearSolverEigen<Block::PoseMatrixType>());
        std::unique_ptr<Block> solver_ptr (new Block(std::move(linearSolver)));

        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr));

        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm(solver);

        int maxKFid = 0;
        for (shared_ptr<Frame> pKFi: mpKFs) {

            int idKF = pKFi->mnKFId;
            if (idKF > maxKFid) {
                maxKFid = idKF;
            }

            // P+R
            VertexPR *vPR = new VertexPR();
            vPR->setEstimate(pKFi->PR());
            vPR->setId(idKF);
            optimizer.addVertex(vPR);

            // fix the first one and the last one.
            if (pKFi == mpKFs.front() /*|| pKFi == mpKFs.back() */ ) {
                vPR->setFixed(true);
            }
        }

        // Set MapPoint vertices
        vector<EdgePRXYZ *> vpEdgePoints;
        vpEdgePoints.reserve(mpPoints.size() * 2);

        vector<shared_ptr<Frame> > vpFrames;
        vector<shared_ptr<MapPoint> > vpMappoints;
        vpFrames.reserve(mpPoints.size());
        vpMappoints.reserve(mpPoints.size());

        const float thHuber = sqrt(5.991);  // 0.95
        const float thHuber2 = 5.991;  // 0.95

        for (shared_ptr<MapPoint> mp: mpPoints) {
            if (mp == nullptr)
                continue;
            if (mp->isBad())
                continue;

            if (mp->Status() == MapPoint::GOOD && mp->Observations() > 1) {

                VertexPointXYZ *vXYZ = new VertexPointXYZ;
                int idMP = mp->mnId + maxKFid + 1;
                vXYZ->setId(idMP);
                vXYZ->setEstimate(mp->GetWorldPos());
                vXYZ->setMarginalized(true);

                int useCnt = 0;

                // add edges in observation
                for (auto &obs: mp->mObservations) {
                    auto f = obs.first;
                    if (f.expired())
                        continue;

                    shared_ptr<Frame> kf = f.lock();

                    // if the MapPoint links to more than one KeyFrame, then add this MapPoint vertex
                    if (useCnt == 0) {
                        optimizer.addVertex(vXYZ);
                    }

                    useCnt++;
                    shared_ptr<Feature> featObs = kf->mFeaturesLeft[obs.second];

                    EdgePRXYZ *eProj = new EdgePRXYZ(kf->mpCam.get());
                    eProj->setVertex(0, optimizer.vertex(kf->mnKFId));
                    eProj->setVertex(1, (g2o::OptimizableGraph::Vertex *) vXYZ);
                    eProj->setMeasurement(featObs->mPixel.cast<double>());

                    const float &invSigma2 = setting::invLevelSigma2[featObs->mLevel];
                    eProj->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    eProj->setRobustKernel(rk);
                    rk->setDelta(thHuber);

                    optimizer.addEdge(eProj);
                    vpEdgePoints.push_back(eProj);
                    vpFrames.push_back(kf);
                    vpMappoints.push_back(mp);
                }

                // if the MapPoint not used, delete it
                if (useCnt == 0) {
                    delete vXYZ;
                }
            }
        }

        if (mbAbortBA)
            return;

        optimizer.initializeOptimization();
        optimizer.optimize(5);

        // Check inliers and optimize again without outliers
        int cntPRXYZOutliers = 0;
        for (EdgePRXYZ *e: vpEdgePoints) {
            if (e->chi2() > thHuber2 || e->isDepthValid() == false) {
                e->setLevel(1);
                cntPRXYZOutliers++;
            } else {
                e->setLevel(0);
            }
            e->setRobustKernel(nullptr);
        }

        LOG(INFO) << "PRXYZ outliers: " << cntPRXYZOutliers << endl;

        if (mbAbortBA)
            return;

        // do it again without outliers
        optimizer.initializeOptimization();
        optimizer.optimize(10);

        // recover the pose and points estimation
        for (shared_ptr<Frame> frame: mpKFs) {
            VertexPR *vPR = (VertexPR *) optimizer.vertex(frame->mnKFId);
            frame->SetPose(SE3d(vPR->R(), vPR->t()));
        }

        // and the points
        for (shared_ptr<MapPoint> mp: vpMappoints) {
            if (mp && mp->Observations() > 1) {
                VertexPointXYZ *v = (VertexPointXYZ *) optimizer.vertex(mp->mnId + maxKFid + 1);
                mp->SetWorldPos(v->estimate());
            }
        }

        int cntSetBad = 0;
        for (size_t i = 0, iend = vpEdgePoints.size(); i < iend; i++) {
            EdgePRXYZ *e = vpEdgePoints[i];
            shared_ptr<MapPoint> mp = vpMappoints[i];
            shared_ptr<Frame> kf = vpFrames[i];
            if (e->chi2() > thHuber2 || e->isDepthValid() == false) {
                // erase observation
                int idx = mp->GetObsFromKF(kf);
                if (idx != -1) {

                    // 1. delete feature in KeyFrame
                    unique_lock<mutex> lock(kf->mMutexFeature);
                    kf->mFeaturesLeft[idx] = nullptr;

                    // 2. delete observation in MapPoint
                    mp->RemoveObservation(kf);

                    if (mp->mObservations.size() < 2) { // if less than 2
                        mp->SetBadFlag();
                        cntSetBad++;
                        continue;
                    }

                    if (kf == mp->mpRefKF.lock()) { // change reference KeyFrame
                        mp->SetAnotherRef();
                    }
                }
            }
        }

        LOG(INFO) << "Set total " << cntSetBad << " bad map points" << endl;
    }

    void BackendSlidingWindowG2O::Shutdown() {
        mbAbortBA = true;
        SetFinish();
    }

    void BackendSlidingWindowG2O::Reset() {
        LOG(INFO) << "Backend is reset" << endl;
        RequestReset();
        mpCurrent = nullptr;
        mpKFs.clear();
        mpPoints.clear();
        mbFirstCall = true;
        LOG(INFO) << "backend reset done." << endl;
    }
}
