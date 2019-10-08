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
//          cancel if single thread
//            unique_lock<mutex> lock(mMutexNewKFs);
            mpCurrent = mpNewKFs.front();
            mpNewKFs.pop_front();
        }

        {
//          cancel if single thread
//            unique_lock<mutex> lockKF(mMutexKFs);
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
//                    pNewMP->ComputeDistinctiveDescriptor();
                    pNewMP->UpdateNormalAndDepth();
                    mpPoints.insert(pNewMP);

                } else if (feat->mpPoint && feat->mpPoint->Status() == MapPoint::GOOD) {
                    shared_ptr<MapPoint> pMP = feat->mpPoint;
                    if (mpPoints.count(pMP) == 0)
                        mpPoints.insert(pMP);
                    if (pMP->GetObsFromKF(mpCurrent) == -1) {
                        pMP->AddObservation(mpCurrent, i);
                        pMP->UpdateNormalAndDepth();
//                        pMP->ComputeDistinctiveDescriptor();
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
//                        pMP->ComputeDistinctiveDescriptor();
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
        if (trackerState == TrackerLK::OK || trackerState == TrackerLK::WEAK) {
            LocalBAXYZWithIMU();
        } else {
            LocalBAXYZWithoutIMU();
        }

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

//          uncancel if singel thread
            SetAcceptKeyFrames(false);
            mbAbortBA = false;
            ProcessNewKeyFrame();
            SetAcceptKeyFrames(true);

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

        const float thHuber = sqrt(5.991);
        const float thHuber2 = 5.991;

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

    void BackendSlidingWindowG2O::LocalBAXYZWithIMU(bool verbose) {

        // Bundle adjustment with IMU
        LOG(INFO) << "Call local ba XYZ with imu" << endl;

        // Gravity vector in world frame
        Vector3d GravityVec = mpTracker->g();

        // Setup optimizer
        typedef g2o::BlockSolverX Block;
        std::unique_ptr<Block::LinearSolverType> linearSolver (new g2o::LinearSolverEigen<Block::PoseMatrixType>());
        std::unique_ptr<Block> solver_ptr (new Block(std::move(linearSolver)));
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr));
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm(solver);

        int maxKFid = 0;
        for (shared_ptr<Frame> pKFi: mpKFs) {

            int idKF = pKFi->mnKFId * 4;
            if (idKF + 3 > maxKFid) {
                maxKFid = idKF + 3;
            }

            // P+R
            VertexPR *vPR = new VertexPR();
            vPR->setEstimate(pKFi->PR());
            vPR->setId(idKF);
            optimizer.addVertex(vPR);

            // speed
            VertexSpeed *vSpeed = new VertexSpeed();
            vSpeed->setEstimate(pKFi->Speed());
            vSpeed->setId(idKF + 1);
            optimizer.addVertex(vSpeed);

            // Ba and Bg
            VertexGyrBias *vBg = new VertexGyrBias();
            vBg->setId(idKF + 2);
//            vBg->setEstimate(pKFi->BiasG());
            vBg->setEstimate(Vector3d::Zero());
            optimizer.addVertex(vBg);

            VertexAcceBias *vBa = new VertexAcceBias();
            vBa->setId(idKF + 3);
//            vBa->setEstimate(pKFi->BiasA());
            vBa->setEstimate(Vector3d::Zero());
            optimizer.addVertex(vBa);

            // fix the first one
            if (pKFi == mpKFs.front()) {
                vPR->setFixed(true);
            }
        }

        vector<EdgePRV *> vpEdgePRV;
        vector<EdgeBiasG *> vpEdgeBg;
        vector<EdgeBiasA *> vpEdgeBa;

        // Use chi2inv() in MATLAB to compute the value corresponding to 0.95/0.99 prob. w.r.t 15DOF: 24.9958/30.5779
        // 12.592/16.812 for 0.95/0.99 6DoF
        // 16.919/21.666 for 0.95/0.99 9DoF
        //const float thHuberNavState = sqrt(30.5779);
        const float thHuberPRV = sqrt(1500 * 21.666);
        const float thHuberBias = sqrt(1500 * 16.812);

        // Inverse covariance of bias random walk
        Matrix3d infoBg = Matrix3d::Identity() / setting::gyrBiasRw2;
        Matrix3d infoBa = Matrix3d::Identity() / setting::accBiasRw2;

        for (shared_ptr<Frame> pKF1: mpKFs) {
            if (pKF1->mpReferenceKF.expired()) {
                if (pKF1 != mpKFs.front()) {
                    LOG(ERROR) << "non-first KeyFrame has no reference KF" << endl;
                }
                continue;
            }

            shared_ptr<Frame> pKF0 = pKF1->mpReferenceKF.lock();   // Previous KF

            // PR0, PR1, V0, V1, Bg0, Ba0
            EdgePRV *ePRV = new EdgePRV(GravityVec);
            ePRV->setVertex(0, optimizer.vertex(pKF0->mnKFId * 4));
            ePRV->setVertex(1, optimizer.vertex(pKF1->mnKFId * 4));
            ePRV->setVertex(2, optimizer.vertex(pKF0->mnKFId * 4 + 1));
            ePRV->setVertex(3, optimizer.vertex(pKF1->mnKFId * 4 + 1));
            ePRV->setVertex(4, optimizer.vertex(pKF0->mnKFId * 4 + 2));
            ePRV->setVertex(5, optimizer.vertex(pKF0->mnKFId * 4 + 3));
            ePRV->setMeasurement(pKF1->GetIMUPreInt());

            // set Covariance
            Matrix9d CovPRV = pKF1->GetIMUPreInt().getCovPVPhi();

            CovPRV.col(3).swap(CovPRV.col(6));
            CovPRV.col(4).swap(CovPRV.col(7));
            CovPRV.col(5).swap(CovPRV.col(8));
            CovPRV.row(3).swap(CovPRV.row(6));
            CovPRV.row(4).swap(CovPRV.row(7));
            CovPRV.row(5).swap(CovPRV.row(8));
            ePRV->setInformation(CovPRV.inverse());

            g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
            ePRV->setRobustKernel(rk);
            rk->setDelta(thHuberPRV);
            optimizer.addEdge(ePRV);
            vpEdgePRV.push_back(ePRV);

            double dt = pKF1->GetIMUPreInt().getDeltaTime();
            EdgeBiasG *eBG = new EdgeBiasG();
            eBG->setVertex(0, optimizer.vertex(pKF0->mnKFId * 4 + 2));
            eBG->setVertex(1, optimizer.vertex(pKF1->mnKFId * 4 + 2));
            eBG->setMeasurement(Vector3d::Zero());
            eBG->setInformation(infoBg / dt);

            g2o::RobustKernelHuber *rkb = new g2o::RobustKernelHuber;
            eBG->setRobustKernel(rkb);
            rkb->setDelta(thHuberBias);
            optimizer.addEdge(eBG);
            vpEdgeBg.push_back(eBG);

            EdgeBiasA *eBA = new EdgeBiasA();
            eBA->setVertex(0, optimizer.vertex(pKF0->mnKFId * 4 + 3));
            eBA->setVertex(1, optimizer.vertex(pKF1->mnKFId * 4 + 3));
            eBA->setMeasurement(Vector3d::Zero());
            eBA->setInformation(infoBa / dt);

            g2o::RobustKernelHuber *rkba = new g2o::RobustKernelHuber;
            eBA->setRobustKernel(rkba);
            rkba->setDelta(thHuberBias);
            optimizer.addEdge(eBA);
            vpEdgeBa.push_back(eBA);
        }

        // Set MapPoint vertices
        vector<EdgePRXYZ *> vpEdgePoints;
        vpEdgePoints.reserve(mpPoints.size());
        vector<shared_ptr<Frame> > vpFrames;
        vector<shared_ptr<MapPoint> > vpMappoints;
        vpFrames.reserve(mpPoints.size());
        vpMappoints.reserve(mpPoints.size());

        const float thHuber = sqrt(5.991);
        const float thHuber2 = 5.991;

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

                int mpUseCnt = 0;

                // add edges in observation
                for (auto &obs: mp->mObservations) {
                    if (obs.first.expired())
                        continue;

                    shared_ptr<Frame> kf = obs.first.lock();
                    // if the MapPoint links to more than one KeyFrame, then add this MapPoint vertex
                    if (mpUseCnt == 0) {
                        optimizer.addVertex(vXYZ);
                    }
                    mpUseCnt++;
                    shared_ptr<Feature> featObs = kf->mFeaturesLeft[obs.second];

                    EdgePRXYZ *eProj = new EdgePRXYZ(kf->mpCam.get());
                    eProj->setVertex(0, optimizer.vertex(kf->mnKFId * 4));
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

                if (mpUseCnt == 0) {
                    delete vXYZ;
                }
            }
        }

        if (mbAbortBA)
            return;

        optimizer.initializeOptimization();
        optimizer.optimize(100);

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

        // do it again
        optimizer.initializeOptimization();
        optimizer.optimize(100);

        // check Edge PRV
        int cntPRVOutliers = 0;
        for (EdgePRV *e: vpEdgePRV) {
            // LOG(INFO) << "PRV chi2 = " << e->chi2() << endl;
            if (e->chi2() > thHuberPRV) {
                cntPRVOutliers++;
            }
        }
        // LOG(INFO) << "PRV outliers: " << cntPRVOutliers << endl;

        if (mbAbortBA)
            return;


        // recover the pose and points estimation
        shared_ptr<Frame> lastFramePtr = nullptr;

        for (shared_ptr<Frame> frame: mpKFs) {
            VertexPR *vPR = (VertexPR *) optimizer.vertex(frame->mnKFId * 4);
            VertexSpeed *vSpeed = (VertexSpeed *) optimizer.vertex(frame->mnKFId * 4 + 1);
            VertexGyrBias *vBg = (VertexGyrBias *) optimizer.vertex(frame->mnKFId * 4 + 2);
            VertexAcceBias *vBa = (VertexAcceBias *) optimizer.vertex(frame->mnKFId * 4 + 3);

            if (verbose) {
                LOG(INFO) << "Frame " << frame->mnKFId << ", pose changed from = \n" << frame->GetPose().matrix()
                          << "\nto \n" << SE3d(vPR->R(), vPR->t()).matrix() << endl;
                LOG(INFO) << "Speed and bias changed from = \n" << frame->mSpeedAndBias.transpose() << "\nto " <<
                          vSpeed->estimate().transpose() << "," << vBg->estimate().transpose() << ","
                          << vBa->estimate().transpose() << endl;
            }

            frame->SetPose(SE3d(vPR->R(), vPR->t()));
//            frame->SetSpeedBias(vSpeed->estimate(), vBg->estimate(), vBa->estimate());
            if(lastFramePtr == nullptr)
                frame->SetSpeedBias(vSpeed->estimate(), frame->BiasG(), frame->BiasA());
            else
                frame->SetSpeedBias(vSpeed->estimate(), lastFramePtr->BiasG()+vBg->estimate(), lastFramePtr->BiasA()+vBa->estimate());
            frame->ComputeIMUPreInt();
            lastFramePtr = frame;
        }

        // and the points
        for (shared_ptr<MapPoint> mp: vpMappoints) {
            if (mp && mp->isBad() == false && mp->mState == MapPoint::GOOD && mp->Observations() > 1) {
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

        mbFirstCall = false;
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
