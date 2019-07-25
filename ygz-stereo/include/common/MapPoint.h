#ifndef YGZ_MAPPOINT_H
#define YGZ_MAPPOINT_H

#include "common/Settings.h"
#include "common/NumTypes.h"

#include <mutex>

using namespace std;

namespace ygz {

    struct Frame;

    struct Feature;

    struct MapPoint {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        enum eMapPointState {
            GOOD = 0,
            IMMATURE,
            BAD
        };

        MapPoint(shared_ptr<Frame> kf, const size_t &indexF);

        MapPoint() {
            mnId = nNextId++;
        }

        ~MapPoint();

        int Observations();

        int GetObsFromKF(shared_ptr<Frame> pKF);

        bool RemoveObservation( shared_ptr<Frame>& pKF );
        bool RemoveObservation( weak_ptr<Frame>& pKF );

        void AddObservation(shared_ptr<Frame> pKF, size_t idx);

        void SetRefKF( shared_ptr<Frame> pKF ) {
            unique_lock<mutex> lock(mMutexFeatures);
            unique_lock<mutex> lock2(mMutexPos);
            mpRefKF = pKF;
        }

        void SetBadFlag();

        bool TestGoodFromImmature();

        void SetStatus(eMapPointState state);

        bool isBad() {
            return mState == eMapPointState::BAD;
        }

        eMapPointState Status() {
            unique_lock<mutex> lock(mMutexFeatures);
            return mState;
        }

        void IncreaseFound(int n = 1);

        inline int GetFound() {
            return mnFound;
        }

        void UpdateNormalAndDepth();

        void SetWorldPos(const Vector3d &Pos);

        Vector3d GetWorldPos();

        Vector3d GetNormal();

        bool SetAnotherRef();

        long unsigned int mnId = 0; ///< Global ID for MapPoint
        static long unsigned int nNextId; ///< next id

        eMapPointState mState = eMapPointState::GOOD;

        Vector3d mWorldPos = Vector3d(0, 0, 0);

        typedef std::map<weak_ptr<Frame>, size_t, std::owner_less<weak_ptr<Frame>>> ObsMap;
        ObsMap mObservations;

        Vector3d mNormalVector = Vector3d(0, 0, 0);

        int mnVisible = 1;
        int mnFound = 1;

        bool mbTrackInView = false;
        float mTrackProjX = -1;
        float mTrackProjY = -1;
        int mnTrackScaleLevel = 0;
        float mTrackViewCos = -1.0;
        uchar mGray = 0;

        weak_ptr<Frame> mpRefKF;

        std::mutex mMutexPos;
        std::mutex mMutexFeatures;

    };
}


#endif
