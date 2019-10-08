#ifndef YGZ_G2OTYPES_H
#define YGZ_G2OTYPES_H

#include "common/NumTypes.h"
#include "common/Settings.h"
#include "common/IMUPreIntegration.h"
#include "common/Camera.h"

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_multi_edge.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/types/slam3d/vertex_pointxyz.h>
#include <g2o/types/slam3d/edge_pointxyz.h>

using namespace Eigen;

using namespace g2o;

namespace ygz {

    struct CameraParam;

    struct Frame;

    // ---------------------------------------------------------------------------------------------------------
    class VertexPR : public BaseVertex<6, Vector6d> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        VertexPR() : BaseVertex<6, Vector6d>() {}

        bool read(std::istream &is) override { return true; }

        bool write(std::ostream &os) const override { return true; }

        virtual void setToOriginImpl() override {
            _estimate.setZero();
        }

        virtual void oplusImpl(const double *update_) override {

            _estimate.segment<3>(0) += Vector3d(update_[0], update_[1], update_[2]);
            _estimate.segment<3>(3) = SO3d::log(
                    SO3d::exp(_estimate.segment<3>(3)) *
                    SO3d::exp(Vector3d(update_[3], update_[4], update_[5])));
        }

        Matrix3d R() const {
            return SO3d::exp(_estimate.segment<3>(3)).matrix();
        }

        Vector3d t() const {
            return _estimate.head<3>();
        }
    };

    // Speed
    typedef g2o::VertexPointXYZ VertexSpeed;

    // Bias Acce
    typedef g2o::VertexPointXYZ VertexAcceBias;

    // Bias Gyro
    typedef g2o::VertexPointXYZ VertexGyrBias;

    // ---------------------------------------------------------------------------------------------------------

    class EdgePRXYZ : public BaseBinaryEdge<2, Vector2d, VertexPR, VertexPointXYZ> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        EdgePRXYZ(CameraParam *cam) : fx(cam->fx), fy(cam->fy), cx(cam->cx), cy(cam->cy) {}

        bool read(std::istream &is) override { return true; }

        bool write(std::ostream &os) const override { return true; }

        virtual void computeError() override;

        virtual void linearizeOplus() override;

        bool isDepthValid() {
            return depth > 0;
        }

    protected:
        double fx, fy, cx, cy;
        double depth = 0;
    };

    /**
     * The pre-integration IMU motion constraint
     * Connect 6 vertex: PR0, V0, biasG0, bias A0 and PR1, V1
     * Vertex 0: PR0
     * Vertex 1: PR1
     * Vertex 2: V0
     * Vertex 3: V1
     * Vertex 4: biasG0
     * Vertex 5: biasA0
     * Error order: error_P, error_R, error_V
     *      different from PVR edge
     */
    class EdgePRV : public BaseMultiEdge<9, IMUPreIntegration> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        EdgePRV(const Vector3d &gw) : BaseMultiEdge<9, IMUPreIntegration>(), GravityVec(gw) {
            resize(6);
        }

        bool read(std::istream &is) override { return true; }

        bool write(std::ostream &os) const override { return true; }

        virtual void computeError() override;

        virtual void linearizeOplus() override;

    protected:
        // Gravity vector in 'world' frame
        Vector3d GravityVec;
    };

    typedef g2o::EdgePointXYZ EdgeBiasG;

    typedef g2o::EdgePointXYZ EdgeBiasA;

    class EdgeProjectPoseOnly : public BaseUnaryEdge<2, Vector2d, VertexPR> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        EdgeProjectPoseOnly(const CameraParam *cam, const Vector3d &pw_)
                : BaseUnaryEdge<2, Vector2d, VertexPR>(), pw(pw_) {
            fx = cam->fx;
            fy = cam->fy;
            cx = cam->cx;
            cy = cam->cy;
        }

        bool read(std::istream &is) override { return true; }

        bool write(std::ostream &os) const override { return true; }

        virtual void computeError() override;

        virtual void linearizeOplus() override;

    private:
        double fx = 0, fy = 0, cx = 0, cy = 0;
        Vector3d pw;    // world 3d position

    };

    /**
     * @brief The EdgeGyrBias class
     * For gyroscope bias compuation in Visual-Inertial initialization
     */

    class EdgeGyrBias : public BaseUnaryEdge<3, Vector3d, VertexGyrBias> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        EdgeGyrBias() : BaseUnaryEdge<3, Vector3d, VertexGyrBias>() {}

        bool read(std::istream &is) override { return true; }

        bool write(std::ostream &os) const override { return true; }

        Matrix3d dRbij;
        Matrix3d J_dR_bg;
        Matrix3d Rwbi;
        Matrix3d Rwbj;

        void computeError() override {
            const VertexGyrBias *v = static_cast<const VertexGyrBias *>(_vertices[0]);
            Vector3d bg = v->estimate();
            Matrix3d dRbg = SO3d::exp(J_dR_bg * bg).matrix();
            SO3d errR((dRbij * dRbg).transpose() * Rwbi.transpose() * Rwbj); // dRij^T * Riw * Rwj
            _error = errR.log();
        }

        virtual void linearizeOplus() override {
            SO3d errR(dRbij.transpose() * Rwbi.transpose() * Rwbj); // dRij^T * Riw * Rwj
            Matrix3d Jlinv = SO3d::JacobianLInv(errR.log());

            _jacobianOplusXi = -Jlinv * J_dR_bg;
        }
    };

}

#endif
