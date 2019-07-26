#ifndef YGZ_G2OTYPES_H
#define YGZ_G2OTYPES_H

#include "common/NumTypes.h"
#include "common/Settings.h"
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

}

#endif
