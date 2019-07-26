#include "common/G2OTypes.h"
#include "common/Camera.h"
#include "common/Frame.h"

namespace ygz {

    void EdgeProjectPoseOnly::computeError() {
        const VertexPR *vPR = static_cast<VertexPR *>(_vertices[0]);
        Matrix3d Rwb = vPR->R();
        Vector3d twb = vPR->t();
        SE3d Twb(Rwb, twb);
        SE3d Tcb = setting::TBC.inverse();
        SE3d Tcw = Tcb * Twb.inverse();
        Vector3d pc = Tcw * pw;
        double invz = 1.0 / pc[2];

        if (invz < 0) {
            LOG(INFO) << "Error, invz = " << invz << endl;
            setLevel(1);
            _error = Vector2d(setting::imageWidth, setting::imageHeight);
            return;
        }

        double px = invz * fx * pc[0] + cx;
        double py = invz * fy * pc[1] + cy;

        _error = _measurement - Vector2d(px, py);
    }

    void EdgeProjectPoseOnly::linearizeOplus() {
        const VertexPR *vPR = static_cast<const VertexPR *>(_vertices[0]);

        Matrix3d Rwb = vPR->R();
        Vector3d Pwb = vPR->t();

        SE3d Twb(Rwb, Pwb);
        SE3d Tcb = setting::TBC.inverse();
        Matrix3d Rcb = Tcb.rotationMatrix();
        Vector3d tcb = Tcb.translation();

        SE3d Tcw = Tcb * Twb.inverse();
        Vector3d Pc = Tcw * pw;

        double x = Pc[0];
        double y = Pc[1];
        double z = Pc[2];
        double invz = 1.0 / z;

        // Jacobian of camera projection
        Matrix<double, 2, 3> Maux;
        Maux.setZero();
        Maux(0, 0) = fx;
        Maux(0, 1) = 0;
        Maux(0, 2) = -x * invz * fx;
        Maux(1, 0) = 0;
        Maux(1, 1) = fy;
        Maux(1, 2) = -y * invz * fy;
        Matrix<double, 2, 3> Jpi = Maux * invz;

        // error = obs - pi( Pc )
        // Pw <- Pw + dPw,          for Point3D
        // Rwb <- Rwb*exp(dtheta),  for R
        // Pwb <- Pwb + dPwb,       for P

        // Jacobian of Pc/error w.r.t dPwb
        Matrix<double, 2, 3> JdPwb = -Jpi * (-Rcb * Rwb.transpose());
        // Jacobian of Pc/error w.r.t dRwb
        Vector3d Paux = Rcb * Rwb.transpose() * (pw - Pwb);
        Matrix<double, 2, 3> JdRwb = -Jpi * (SO3d::hat(Paux) * Rcb);

        // Jacobian of Pc w.r.t P,R
        Matrix<double, 2, 6> JPR = Matrix<double, 2, 6>::Zero();
        JPR.block<2, 3>(0, 0) = JdPwb;
        JPR.block<2, 3>(0, 3) = JdRwb;

        _jacobianOplusXi = JPR;
    }

    void EdgePRXYZ::computeError() {
        VertexPointXYZ *v1 = dynamic_cast<VertexPointXYZ *>(_vertices[1]);
        VertexPR *vPR0 = dynamic_cast<VertexPR *>(_vertices[0]);

        const Matrix3d Rwb0 = vPR0->R();
        const Vector3d twb0 = vPR0->t();
        SE3d Twc = SE3d(Rwb0, twb0) * setting::TBC;
        SE3d Tcw = Twc.inverse();

        Vector3d Pc = Tcw * v1->estimate();
        depth = Pc[2];

        // point inverse depth in reference KF
        double xi = Pc[0] / Pc[2];
        double yi = Pc[1] / Pc[2];
        double u = fx * xi + cx;
        double v = fy * yi + cy;
        _error = _measurement - Vector2d(u, v);
    }

    void EdgePRXYZ::linearizeOplus() {
        const VertexPR *vPR0 = dynamic_cast<const VertexPR *>(_vertices[0]);
        const VertexPointXYZ *vXYZ = dynamic_cast<const VertexPointXYZ *>(_vertices[1]);

        const Matrix3d Rwb = vPR0->R();
        const Vector3d Pwb = vPR0->t();
        SE3d Twb(Rwb, Pwb);
        SE3d Tcb = setting::TBC.inverse();
        const Matrix3d Rcb = Tcb.rotationMatrix();
        const Vector3d Pcb = Tcb.translation();
        const Vector3d Pw = vXYZ->estimate();

        // point coordinate in reference KF, body
        Vector3d Pc = (Twb * setting::TBC).inverse() * vXYZ->estimate();

        double x = Pc[0];
        double y = Pc[1];
        double z = Pc[2];
        double zinv = 1.0 / (z + 1e-9);

        Matrix<double, 2, 3> Maux;
        Maux.setZero();
        Maux(0, 0) = fx;
        Maux(0, 1) = 0;
        Maux(0, 2) = -x * zinv * fx;
        Maux(1, 0) = 0;
        Maux(1, 1) = fy;
        Maux(1, 2) = -y * zinv * fy;
        Matrix<double, 2, 3> Jpi = Maux / z;

        // Jacobian of Pc/error w.r.t dPwb
        Matrix<double, 2, 3> JdPwb = -Jpi * (-Rcb * Rwb.transpose());
        // Jacobian of Pc/error w.r.t dRwb
        Vector3d Paux = Rcb * Rwb.transpose() * (Pw - Pwb);
        Matrix<double, 2, 3> JdRwb = -Jpi * (SO3d::hat(Paux) * Rcb);

        Matrix<double, 2, 6> JPR = Matrix<double, 2, 6>::Zero();
        JPR.block<2, 3>(0, 0) = JdPwb;
        JPR.block<2, 3>(0, 3) = JdRwb;

        _jacobianOplusXi = JPR;
        _jacobianOplusXj = Jpi * Rcb * Rwb.transpose();
    }

}
