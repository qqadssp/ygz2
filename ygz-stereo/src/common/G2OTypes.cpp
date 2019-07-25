#include "common/G2OTypes.h"
#include "common/Camera.h"
#include "common/Frame.h"

namespace ygz {


    void EdgeIDPPrior::computeError() {
        const VertexPointInvDepth *vIDP = static_cast<const VertexPointInvDepth *>(_vertices[0]);
        _error.setZero();
        _error(0) = vIDP->estimate() - measurement();
        if (measurement() <= 0) cerr << "EdgeIDPPrior measurement < 0, = " << measurement() << endl;
    }

    void EdgeIDPPrior::linearizeOplus() {
        _jacobianOplusXi.setZero();
        _jacobianOplusXi(0) = 1;
    }

    /**
     * Erorr = obs - pi(Px)
     */
    void EdgePRIDP::computeError() {
        const VertexPointInvDepth *vIDP = dynamic_cast<const VertexPointInvDepth *>( _vertices[0]);
        const VertexPR *vPR0 = dynamic_cast<const VertexPR *>(_vertices[1]);
        const VertexPR *vPRi = dynamic_cast<const VertexPR *>(_vertices[2]);
        const VertexPR *vPRcb = dynamic_cast<const VertexPR *>(_vertices[3]);

        const Matrix3d R0 = vPR0->R();
        const Vector3d t0 = vPR0->t();
        const Matrix3d Ri = vPRi->R();
        const Vector3d ti = vPRi->t();
        const Matrix3d Rcb = vPRcb->R();
        const Vector3d tcb = vPRcb->t();

        // point inverse depth in reference KF
        double rho = vIDP->estimate();
        if (rho < 1e-6) {
            LOG(WARNING) << "Inv depth should not be negative: " << rho << endl;
        }

        // point coordinate in reference KF, body
        Vector3d P0(x, y, 1.0);
        P0 = P0 * (1.0f / rho);
        const Matrix3d Rcic0 = Rcb * Ri.transpose() * R0 * Rcb.transpose();
        const Vector3d Pi = Rcic0 * P0 + tcb - Rcic0 * tcb + Rcb * Ri.transpose() * (t0 - ti);

        double xi = Pi[0] / Pi[2];
        double yi = Pi[1] / Pi[2];
        double u = mpCam->fx * xi + mpCam->cx;
        double v = mpCam->fy * yi + mpCam->cy;
        _error = _measurement - Vector2d(u, v);
    }

    void EdgePRIDP::linearizeOplus() {

        const VertexPointInvDepth *vIDP = dynamic_cast<const VertexPointInvDepth *>(_vertices[0]);
        const VertexPR *vPR0 = dynamic_cast<const VertexPR *>(_vertices[1]);
        const VertexPR *vPRi = dynamic_cast<const VertexPR *>(_vertices[2]);
        const VertexPR *vPRcb = dynamic_cast<const VertexPR *>(_vertices[3]);

        const Matrix3d R0 = vPR0->R();
        const Vector3d t0 = vPR0->t();
        const Matrix3d Ri = vPRi->R();
        const Matrix3d RiT = Ri.transpose();
        const Vector3d ti = vPRi->t();
        const Matrix3d Rcb = vPRcb->R();
        const Vector3d tcb = vPRcb->t();

        // point inverse depth in reference KF
        double rho = vIDP->estimate();
        if (rho < 1e-6) {
            // LOG(WARNING) << "2. rho = " << rho << ", rho<1e-6" << std::endl;
        }

        // point coordinate in reference KF, body
        Vector3d P0;
        P0 << x, y, 1;
        double d = 1.0 / rho;   // depth
        P0 *= d;

        // Pi = Rcb*Ri^T*R0*Rcb^T* p0 + ( tcb - Rcb*Ri^T*R0*Rcb^T *tcb + Rcb*Ri^T*(t0-ti) )
        const Matrix3d Rcic0 = Rcb * Ri.transpose() * R0 * Rcb.transpose();
        const Vector3d Pi = Rcic0 * P0 + tcb - Rcic0 * tcb + Rcb * Ri.transpose() * (t0 - ti);

        // err = obs - pi(Px)
        // Jx = -Jpi * dpi/dx
        double x = Pi[0];
        double y = Pi[1];
        double z = Pi[2];
        double fx = mpCam->fx;
        double fy = mpCam->fy;

        Matrix<double, 2, 3> Maux;
        Maux.setZero();
        Maux(0, 0) = fx;
        Maux(0, 1) = 0;
        Maux(0, 2) = -x / z * fx;
        Maux(1, 0) = 0;
        Maux(1, 1) = fy;
        Maux(1, 2) = -y / z * fy;
        Matrix<double, 2, 3> Jpi = Maux / z;

        // 1. J_e_rho, 2x1
        // Vector3d J_pi_rho = Rcic0 * (-d * P0);
        Vector3d J_pi_rho = Rcic0 * (-d * P0); // (xiang) this should be squared?
        _jacobianOplus[0] = -Jpi * J_pi_rho;

        // 2. J_e_pr0, 2x6
        Matrix3d J_pi_t0 = Rcb * RiT;
        Matrix3d J_pi_r0 = -Rcic0 * SO3d::hat(P0 - tcb) * Rcb;
        Matrix<double, 3, 6> J_pi_pr0;
        J_pi_pr0.topLeftCorner(3, 3) = J_pi_t0;
        J_pi_pr0.topRightCorner(3, 3) = J_pi_r0;
        _jacobianOplus[1] = -Jpi * J_pi_pr0;

        // 3. J_e_pri, 2x6
        Matrix3d J_pi_ti = -Rcb * RiT;
        Vector3d taux = RiT * (R0 * Rcb.transpose() * (P0 - tcb) + t0 - ti);
        Matrix3d J_pi_ri = Rcb * SO3d::hat(taux);
        Matrix<double, 3, 6> J_pi_pri;
        J_pi_pri.topLeftCorner(3, 3) = J_pi_ti;
        J_pi_pri.topRightCorner(3, 3) = J_pi_ri;
        _jacobianOplus[2] = -Jpi * J_pi_pri;

        // 4. J_e_prcb, 2x6
        Matrix3d J_pi_tcb = Matrix3d::Identity() - Rcic0;
        Matrix3d J_pi_rcb = -SO3d::hat(Rcic0 * (P0 - tcb)) * Rcb
                            + Rcic0 * SO3d::hat(P0 - tcb) * Rcb
                            - Rcb * SO3d::hat(RiT * (t0 - ti));
        Matrix<double, 3, 6> J_pi_prcb;
        J_pi_prcb.topLeftCorner(3, 3) = J_pi_tcb;
        J_pi_prcb.topRightCorner(3, 3) = J_pi_rcb;
        _jacobianOplus[3] = -Jpi * J_pi_prcb;
    }

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
