#include "common/G2OTypes.h"
#include "common/Camera.h"
#include "common/Frame.h"

namespace ygz {

    void EdgePRV::computeError() {

        const VertexPR *vPRi = static_cast<const VertexPR *>(_vertices[0]);
        const VertexPR *vPRj = static_cast<const VertexPR *>(_vertices[1]);
        const VertexSpeed *vVi = static_cast<const VertexSpeed *>(_vertices[2]);
        const VertexSpeed *vVj = static_cast<const VertexSpeed *>(_vertices[3]);
        const VertexGyrBias *vBiasGi = static_cast<const VertexGyrBias *>(_vertices[4]);
        const VertexAcceBias *vBiasAi = static_cast<const VertexAcceBias *>(_vertices[5]);

        // terms need to computer error in vertex i, except for bias error
        const Vector3d Pi = vPRi->t();
        const Matrix3d Ri = vPRi->R();

        const Vector3d Vi = vVi->estimate();

        // Bias from the bias vertex
        const Vector3d dBgi = vBiasGi->estimate();
        const Vector3d dBai = vBiasAi->estimate();

        // terms need to computer error in vertex j, except for bias error
        const Vector3d Pj = vPRj->t();
        const Matrix3d Rj = vPRj->R();

        const Vector3d Vj = vVj->estimate();

        // IMU Preintegration measurement
        const IMUPreIntegration &M = _measurement;

        const double dTij = M.getDeltaTime();   // Delta Time
        const double dT2 = dTij * dTij;
        const Vector3d dPij = M.getDeltaP().cast<double>();    // Delta Position pre-integration measurement
        const Vector3d dVij = M.getDeltaV().cast<double>();    // Delta Velocity pre-integration measurement
        const Matrix3d dRij = M.getDeltaR().cast<double>();    // Delta Rotation pre-integration measurement

        // tmp variable, transpose of Ri
        const Matrix3d RiT = Ri.inverse();
        // residual error of Delta Position measurement
        const Vector3d rPij = RiT * (Pj - Pi - Vi * dTij - 0.5 * GravityVec * dT2)
                              - (dPij + M.getJPBiasg() * dBgi +
                                 M.getJPBiasa() * dBai);   // this line includes correction term of bias change.

        // residual error of Delta Velocity measurement
        const Vector3d rVij = RiT * (Vj - Vi - GravityVec * dTij)
                              - (dVij + M.getJVBiasg() * dBgi +
                                 M.getJVBiasa() * dBai);   //this line includes correction term of bias change

        // residual error of Delta Rotation measurement
        const Matrix3d dR_dbg = SO3d::exp(M.getJRBiasg() * dBgi).matrix();
        const Matrix3d rRij = (dRij * dR_dbg).inverse() * RiT * Rj;
        const Vector3d rPhiij = SO3d::log(rRij);

        // 9-Dim error vector order:
        // position-velocity-rotation
        // rPij - rPhiij - rVij
        _error.segment<3>(0) = rPij;       // position error
        _error.segment<3>(3) = rPhiij;     // rotation phi error
        _error.segment<3>(6) = rVij;       // velocity error
    }

    void EdgePRV::linearizeOplus() {

        const VertexPR *vPRi = static_cast<const VertexPR *>(_vertices[0]);
        const VertexPR *vPRj = static_cast<const VertexPR *>(_vertices[1]);
        const VertexSpeed *vVi = static_cast<const VertexSpeed *>(_vertices[2]);
        const VertexSpeed *vVj = static_cast<const VertexSpeed *>(_vertices[3]);
        const VertexGyrBias *vBiasGi = static_cast<const VertexGyrBias *>(_vertices[4]);
        const VertexAcceBias *vBiasAi = static_cast<const VertexAcceBias *>(_vertices[5]);

        // terms need to computer error in vertex i, except for bias error
        const Vector3d Pi = vPRi->t();
        const Matrix3d Ri = vPRi->R();

        const Vector3d Vi = vVi->estimate();

        // Bias from the bias vertex
        const Vector3d dBgi = vBiasGi->estimate();
        const Vector3d dBai = vBiasAi->estimate();

        // terms need to computer error in vertex j, except for bias error
        const Vector3d Pj = vPRj->t();
        const Matrix3d Rj = vPRj->R();

        const Vector3d Vj = vVj->estimate();

        // IMU Preintegration measurement
        const IMUPreIntegration &M = _measurement;
        const double dTij = M.getDeltaTime();   // Delta Time
        const double dT2 = dTij * dTij;

        // some temp variable
        Matrix3d O3x3 = Matrix3d::Zero();       // 0_3x3
        Matrix3d RiT = Ri.transpose();          // Ri^T
        Matrix3d RjT = Rj.transpose();          // Rj^T
        Vector3d rPhiij = _error.segment<3>(3); // residual of rotation, rPhiij
        Matrix3d JrInv_rPhi = SO3d::JacobianRInv(rPhiij);    // inverse right jacobian of so3 term #rPhiij#
        Matrix3d J_rPhi_dbg = M.getJRBiasg();              // jacobian of preintegrated rotation-angle to gyro bias i

        // this is really messy
        // 1.
        // increment is the same as Forster 15'RSS
        // pi = pi + dpi,    pj = pj + dpj
        // Ri = Ri*Exp(dphi_i), Rj = Rj*Exp(dphi_j)
        // vi = vi + dvi,       vj = vj + dvj
        //      Note: the optimized bias term is the 'delta bias'
        // dBgi = dBgi + dbgi_update,    dBgj = dBgj + dbgj_update
        // dBai = dBai + dbai_update,    dBaj = dBaj + dbaj_update

        // 2.
        // 9-Dim error vector order in PVR:
        // position-velocity-rotation
        // rPij - rPhiij - rVij
        //      Jacobian row order:
        // J_rPij_xxx
        // J_rPhiij_xxx
        // J_rVij_xxx

        // 3.
        // order in 'update_' in PR
        // Vertex_i : dPi, dPhi_i
        // Vertex_j : dPj, dPhi_j
        // 6-Dim error vector order in Bias:
        // dBiasg_i - dBiasa_i

        // Jacobians:
        // dP/dPR0, dP/dPR1, dP/dV0, dP/dV1, dP/dBiasG, dP/dBiasG
        // dR/dPR0, dR/dPR1, dR/dV0, dR/dV1, dR/dBiasG, dR/dBiasG
        // dV/dPR0, dV/dPR1, dV/dV0, dV/dV1, dV/dBiasG, dV/dBiasG

        // 4. PR0 & V0
        // For Vertex_PR_i, J [dP;dR;dV] / [dP0 dR0]
        Matrix<double, 9, 6> JPRi;
        JPRi.setZero();
        // J_rPij_xxx_i for Vertex_PR_i
        JPRi.block<3, 3>(0, 0) = -RiT;      //J_rP_dpi
        JPRi.block<3, 3>(0, 3) = SO3d::hat(
                RiT * (Pj - Pi - Vi * dTij - 0.5 * GravityVec * dT2));    //J_rP_dPhi_i
        // J_rPhiij_xxx_i for Vertex_PR_i
        Matrix3d ExprPhiijTrans = SO3d::exp(rPhiij).inverse().matrix();
        Matrix3d JrBiasGCorr = SO3d::JacobianR(J_rPhi_dbg * dBgi);
        JPRi.block<3, 3>(3, 0) = O3x3;    //dpi
        JPRi.block<3, 3>(3, 3) = -JrInv_rPhi * RjT * Ri;    //dphi_i
        // J_rVij_xxx_i for Vertex_PVR_i
        JPRi.block<3, 3>(6, 0) = O3x3;    //dpi
        JPRi.block<3, 3>(6, 3) = SO3d::hat(RiT * (Vj - Vi - GravityVec * dTij));    //dphi_i

        // For Vertex_V_i, J [dP;dR;dV] / dV0
        Matrix<double, 9, 3> JVi;
        JVi.setZero();
        JVi.block<3, 3>(0, 0) = -RiT * dTij;  //J_rP_dvi
        JVi.block<3, 3>(3, 0) = O3x3;    //rR_dvi
        JVi.block<3, 3>(6, 0) = -RiT;    //rV_dvi

        // 5. PR1 & V1
        // For Vertex_PR_j, J [dP;dR;dV] / [dP1 dR1]
        Matrix<double, 9, 6> JPRj;
        JPRj.setZero();
        // J_rPij_xxx_j for Vertex_PR_j
        JPRj.block<3, 3>(0, 0) = RiT;  //rP_dpj
        JPRj.block<3, 3>(0, 3) = O3x3;    //rP_dphi_j
        // J_rPhiij_xxx_j for Vertex_PR_j
        JPRj.block<3, 3>(3, 0) = O3x3;    //rR_dpj
        JPRj.block<3, 3>(3, 3) = JrInv_rPhi;    //rR_dphi_j
        // J_rVij_xxx_j for Vertex_PR_j
        JPRj.block<3, 3>(6, 0) = O3x3;    //rV_dpj
        JPRj.block<3, 3>(6, 3) = O3x3;    //rV_dphi_j

        // For Vertex_V_i, J [dP;dR;dV] / dV1
        Matrix<double, 9, 3> JVj;
        JVj.setZero();
        JVj.block<3, 3>(0, 0) = O3x3;    //rP_dvj
        JVj.block<3, 3>(3, 0) = O3x3;    //rR_dvj
        JVj.block<3, 3>(6, 0) = RiT;    //rV_dvj

        // 6.
        // For Vertex_Bias_i
        Matrix<double, 9, 3> JBiasG;
        Matrix<double, 9, 3> JBiasA;
        JBiasG.setZero();
        JBiasA.setZero();

        // bias
        JBiasG.block<3, 3>(0, 0) = -M.getJPBiasg();     //J_rP_dbgi
        JBiasG.block<3, 3>(3, 0) = -JrInv_rPhi * ExprPhiijTrans * JrBiasGCorr * J_rPhi_dbg;    //dbg_i
        JBiasG.block<3, 3>(6, 0) = -M.getJVBiasg();    //dbg_i

        JBiasA.block<3, 3>(0, 0) = -M.getJPBiasa();     //J_rP_dbai
        JBiasA.block<3, 3>(3, 0) = O3x3;    //dba_i
        JBiasA.block<3, 3>(6, 0) = -M.getJVBiasa();    //dba_i

        // set all jacobians
        _jacobianOplus[0] = JPRi;
        _jacobianOplus[1] = JPRj;
        _jacobianOplus[2] = JVi;
        _jacobianOplus[3] = JVj;
        _jacobianOplus[4] = JBiasG;
        _jacobianOplus[5] = JBiasA;
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
            // 点在后面？
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
