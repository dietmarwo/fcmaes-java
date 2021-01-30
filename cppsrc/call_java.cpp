#include "keplerian_toolbox/core_functions/propagate_lagrangian.hpp"
#include "keplerian_toolbox/core_functions/propagate_lagrangian_u.hpp"
#include "keplerian_toolbox/core_functions/propagate_taylor.hpp"
#include "keplerian_toolbox/core_functions/propagate_taylor_J2.hpp"
#include "keplerian_toolbox/core_functions/propagate_taylor_disturbance.hpp"
#include "keplerian_toolbox/core_functions/propagate_taylor_jorba.hpp"
#include "keplerian_toolbox/core_functions/propagate_taylor_s.hpp"
#include "keplerian_toolbox/core_functions/three_impulses_approximation.hpp"
#include "keplerian_toolbox/core_functions/par2ic.hpp"
#include "keplerian_toolbox/core_functions/ic2par.hpp"
#include "keplerian_toolbox/core_functions/lambert_find_N.hpp"
#include "keplerian_toolbox/core_functions/closest_distance.hpp"
#include "keplerian_toolbox/planet/base.hpp"
#include "keplerian_toolbox/planet/jpl_low_precision.hpp"
#include "keplerian_toolbox/lambert_problem.hpp"
#include "PowSwingByInv.hpp"
#include <iostream>
#include <vector>
#include "fcmaes_core_Jni.h"
#include "fcmaes_examples_Jni.h"

using namespace std;

/*
 * Class:     fcmaes_core_Jni
 * Method:    lambertProblem
 * Signature: (DDDDDDDZID)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_fcmaes_core_Jni_lambertProblem(JNIEnv *env,
        jclass cls, jdouble r1x, jdouble r1y, jdouble r1z, jdouble r2x,
        jdouble r2y, jdouble r2z, jdouble tf, jboolean longWay, jint N,
        jdouble mu) {

    try {
        kep_toolbox::array3D r1;
        kep_toolbox::array3D r2;
        double vLamb[2][3];
        r1[0] = r1x;
        r1[1] = r1y;
        r1[2] = r1z;
        r2[0] = r2x;
        r2[1] = r2y;
        r2[2] = r2z;

        kep_toolbox::lambert_problem lp = kep_toolbox::lambert_problem(r1, r2,
                tf, mu, longWay ? 1 : 0, N);

        std::vector<kep_toolbox::array3D> v1s = lp.get_v1();
        std::vector<kep_toolbox::array3D> v2s = lp.lambert_problem::get_v2();
        std::vector<int> iters = lp.lambert_problem::get_iters();

        int size = 6 * v1s.size();
        jdoubleArray jres = env->NewDoubleArray(size);
        double *dres = env->GetDoubleArrayElements(jres, JNI_FALSE);
        int i = 0;
        for (int j = 0; j < v1s.size(); j++) {
            dres[i++] = v1s[j][0];
            dres[i++] = v1s[j][1];
            dres[i++] = v1s[j][2];
            dres[i++] = v2s[j][0];
            dres[i++] = v2s[j][1];
            dres[i++] = v2s[j][2];
        }
        env->SetDoubleArrayRegion(jres, 0, size, (jdouble*) dres);
        env->ReleaseDoubleArrayElements(jres, dres, 0);
        return jres;

    } catch (std::exception &e) {
        cout << e.what() << endl;
        return NULL;
    }
}

/*
 * Class:     fcmaes_core_Jni
 * Method:    powSwingByInvC
 * Signature: (DDD)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_fcmaes_core_Jni_powSwingByInvC(JNIEnv *env,
        jclass cls, jdouble vin, jdouble vout, jdouble alpha) {
    try {
        jdoubleArray jres = env->NewDoubleArray(2);
        double *dres = env->GetDoubleArrayElements(jres, JNI_FALSE);
        PowSwingByInv(vin, vout, alpha, dres[0], dres[1]);
        env->SetDoubleArrayRegion(jres, 0, 2, (jdouble*) dres);
        env->ReleaseDoubleArrayElements(jres, dres, 0);
        return jres;
    } catch (std::exception &e) {
        cout << e.what() << endl;
        return NULL;
    }
}

void propagateJOR(const double *r0_in, const double *v0_in, const double *c0_in,
        const double &dt, const double &m0, const double &mu,
        const double &veff, const double &log10tolerance,
        const double &log10rtolerance, double &m, double *r, double *v) {
    kep_toolbox::array3D r0;
    kep_toolbox::array3D v0;
    kep_toolbox::array3D c0;
    for (int i = 0; i < 3; i++) {
        r0[i] = r0_in[i];
        v0[i] = v0_in[i];
        c0[i] = c0_in[i];
    }
    m = m0;
    kep_toolbox::propagate_taylor_jorba(r0, v0, m, c0, dt, mu, veff,
            log10tolerance, log10rtolerance);
    for (int i = 0; i < 3; i++) {
        r[i] = r0[i];
        v[i] = v0[i];
    }
}

void propagateTAY(const double *r0_in, const double *v0_in, const double *c0_in,
        const double &dt, const double &m0, const double &mu,
        const double &veff, const double &log10tolerance,
        const double &log10rtolerance, double &m, double *r, double *v) {
    kep_toolbox::array3D r0;
    kep_toolbox::array3D v0;
    kep_toolbox::array3D c0;
    for (int i = 0; i < 3; i++) {
        r0[i] = r0_in[i];
        v0[i] = v0_in[i];
        c0[i] = c0_in[i];
    }
    m = m0;
    kep_toolbox::propagate_taylor(r0, v0, m, c0, dt, mu, veff, log10tolerance,
            log10rtolerance);
    for (int i = 0; i < 3; i++) {
        r[i] = r0[i];
        v[i] = v0[i];
    }
}

void propagateTAYJ2(const double *r0_in, const double *v0_in,
        const double *c0_in, const double &dt, const double &m0,
        const double &mu, const double &veff, const double &j2rg2,
        const double &log10tolerance, const double &log10rtolerance, double &m,
        double *r, double *v) {
    kep_toolbox::array3D r0;
    kep_toolbox::array3D v0;
    kep_toolbox::array3D c0;
    for (int i = 0; i < 3; i++) {
        r0[i] = r0_in[i];
        v0[i] = v0_in[i];
        c0[i] = c0_in[i];
    }
    m = m0;
    kep_toolbox::propagate_taylor_J2(r0, v0, m, c0, dt, mu, veff, j2rg2,
            log10tolerance, log10rtolerance);
    for (int i = 0; i < 3; i++) {
        r[i] = r0[i];
        v[i] = v0[i];
    }
}

void propagateTAYDIST(const double *r0_in, const double *v0_in,
        const double *c0_in, const double *d0_in, const double &dt,
        const double &m0, const double &mu, const double &veff,
        const double &log10tolerance, const double &log10rtolerance, double &m,
        double *r, double *v) {
    kep_toolbox::array3D r0;
    kep_toolbox::array3D v0;
    kep_toolbox::array3D c0;
    kep_toolbox::array3D d0;
    for (int i = 0; i < 3; i++) {
        r0[i] = r0_in[i];
        v0[i] = v0_in[i];
        c0[i] = c0_in[i];
        d0[i] = d0_in[i];
    }
    m = m0;
    kep_toolbox::propagate_taylor_disturbance(r0, v0, m, c0, d0, dt, mu, veff,
            log10tolerance, log10rtolerance);
    for (int i = 0; i < 3; i++) {
        r[i] = r0[i];
        v[i] = v0[i];
    }
}

void propagateLAG(const double *r0_in, const double *v0_in, const double &dt,
        const double &mu, double *r, double *v) {
    kep_toolbox::array3D r0;
    kep_toolbox::array3D v0;
    for (int i = 0; i < 3; i++) {
        r0[i] = r0_in[i];
        v0[i] = v0_in[i];
    }
    kep_toolbox::propagate_lagrangian(r0, v0, dt, mu);
    for (int i = 0; i < 3; i++) {
        r[i] = r0[i];
        v[i] = v0[i];
    }
}

void propagateULAG(const double *r0_in, const double *v0_in, const double &dt,
        const double &mu, double *r, double *v) {
    kep_toolbox::array3D r0;
    kep_toolbox::array3D v0;
    for (int i = 0; i < 3; i++) {
        r0[i] = r0_in[i];
        v0[i] = v0_in[i];
    }
    kep_toolbox::propagate_lagrangian_u(r0, v0, dt, mu);
    for (int i = 0; i < 3; i++) {
        r[i] = r0[i];
        v[i] = v0[i];
    }
}

void propagateTAYmulti(const double *r0_in, const double *v0_in,
        const double *ts, const int &num, const double *cs_in, const double &m0,
        const double &mu, const double &veff, const double &log10tolerance,
        const double &log10rtolerance, double &m, double *r, double *v) {

    double r0[3], v0[3], c0[3];
    for (int j = 0; j < 3; j++) {
        r0[j] = r0_in[j];
        v0[j] = v0_in[j];
    }
    m = m0;
    double dt, m1, r1[3], v1[3];
    for (int i = 0; i < num; i++) {
        dt = ts[i];
        for (int j = 0; j < 3; j++)
            c0[j] = cs_in[i * 3 + j];
        propagateTAY(r0, v0, c0, dt, m, mu, veff, log10tolerance,
                log10rtolerance, m1, r1, v1);
        m = m1;
        for (int j = 0; j < 3; j++) {
            r0[j] = r1[j];
            v0[j] = v1[j];
        }
    }
    for (int j = 0; j < 3; j++) {
        r[j] = r1[j];
        v[j] = v1[j];
    }
}

void propagateTAYJ2multi(const double *r0_in, const double *v0_in,
        const double *ts, const int &num, const double *cs_in, const double &m0,
        const double &mu, const double &veff, const double &j2rg2,
        const double &log10tolerance, const double &log10rtolerance, double &m,
        double *r, double *v) {

    double r0[3], v0[3], c0[3];
    for (int j = 0; j < 3; j++) {
        r0[j] = r0_in[j];
        v0[j] = v0_in[j];
    }
    m = m0;
    double dt, m1, r1[3], v1[3];
    for (int i = 0; i < num; i++) {
        dt = ts[i];
        for (int j = 0; j < 3; j++)
            c0[j] = cs_in[i * 3 + j];
        propagateTAYJ2(r0, v0, c0, dt, m, mu, veff, j2rg2, log10tolerance,
                log10rtolerance, m1, r1, v1);
        m = m1;
        for (int j = 0; j < 3; j++) {
            r0[j] = r1[j];
            v0[j] = v1[j];
        }
    }
    for (int j = 0; j < 3; j++) {
        r[j] = r1[j];
        v[j] = v1[j];
    }
}

void propagateTAYDISTmulti(const double *r0_in, const double *v0_in,
        const double *ts, const int &num, const double *cs_in,
        const double *ds_in, const double &m0, const double &mu,
        const double &veff, const double &log10tolerance,
        const double &log10rtolerance, double &m, double *r, double *v) {

    double r0[3], v0[3], c0[3], d0[3];
    for (int j = 0; j < 3; j++) {
        r0[j] = r0_in[j];
        v0[j] = v0_in[j];
    }
    m = m0;
    double dt, m1, r1[3], v1[3];
    for (int i = 0; i < num; i++) {
        dt = ts[i];
        for (int j = 0; j < 3; j++) {
            c0[j] = cs_in[i * 3 + j];
            d0[j] = ds_in[i * 3 + j];
        }
        propagateTAYDIST(r0, v0, c0, d0, dt, m, mu, veff, log10tolerance,
                log10rtolerance, m1, r1, v1);
        m = m1;
        for (int j = 0; j < 3; j++) {
            r0[j] = r1[j];
            v0[j] = v1[j];
        }
    }
    for (int j = 0; j < 3; j++) {
        r[j] = r1[j];
        v[j] = v1[j];
    }
}

void propagateJORmulti(const double *r0_in, const double *v0_in,
        const double *ts, const int &num, const double *cs_in, const double &m0,
        const double &mu, const double &veff, const double &log10tolerance,
        const double &log10rtolerance, double &m, double *r, double *v) {

    double r0[3], v0[3], c0[3];
    for (int j = 0; j < 3; j++) {
        r0[j] = r0_in[j];
        v0[j] = v0_in[j];
    }
    m = m0;
    double dt, m1, r1[3], v1[3];
    for (int i = 0; i < num; i++) {
        dt = ts[i];
        for (int j = 0; j < 3; j++)
            c0[j] = cs_in[i * 3 + j];
        propagateJOR(r0, v0, c0, dt, m, mu, veff, log10tolerance,
                log10rtolerance, m1, r1, v1);
        m = m1;
        for (int j = 0; j < 3; j++) {
            r0[j] = r1[j];
            v0[j] = v1[j];
        }
    }
    for (int j = 0; j < 3; j++) {
        r[j] = r1[j];
        v[j] = v1[j];
    }
}

void propagateLAGmulti(const double *r0_in, const double *v0_in,
        const double *ts, const int &num, const double *dvs, const double &mu,
        double *r, double *v) {
    double r0[3], v0[3];
    for (int j = 0; j < 3; j++) {
        r0[j] = r0_in[j];
        v0[j] = v0_in[j];
    }
    double t, r1[3], v1[3];
    for (int i = 0; i < num; i++) {
        t = ts[i];
        //propagate to impulse
        double dt = 0.5 * t;
        if (i > 0)
            dt += 0.5 * ts[i - 1];
        propagateLAG(r0, v0, dt, mu, r1, v1);
        for (int j = 0; j < 3; j++) {
            r0[j] = r1[j];
            v0[j] = v1[j] + dvs[i * 3 + j];
        }
    }
    propagateLAG(r0, v0, 0.5 * t, mu, r1, v1);
    for (int j = 0; j < 3; j++) {
        r[j] = r1[j];
        v[j] = v1[j];
    }
}

void propagateULAGmulti(const double *r0_in, const double *v0_in,
        const double *ts, const int &num, const double *dvs, const double &mu,
        double *r, double *v) {
    double r0[3], v0[3];
    for (int j = 0; j < 3; j++) {
        r0[j] = r0_in[j];
        v0[j] = v0_in[j];
    }
    double t, r1[3], v1[3];
    for (int i = 0; i < num; i++) {
        t = ts[i];
        //propagate to impulse
        double dt = 0.5 * t;
        if (i > 0)
            dt += 0.5 * ts[i - 1];
        propagateULAG(r0, v0, dt, mu, r1, v1);
        for (int j = 0; j < 3; j++) {
            r0[j] = r1[j];
            v0[j] = v1[j] + dvs[i * 3 + j];
        }
    }
    propagateULAG(r0, v0, 0.5 * t, mu, r1, v1);
    for (int j = 0; j < 3; j++) {
        r[j] = r1[j];
        v[j] = v1[j];
    }
}

void planetEph(const int pli, const double mjd2000, double *r, double *v) {
    std::string pstr;
    switch (pli) {
    case (1): {
        pstr = "mercury";
    }
        break;
    case (2): {
        pstr = "venus";
    }
        break;
    case (3): {
        pstr = "earth";
    }
        break;
    case (4): {
        pstr = "mars";
    }
        break;
    case (5): {
        pstr = "jupiter";
    }
        break;
    case (6): {
        pstr = "saturn";
    }
        break;
    case (7): {
        pstr = "uranus";
    }
        break;
    case (8): {
        pstr = "neptune";
    }
        break;
    case (9): {
        pstr = "pluto";
    }
        break;
    default: {
        throw_value_error(std::string("unknown planet index"));
    }
    }
    kep_toolbox::planet::jpl_lp pl(pstr);
    kep_toolbox::array3D r1, v1;
    pl.eph(mjd2000, r1, v1);
    for (int j = 0; j < 3; j++) {
        r[j] = r1[j];
        v[j] = v1[j];
    }
}

/*
 * Class:     fcmaes_core_Jni
 * Method:    planetEplC
 * Signature: (ID[D[D)V
 */
JNIEXPORT void JNICALL Java_fcmaes_core_Jni_planetEplC
(JNIEnv *env, jclass cls, jint pli, jdouble mjd2000, jdoubleArray r, jdoubleArray v) {

    double* r1 = env->GetDoubleArrayElements(r, JNI_FALSE);
    double* v1 = env->GetDoubleArrayElements(v, JNI_FALSE);

    planetEph(pli, mjd2000, r1, v1);

    env->SetDoubleArrayRegion (r, 0, 3, (jdouble*)r1);
    env->SetDoubleArrayRegion (v, 0, 3, (jdouble*)v1);

    env->ReleaseDoubleArrayElements(r, r1, 0);
    env->ReleaseDoubleArrayElements(v, v1, 0);
}

/*
 * Class:     fcmaes_core_Jni
 * Method:    propagateLAGmultiC
 * Signature: ([D[D[DI[DD[D[D)V
 */
JNIEXPORT void JNICALL Java_fcmaes_core_Jni_propagateLAGmultiC
(JNIEnv *env, jclass cls, jdoubleArray r0_in, jdoubleArray v0_in, jdoubleArray ts_in,
        jint num, jdoubleArray dvs_in,
        jdouble mu, jdoubleArray r, jdoubleArray v) {

    double* r0 = env->GetDoubleArrayElements(r0_in, JNI_FALSE);
    double* v0 = env->GetDoubleArrayElements(v0_in, JNI_FALSE);
    double* ts = env->GetDoubleArrayElements(ts_in, JNI_FALSE);
    double* dvs = env->GetDoubleArrayElements(dvs_in, JNI_FALSE);
    double* r1 = env->GetDoubleArrayElements(r, JNI_FALSE);
    double* v1 = env->GetDoubleArrayElements(v, JNI_FALSE);

    propagateLAGmulti(r0, v0, ts, num, dvs, mu, r1, v1);

    env->SetDoubleArrayRegion (r, 0, 3, (jdouble*)r1);
    env->SetDoubleArrayRegion (v, 0, 3, (jdouble*)v1);

    env->ReleaseDoubleArrayElements(r0_in, r0, 0);
    env->ReleaseDoubleArrayElements(v0_in, v0, 0);
    env->ReleaseDoubleArrayElements(ts_in, ts, 0);
    env->ReleaseDoubleArrayElements(dvs_in, dvs, 0);
    env->ReleaseDoubleArrayElements(r, r1, 0);
    env->ReleaseDoubleArrayElements(v, v1, 0);
}

/*
 * Class:     fcmaes_core_Jni
 * Method:    propagateJORmultiC
 * Signature: ([D[D[DI[DDDDDD[D[D)D
 */
JNIEXPORT jdouble JNICALL Java_fcmaes_core_Jni_propagateJORmultiC(JNIEnv *env,
        jclass cls, jdoubleArray r0_in, jdoubleArray v0_in, jdoubleArray ts_in,
        jint num, jdoubleArray cs_in, jdouble m0, jdouble mu, jdouble veff,
        jdouble log10tolerance, jdouble log10rtolerance, jdoubleArray r,
        jdoubleArray v) {

    double *r0 = env->GetDoubleArrayElements(r0_in, JNI_FALSE);
    double *v0 = env->GetDoubleArrayElements(v0_in, JNI_FALSE);
    double *ts = env->GetDoubleArrayElements(ts_in, JNI_FALSE);
    double *cs = env->GetDoubleArrayElements(cs_in, JNI_FALSE);
    double *r1 = env->GetDoubleArrayElements(r, JNI_FALSE);
    double *v1 = env->GetDoubleArrayElements(v, JNI_FALSE);

    double m;
    propagateJORmulti(r0, v0, ts, num, cs, m0, mu, veff, log10tolerance,
            log10rtolerance, m, r1, v1);

    env->SetDoubleArrayRegion(r, 0, 3, (jdouble*) r1);
    env->SetDoubleArrayRegion(v, 0, 3, (jdouble*) v1);

    env->ReleaseDoubleArrayElements(r0_in, r0, 0);
    env->ReleaseDoubleArrayElements(v0_in, v0, 0);
    env->ReleaseDoubleArrayElements(ts_in, ts, 0);
    env->ReleaseDoubleArrayElements(cs_in, cs, 0);
    env->ReleaseDoubleArrayElements(r, r1, 0);
    env->ReleaseDoubleArrayElements(v, v1, 0);
    return m;
}

/*
 * Class:     fcmaes_core_Jni
 * Method:    propagateTAYC
 * Signature: ([D[DD[DDDDDD[D[D)D
 */
JNIEXPORT jdouble JNICALL Java_fcmaes_core_Jni_propagateTAYC(JNIEnv *env,
        jclass cls, jdoubleArray r0_in, jdoubleArray v0_in, jdouble dt,
        jdoubleArray c0_in, jdouble m0, jdouble mu, jdouble veff,
        jdouble log10tolerance, jdouble log10rtolerance, jdoubleArray r,
        jdoubleArray v) {

    double *r0 = env->GetDoubleArrayElements(r0_in, JNI_FALSE);
    double *v0 = env->GetDoubleArrayElements(v0_in, JNI_FALSE);
    double *c0 = env->GetDoubleArrayElements(c0_in, JNI_FALSE);
    double *r1 = env->GetDoubleArrayElements(r, JNI_FALSE);
    double *v1 = env->GetDoubleArrayElements(v, JNI_FALSE);

    double m;
    propagateTAY(r0, v0, c0, dt, m0, mu, veff, log10tolerance, log10rtolerance,
            m, r1, v1);

    env->SetDoubleArrayRegion(r, 0, 3, (jdouble*) r1);
    env->SetDoubleArrayRegion(v, 0, 3, (jdouble*) v1);

    env->ReleaseDoubleArrayElements(r0_in, r0, 0);
    env->ReleaseDoubleArrayElements(v0_in, v0, 0);
    env->ReleaseDoubleArrayElements(c0_in, c0, 0);
    env->ReleaseDoubleArrayElements(r, r1, 0);
    env->ReleaseDoubleArrayElements(v, v1, 0);
    return m;
}

/*
 * Class:     fcmaes_core_Jni
 * Method:    propagateTAYDISTC
 * Signature: ([D[DD[D[DDDDDD[D[D)D
 */
JNIEXPORT jdouble JNICALL Java_fcmaes_core_Jni_propagateTAYDISTC(JNIEnv *env,
        jclass cls, jdoubleArray r0_in, jdoubleArray v0_in, jdouble dt,
        jdoubleArray c0_in, jdoubleArray d0_in, jdouble m0, jdouble mu,
        jdouble veff, jdouble log10tolerance, jdouble log10rtolerance,
        jdoubleArray r, jdoubleArray v) {

    double *r0 = env->GetDoubleArrayElements(r0_in, JNI_FALSE);
    double *v0 = env->GetDoubleArrayElements(v0_in, JNI_FALSE);
    double *c0 = env->GetDoubleArrayElements(c0_in, JNI_FALSE);
    double *d0 = env->GetDoubleArrayElements(d0_in, JNI_FALSE);
    double *r1 = env->GetDoubleArrayElements(r, JNI_FALSE);
    double *v1 = env->GetDoubleArrayElements(v, JNI_FALSE);

    double m;
    propagateTAYDIST(r0, v0, c0, d0, dt, m0, mu, veff, log10tolerance,
            log10rtolerance, m, r1, v1);

    env->SetDoubleArrayRegion(r, 0, 3, (jdouble*) r1);
    env->SetDoubleArrayRegion(v, 0, 3, (jdouble*) v1);

    env->ReleaseDoubleArrayElements(r0_in, r0, 0);
    env->ReleaseDoubleArrayElements(v0_in, v0, 0);
    env->ReleaseDoubleArrayElements(c0_in, c0, 0);
    env->ReleaseDoubleArrayElements(d0_in, d0, 0);
    env->ReleaseDoubleArrayElements(r, r1, 0);
    env->ReleaseDoubleArrayElements(v, v1, 0);
    return m;
}

/*
 * Class:     fcmaes_core_Jni
 * Method:    propagateTAYJ2C
 * Signature: ([D[DD[DDDDDDD[D[D)D
 */
JNIEXPORT jdouble JNICALL Java_fcmaes_core_Jni_propagateTAYJ2C(JNIEnv *env,
        jclass cls, jdoubleArray r0_in, jdoubleArray v0_in, jdouble dt,
        jdoubleArray c0_in, jdouble m0, jdouble mu, jdouble veff, jdouble j2rg2,
        jdouble log10tolerance, jdouble log10rtolerance, jdoubleArray r,
        jdoubleArray v) {

    double *r0 = env->GetDoubleArrayElements(r0_in, JNI_FALSE);
    double *v0 = env->GetDoubleArrayElements(v0_in, JNI_FALSE);
    double *c0 = env->GetDoubleArrayElements(c0_in, JNI_FALSE);
    double *r1 = env->GetDoubleArrayElements(r, JNI_FALSE);
    double *v1 = env->GetDoubleArrayElements(v, JNI_FALSE);

    double m;
    propagateTAYJ2(r0, v0, c0, dt, m0, mu, veff, j2rg2, log10tolerance,
            log10rtolerance, m, r1, v1);

    env->SetDoubleArrayRegion(r, 0, 3, (jdouble*) r1);
    env->SetDoubleArrayRegion(v, 0, 3, (jdouble*) v1);

    env->ReleaseDoubleArrayElements(r0_in, r0, 0);
    env->ReleaseDoubleArrayElements(v0_in, v0, 0);
    env->ReleaseDoubleArrayElements(c0_in, c0, 0);
    env->ReleaseDoubleArrayElements(r, r1, 0);
    env->ReleaseDoubleArrayElements(v, v1, 0);
    return m;
}

/*
 * Class:     fcmaes_core_Jni
 * Method:    propagateTAYmultiC
 * Signature: ([D[D[DI[DDDDDD[D[D)D
 */
JNIEXPORT jdouble JNICALL Java_fcmaes_core_Jni_propagateTAYmultiC(JNIEnv *env,
        jclass cls, jdoubleArray r0_in, jdoubleArray v0_in, jdoubleArray ts_in,
        jint num, jdoubleArray cs_in, jdouble m0, jdouble mu, jdouble veff,
        jdouble log10tolerance, jdouble log10rtolerance, jdoubleArray r,
        jdoubleArray v) {

    double *r0 = env->GetDoubleArrayElements(r0_in, JNI_FALSE);
    double *v0 = env->GetDoubleArrayElements(v0_in, JNI_FALSE);
    double *ts = env->GetDoubleArrayElements(ts_in, JNI_FALSE);
    double *cs = env->GetDoubleArrayElements(cs_in, JNI_FALSE);
    double *r1 = env->GetDoubleArrayElements(r, JNI_FALSE);
    double *v1 = env->GetDoubleArrayElements(v, JNI_FALSE);

    double m;
    propagateTAYmulti(r0, v0, ts, num, cs, m0, mu, veff, log10tolerance,
            log10rtolerance, m, r1, v1);

    env->SetDoubleArrayRegion(r, 0, 3, (jdouble*) r1);
    env->SetDoubleArrayRegion(v, 0, 3, (jdouble*) v1);

    env->ReleaseDoubleArrayElements(r0_in, r0, 0);
    env->ReleaseDoubleArrayElements(v0_in, v0, 0);
    env->ReleaseDoubleArrayElements(ts_in, ts, 0);
    env->ReleaseDoubleArrayElements(cs_in, cs, 0);
    env->ReleaseDoubleArrayElements(r, r1, 0);
    env->ReleaseDoubleArrayElements(v, v1, 0);
    return m;
}

/*
 * Class:     fcmaes_core_Jni
 * Method:    propagateULAGmultiC
 * Signature: ([D[D[DI[DD[D[D)V
 */
JNIEXPORT void JNICALL Java_fcmaes_core_Jni_propagateULAGmultiC
(JNIEnv *env, jclass cls, jdoubleArray r0_in, jdoubleArray v0_in, jdoubleArray ts_in,
        jint num, jdoubleArray dvs_in,
        jdouble mu, jdoubleArray r, jdoubleArray v) {

    double* r0 = env->GetDoubleArrayElements(r0_in, JNI_FALSE);
    double* v0 = env->GetDoubleArrayElements(v0_in, JNI_FALSE);
    double* ts = env->GetDoubleArrayElements(ts_in, JNI_FALSE);
    double* dvs = env->GetDoubleArrayElements(dvs_in, JNI_FALSE);
    double* r1 = env->GetDoubleArrayElements(r, JNI_FALSE);
    double* v1 = env->GetDoubleArrayElements(v, JNI_FALSE);

    propagateULAGmulti(r0, v0, ts, num, dvs, mu, r1, v1);

    env->SetDoubleArrayRegion (r, 0, 3, (jdouble*)r1);
    env->SetDoubleArrayRegion (v, 0, 3, (jdouble*)v1);

    env->ReleaseDoubleArrayElements(r0_in, r0, 0);
    env->ReleaseDoubleArrayElements(v0_in, v0, 0);
    env->ReleaseDoubleArrayElements(ts_in, ts, 0);
    env->ReleaseDoubleArrayElements(dvs_in, dvs, 0);
    env->ReleaseDoubleArrayElements(r, r1, 0);
    env->ReleaseDoubleArrayElements(v, v1, 0);
}

/*
 * Class:     fcmaes_core_Jni
 * Method:    iC2parC
 * Signature: ([D[DD[D)V
 */
JNIEXPORT void JNICALL Java_fcmaes_core_Jni_iC2parC
(JNIEnv *env, jclass cls, jdoubleArray r0, jdoubleArray v0, jdouble mu, jdoubleArray E) {

    double* r = env->GetDoubleArrayElements(r0, JNI_FALSE);
    double* v = env->GetDoubleArrayElements(v0, JNI_FALSE);
    double* Ec = env->GetDoubleArrayElements(E, JNI_FALSE);
    kep_toolbox::array3D rk;
    kep_toolbox::array3D vk;
    for (int i = 0; i < 3; i++) {
        rk[i] = r[i];
        vk[i] = v[i];
    }
    //IC2par(r, v, mu, Ec);

    kep_toolbox::ic2par(rk, vk, mu, Ec);

    env->SetDoubleArrayRegion (E, 0, 3, (jdouble*)Ec);

    env->ReleaseDoubleArrayElements(r0, r, 0);
    env->ReleaseDoubleArrayElements(v0, v, 0);
    env->ReleaseDoubleArrayElements(E, Ec, 0);
}

/*
 * Class:     fcmaes_core_Jni
 * Method:    par2IC
 * Signature: ([DD[D[D)V
 */
JNIEXPORT void JNICALL Java_fcmaes_core_Jni_par2IC
(JNIEnv *env, jclass cls, jdoubleArray E, jdouble mu, jdoubleArray r0, jdoubleArray v0) {

    double* Ec = env->GetDoubleArrayElements(E, JNI_FALSE);
    double* r = env->GetDoubleArrayElements(r0, JNI_FALSE);
    double* v = env->GetDoubleArrayElements(v0, JNI_FALSE);

    //par2IC(Ec, mu, r, v);
    kep_toolbox::par2ic(Ec, mu, r, v);
    //kep_toolbox::array6D arr6;

    env->SetDoubleArrayRegion (r0, 0, 3, (jdouble*)r);
    env->SetDoubleArrayRegion (v0, 0, 3, (jdouble*)v);

    env->ReleaseDoubleArrayElements(E, Ec, 0);
    env->ReleaseDoubleArrayElements(r0, r, 0);
    env->ReleaseDoubleArrayElements(v0, v, 0);
}

/*
 * Class:     fcmaes_core_Jni
 * Method:    closest_distance
 * Signature: ([D[D[D[D[DD)V
 */
JNIEXPORT void JNICALL Java_fcmaes_core_Jni_closest_1distance
(JNIEnv *env, jclass cls, jdoubleArray result, jdoubleArray r0, jdoubleArray v0, jdoubleArray r1, jdoubleArray v1, jdouble mu) {

    // result = jdouble d_min, jdouble ra

    double* resultC = env->GetDoubleArrayElements(result, JNI_FALSE);
    double* r0C = env->GetDoubleArrayElements(r0, JNI_FALSE);
    double* v0C = env->GetDoubleArrayElements(v0, JNI_FALSE);
    double* r1C = env->GetDoubleArrayElements(r1, JNI_FALSE);
    double* v1C = env->GetDoubleArrayElements(v1, JNI_FALSE);
    kep_toolbox::closest_distance(resultC[0], resultC[1], r0C, v0C, r1C, v1C, mu);
    env->ReleaseDoubleArrayElements(result, resultC, 0);
    env->ReleaseDoubleArrayElements(r0, r0C, 0);
    env->ReleaseDoubleArrayElements(v0, v0C, 0);
    env->ReleaseDoubleArrayElements(r0, r0C, 0);
    env->ReleaseDoubleArrayElements(v0, v0C, 0);
}

/*
 * Class:     fcmaes_core_Jni
 * Method:    lambert_find_N
 * Signature: (DDDI)I
 */
JNIEXPORT jint JNICALL Java_fcmaes_core_Jni_lambert_1find_1N(JNIEnv *env,
        jclass cls, jdouble s, jdouble c, jdouble tof, jint lw) {
    return kep_toolbox::lambert_find_N(s, c, tof, lw);
}

/*
 * Class:     fcmaes_core_Jni
 * Method:    propagate_lagrangian
 * Signature: ([D[DDD)V
 */
JNIEXPORT void JNICALL Java_fcmaes_core_Jni_propagate_1lagrangian
(JNIEnv *env, jclass cls, jdoubleArray r0, jdoubleArray v0, jdouble t, jdouble mu) {
    double* r0C = env->GetDoubleArrayElements(r0, JNI_FALSE);
    double* v0C = env->GetDoubleArrayElements(v0, JNI_FALSE);
    kep_toolbox::propagate_lagrangian(r0C, v0C, t, mu);
    env->SetDoubleArrayRegion (r0, 0, 3, (jdouble*)r0C);
    env->SetDoubleArrayRegion (v0, 0, 3, (jdouble*)v0C);
    env->ReleaseDoubleArrayElements(r0, r0C, 0);
    env->ReleaseDoubleArrayElements(v0, v0C, 0);
}

/*
 * Class:     fcmaes_core_Jni
 * Method:    propagate_lagrangian_u
 * Signature: ([D[DDD)V
 */
JNIEXPORT void JNICALL Java_fcmaes_core_Jni_propagate_1lagrangian_1u
(JNIEnv *env, jclass cls, jdoubleArray r0, jdoubleArray v0, jdouble t, jdouble mu) {
    double* r0C = env->GetDoubleArrayElements(r0, JNI_FALSE);
    double* v0C = env->GetDoubleArrayElements(v0, JNI_FALSE);
    kep_toolbox::array3D r0A;
    kep_toolbox::array3D v0A;
    for (int i = 0; i < 3; i++) {
        r0A[i] = r0C[i];
        v0A[i] = v0C[i];
    }
    try {
        kep_toolbox::propagate_lagrangian_u(r0A, v0A, t, mu);
    } catch (std::exception& e) {
        cout << "errör" << e.what() << endl;
    }

    for (int i = 0; i < 3; i++) {
        r0C[i] = r0A[i];
        v0C[i] = v0A[i];
    }
    env->SetDoubleArrayRegion (r0, 0, 3, (jdouble*)r0C);
    env->SetDoubleArrayRegion (v0, 0, 3, (jdouble*)v0C);
    env->ReleaseDoubleArrayElements(r0, r0C, 0);
    env->ReleaseDoubleArrayElements(v0, v0C, 0);
}

/*
 * Class:     fcmaes_core_Jni
 * Method:    three_impulses_approx
 * Signature: (DDDDDDDDD)D
 */
JNIEXPORT jdouble JNICALL Java_fcmaes_core_Jni_three_1impulses_1approx(
        JNIEnv *env, jclass cls, jdouble a1, jdouble e1, jdouble i1, jdouble W1,
        jdouble a2, jdouble e2, jdouble i2, jdouble W2, jdouble mu) {
    return kep_toolbox::three_impulses_approx(a1, e1, i1, W1, a2, e2, i2, W2,
            mu);
}

