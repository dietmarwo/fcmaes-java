#include "keplerian_toolbox/core_functions/propagate_lagrangian.hpp"
#include "keplerian_toolbox/core_functions/propagate_taylor.hpp"
#include "keplerian_toolbox/core_functions/propagate_taylor_J2.hpp"
#include "keplerian_toolbox/core_functions/par2ic.hpp"
#include "keplerian_toolbox/core_functions/ic2par.hpp"
#include "keplerian_toolbox/core_functions/fb_vel.hpp"
#include "keplerian_toolbox/core_functions/fb_prop.hpp"
#include "keplerian_toolbox/core_functions/lambert_find_N.hpp"
#include "keplerian_toolbox/planet/base.hpp"
#include "keplerian_toolbox/planet/jpl_low_precision.hpp"
#include "keplerian_toolbox/lambert_problem.hpp"
#include <iostream>
#include <vector>
#include "gtopx.cpp"
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

std::string getPlanet(int pli) {
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
    return pstr;
}

void planetEph(const int pli, const double mjd2000, double *r, double *v) {
    kep_toolbox::array3D r1, v1;
    kep_toolbox::planet::jpl_lp pl(getPlanet(pli));
    pl.eph(mjd2000, r1, v1);
    for (int j = 0; j < 3; j++) {
        r[j] = r1[j];
        v[j] = v1[j];
    }
}

void planetData(const int pli, const double mjd2000, double *data) {
    kep_toolbox::planet::jpl_lp pl(getPlanet(pli));
    data[0] = pl.compute_period(mjd2000);
    data[1] = pl.get_mu_central_body();
    data[2] = pl.get_mu_self();
    data[3] = pl.get_radius();
    data[4] = pl.get_safe_radius();
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
 * Method:    planetDataC
 * Signature: (ID[D)V
 */
JNIEXPORT void JNICALL Java_fcmaes_core_Jni_planetDataC
(JNIEnv *env, jclass cls, jint pli, jdouble mjd2000, jdoubleArray d) {

    double* data = env->GetDoubleArrayElements(d, JNI_FALSE);

    planetData(pli, mjd2000, data);

    env->SetDoubleArrayRegion (d, 0, 5, (jdouble*)data);
    env->ReleaseDoubleArrayElements(d,data, 0);
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
 * Method:    fb_prop
 * Signature: ([D[DDDD[D)V
 */
JNIEXPORT void JNICALL Java_fcmaes_core_Jni_fb_1prop
  (JNIEnv *env, jclass cls, jdoubleArray v_in, jdoubleArray v_pla, jdouble rp, jdouble beta,
          jdouble mu, jdoubleArray v_out) {

    double* vin = env->GetDoubleArrayElements(v_in, JNI_FALSE);
    double* vpla = env->GetDoubleArrayElements(v_pla, JNI_FALSE);
    double* vout = env->GetDoubleArrayElements(v_out, JNI_FALSE);

    kep_toolbox::array3D vin_a;
    kep_toolbox::array3D vpla_a;
    for (int i = 0; i < 3; i++) {
        vin_a[i] = vin[i];
        vpla_a[i] = vpla[i];
    }
    kep_toolbox::array3D vout_a;

    kep_toolbox::fb_prop(vout_a, vin_a, vpla_a, rp, beta, mu);
    for (int i = 0; i < 3; i++)
        vout[i] = vout_a[i];

    env->SetDoubleArrayRegion (v_out, 0, 3, (jdouble*)vout);

    env->ReleaseDoubleArrayElements(v_in, vin, 0);
    env->ReleaseDoubleArrayElements(v_pla, vpla, 0);
    env->ReleaseDoubleArrayElements(v_out, vout, 0);
}

/*
 * Class:     fcmaes_core_Jni
 * Method:    fb_vel
 * Signature: ([D[DI)D
 */
JNIEXPORT jdouble JNICALL Java_fcmaes_core_Jni_fb_1vel
  (JNIEnv *env, jclass cls, jdoubleArray v_rel_in, jdoubleArray v_rel_out, jint pli) {
    double* vrelin = env->GetDoubleArrayElements(v_rel_in, JNI_FALSE);
    double* vrelout = env->GetDoubleArrayElements(v_rel_out, JNI_FALSE);

    kep_toolbox::array3D vrelin_a;
    kep_toolbox::array3D vrelout_a;
    for (int i = 0; i < 3; i++) {
        vrelin_a[i] = vrelin[i];
        vrelout_a[i] = vrelout[i];
    }
    kep_toolbox::planet::jpl_lp pl(getPlanet(pli));
    double dV;
    kep_toolbox::fb_vel(dV, vrelin_a, vrelout_a, pl);

    env->ReleaseDoubleArrayElements(v_rel_in, vrelin, 0);
    env->ReleaseDoubleArrayElements(v_rel_out, vrelout, 0);

    return dV;
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


