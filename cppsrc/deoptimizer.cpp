// Copyright (c) Dietmar Wolz.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory.

// Eigen based implementation of differential evolution using on the DE/best/1 strategy.
// Uses two deviations from the standard DE algorithm:
// a) temporal locality introduced in 
// https://www.researchgate.net/publication/309179699_Differential_evolution_for_protein_folding_optimization_based_on_a_three-dimensional_AB_off-lattice_model
// b) reinitialization of individuals based on their age. 
// requires https://github.com/imneme/pcg-cpp

#include <Eigen/Core>
#include <iostream>
#include <float.h>
#include <stdint.h>
#include <ctime>
#include <random>
#include <queue>
#include <tuple>
#include "pcg_random.hpp"
#include "call_java.hpp"

using namespace std;

typedef Eigen::Matrix<double, Eigen::Dynamic, 1> vec;
typedef Eigen::Matrix<int, Eigen::Dynamic, 1> ivec;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> mat;

typedef double (*callback_type)(int, double[]);

namespace differential_evolution {

static uniform_real_distribution<> distr_01 = std::uniform_real_distribution<>(
        0, 1);

static normal_distribution<> gauss_01 = std::normal_distribution<>(0, 1);

static vec zeros(int n) {
    return Eigen::MatrixXd::Zero(n, 1);
}

static Eigen::MatrixXd uniform(int dx, int dy, pcg64 &rs) {
    return Eigen::MatrixXd::NullaryExpr(dx, dy, [&]() {
        return distr_01(rs);
    });
}

static Eigen::MatrixXd uniformVec(int dim, pcg64 &rs) {
    return Eigen::MatrixXd::NullaryExpr(dim, 1, [&]() {
        return distr_01(rs);
    });
}

static Eigen::MatrixXd normalVec(int dim, pcg64 &rs) {
    return Eigen::MatrixXd::NullaryExpr(dim, 1, [&]() {
        return gauss_01(rs);
    });
}

static int index_min(vec &v) {
    double minv = DBL_MAX;
    int mi = -1;
    for (int i = 0; i < v.size(); i++) {
        if (v[i] < minv) {
            mi = i;
            minv = v[i];
        }
    }
    return mi;
}

// wrapper around the fitness function, scales according to boundaries

class Fitness {

public:

    Fitness(CallJava *pfunc, int dimension, const vec &lower_limit,
            const vec &upper_limit) {
        func = pfunc;
        dim = dimension;
        lower = lower_limit;
        upper = upper_limit;
        evaluationCounter = 0;
        if (lower.size() > 0) // bounds defined
            scale = (upper - lower);
    }

    double eval(const vec &X) {
        int n = X.size();
        double parg[n];
        for (int i = 0; i < n; i++)
            parg[i] = X(i);
        double res = func->evalJava1(n, parg);
        evaluationCounter++;
        return res;
    }

    void values(const mat &popX, int popsize, vec &ys) {
        for (int p = 0; p < popsize; p++)
            ys[p] = eval(popX.col(p));
    }

    vec getClosestFeasible(const vec &X) const {
        if (lower.size() > 0)
            return X.cwiseMin(upper).cwiseMax(lower);
        else
            return X;
    }

    bool feasible(int i, double x) {
        return lower.size() == 0 || (x >= lower[i] && x <= upper[i]);
    }

    vec sample(pcg64 &rs) {
        if (lower.size() > 0) {
            vec rv = uniformVec(dim, rs);
            return (rv.array() * scale.array()).matrix() + lower;
        } else
            return normalVec(dim, rs);
    }

    double sample_i(int i, pcg64 &rs) {
        if (lower.size() > 0)
            return lower[i] + scale[i] * distr_01(rs);
        else
            return gauss_01(rs);
    }

    int getEvaluations() {
        return evaluationCounter;
    }

private:
    CallJava *func;
    int dim;
    vec lower;
    vec upper;
    long evaluationCounter;
    vec scale;
};

class DeOptimizer {

public:

    DeOptimizer(long runid_, Fitness *fitfun_, int dim_, int seed_,
            int popsize_, int maxEvaluations_, double keep_,
            double stopfitness_, double F_, double CR_) {
        // runid used to identify a specific run
        runid = runid_;
        // fitness function to minimize
        fitfun = fitfun_;
        // Number of objective variables/problem dimension
        dim = dim_;
        // Population size
        popsize = popsize_ > 0 ? popsize_ : 15 * dim;
        // maximal number of evaluations allowed.
        maxEvaluations = maxEvaluations_ > 0 ? maxEvaluations_ : 50000;
        // keep best young after each iteration.
        keep = keep_ > 0 ? keep_ : 30;
        // Limit for fitness value.
        stopfitness = stopfitness_;
        F = F0 = F_ > 0 ? F_ : 0.5;
        CR = CR0 = CR_ > 0 ? CR_ : 0.9;
        // Number of iterations already performed.
        iterations = 0;
        bestY = DBL_MAX;
        // stop criteria
        stop = 0;
        pos = 0;
        //std::random_device rd;
        rs = new pcg64(seed_);
        init();
    }

    ~DeOptimizer() {
        delete rs;
    }

    double rnd01() {
        return distr_01(*rs);
    }

    int rndInt(int max) {
        return (int) (max * distr_01(*rs));
    }

    vec nextX(int p, const vec &xp, const vec &xb) {
        if (p == 0) {
            iterations++;
            CR = iterations % 2 == 0 ? 0.5 * CR0 : CR0;
            F = iterations % 2 == 0 ? 0.5 * F0 : F0;
        }
        int r1, r2;
        do {
            r1 = rndInt(popsize);
        } while (r1 == p || r1 == bestI);
        do {
            r2 = rndInt(popsize);
        } while (r2 == p || r2 == bestI || r2 == r1);
        vec x1 = popX.col(r1);
        vec x2 = popX.col(r2);
        vec x = xb + (x1 - x2) * F;
        int r = rndInt(dim);
        for (int j = 0; j < dim; j++)
            if (j != r && rnd01() > CR)
                x[j] = xp[j];
        return fitfun->getClosestFeasible(x);
    }

    vec next_improve(const vec &xb, const vec &x, const vec &xi) {
        return fitfun->getClosestFeasible(xb + ((x - xi) * 0.5));
    }

    vec ask_one(int &p) {
        // ask for one new argument vector.
        if (improvesX.empty()) {
            p = pos;
            vec x = nextX(p, popX.col(p), popX.col(bestI));
            pos = (pos + 1) % popsize;
            return x;
        } else {
            p = improvesP.front();
            vec x = improvesX.front();
            improvesP.pop();
            improvesX.pop();
            return x;
        }
    }

    int tell_one(double y, const vec &x, int p) {
        //tell function value for a argument list retrieved by ask_one().
        if (isfinite(y) && y < popY[p]) {
            if (iterations > 1) {
                // temporal locality
                improvesP.push(p);
                improvesX.push(next_improve(popX.col(bestI), x, popX0.col(p)));
            }
            popX0.col(p) = popX.col(p);
            popX.col(p) = x;
            popY[p] = y;
            popIter[p] = iterations;
            if (y < popY[bestI]) {
                bestI = p;
                if (y < bestY) {
                    bestY = y;
                    bestX = x;
                    if (isfinite(stopfitness) && bestY < stopfitness)
                        stop = 1;
                }
            }
        } else {
            // reinitialize individual
            if (keep * rnd01() < iterations - popIter[p]) {
                popX.col(p) = fitfun->sample(*rs);
                popY[p] = DBL_MAX;
            }
        }
        return stop;
    }

    void doOptimize() {

        // -------------------- Generation Loop --------------------------------
        for (iterations = 1; fitfun->getEvaluations() < maxEvaluations;
                iterations++) {

            CR = iterations % 2 == 0 ? 0.5 * CR0 : CR0;
            F = iterations % 2 == 0 ? 0.5 * F0 : F0;

            for (int p = 0; p < popsize; p++) {
                vec xp = popX.col(p);
                vec xb = popX.col(bestI);

//                vec x = nextX(p, xp, xb);

                int r1, r2;
                do {
                    r1 = rndInt(popsize);
                } while (r1 == p || r1 == bestI);
                do {
                    r2 = rndInt(popsize);
                } while (r2 == p || r2 == bestI || r2 == r1);
                vec x1 = popX.col(r1);
                vec x2 = popX.col(r2);
                int r = rndInt(dim);
                vec x = vec(xp);
                for (int j = 0; j < dim; j++) {
                    if (j == r || rnd01() < CR) {
                        x[j] = xb[j] + F * (x1[j] - x2[j]);
                        if (!fitfun->feasible(j, x[j]))
                            x[j] = fitfun->sample_i(j, *rs);
                    }
                }

                double y = fitfun->eval(x);
                if (isfinite(y) && y < popY[p]) {
                    // temporal locality
                    vec x2 = next_improve(xb, x, xp);
                    double y2 = fitfun->eval(x2);
                    if (isfinite(y2) && y2 < y) {
                        y = y2;
                        x = x2;
                    }
                    popX.col(p) = x;
                    popY(p) = y;
                    popIter[p] = iterations;
                    if (y < popY[bestI]) {
                        bestI = p;
                        if (y < bestY) {
                            bestY = y;
                            bestX = x;
                            if (isfinite(stopfitness) && bestY < stopfitness) {
                                stop = 1;
                                return;
                            }
                        }
                    }
                } else {
                    // reinitialize individual
                    if (keep * rnd01() < iterations - popIter[p]) {
                        popX.col(p) = fitfun->sample(*rs);
                        popY[p] = DBL_MAX;
                    }
                }
            }
        }
    }

    void init() {
        popX = mat(dim, popsize);
        popX0 = mat(dim, popsize);
        popY = vec(popsize);
        for (int p = 0; p < popsize; p++) {
            popX0.col(p) = popX.col(p) = fitfun->sample(*rs);
            popY[p] = DBL_MAX; // compute fitness
        }
        bestI = 0;
        bestX = popX.col(bestI);
        popIter = zeros(popsize);
    }

    vec getBestX() {
        return bestX;
    }

    double getBestValue() {
        return bestY;
    }

    double getIterations() {
        return iterations;
    }

    double getStop() {
        return stop;
    }

    Fitness* getFitfun() {
        return fitfun;
    }

    int getDim() {
        return dim;
    }

private:
    long runid;
    Fitness *fitfun;
    int popsize; // population size
    int dim;
    int maxEvaluations;
    double keep;
    double stopfitness;
    int iterations;
    double bestY;
    vec bestX;
    int bestI;
    int stop;
    double F0;
    double CR0;
    double F;
    double CR;
    pcg64 *rs;
    mat popX;
    mat popX0;
    vec popY;
    vec popIter;
    queue<vec> improvesX;
    queue<int> improvesP;
    int pos;
};

// see https://cvstuff.wordpress.com/2014/11/27/wraping-c-code-with-python-ctypes-memory-and-pointers/

}

using namespace differential_evolution;

/*
 * Class:     fcmaes_core_Jni
 * Method:    optimizeDE
 * Signature: (Lutils/Jni/Fitness;[D[D[DIDIDDDJI)I
 */
JNIEXPORT jint JNICALL Java_fcmaes_core_Jni_optimizeDE(JNIEnv *env, jclass cls,
        jobject func, jdoubleArray jlower, jdoubleArray jupper,
        jdoubleArray jinit, jint maxEvals, jdouble stopfitness, jint popsize,
        jdouble keep, jdouble F, jdouble CR, jlong seed, jint runid) {

    double *init = env->GetDoubleArrayElements(jinit, JNI_FALSE);
    double *lower = env->GetDoubleArrayElements(jlower, JNI_FALSE);
    double *upper = env->GetDoubleArrayElements(jupper, JNI_FALSE);
    int dim = env->GetArrayLength(jinit);
    vec lower_limit(dim), upper_limit(dim);
    bool useLimit = false;
    for (int i = 0; i < dim; i++) {
        lower_limit[i] = lower[i];
        upper_limit[i] = upper[i];
        useLimit |= (lower[i] != 0);
        useLimit |= (upper[i] != 0);
    }
    if (useLimit == false) {
        lower_limit.resize(0);
        upper_limit.resize(0);
    }
    CallJava callJava(func, env);
    Fitness fitfun(&callJava, dim, lower_limit, upper_limit);

    DeOptimizer opt(runid, &fitfun, dim, seed, popsize, maxEvals, keep,
            stopfitness, F, CR);
    try {
        opt.doOptimize();
        vec bestX = opt.getBestX();
        double bestY = opt.getBestValue();

        for (int i = 0; i < dim; i++)
            init[i] = bestX[i];

        env->SetDoubleArrayRegion(jinit, 0, dim, (jdouble*) init);
        env->ReleaseDoubleArrayElements(jinit, init, 0);
        env->ReleaseDoubleArrayElements(jupper, upper, 0);
        env->ReleaseDoubleArrayElements(jlower, lower, 0);
        return fitfun.getEvaluations();

    } catch (std::exception &e) {
        cout << e.what() << endl;
        return fitfun.getEvaluations();
    }
    return 0;
}

/*
 * Class:     fcmaes_core_Jni
 * Method:    initDE
 * Signature: ([D[D[DIDDDJI)J
 */
JNIEXPORT intptr_t JNICALL Java_fcmaes_core_Jni_initDE(JNIEnv *env, jclass cls,
        jdoubleArray jlower, jdoubleArray jupper, jdoubleArray jinit,
        jint popsize, jdouble keep, jdouble F, jdouble CR, jlong seed,
        jint runid) {
    double *init = env->GetDoubleArrayElements(jinit, JNI_FALSE);
    double *lower = env->GetDoubleArrayElements(jlower, JNI_FALSE);
    double *upper = env->GetDoubleArrayElements(jupper, JNI_FALSE);
    int dim = env->GetArrayLength(jinit);

    vec guess(dim), lower_limit(dim), upper_limit(dim);
    bool useLimit = false;
    for (int i = 0; i < dim; i++) {
        guess[i] = init[i];
        lower_limit[i] = lower[i];
        upper_limit[i] = upper[i];
        useLimit |= (lower[i] != 0);
        useLimit |= (upper[i] != 0);
    }
    if (useLimit == false) {
        lower_limit.resize(0);
        upper_limit.resize(0);
    }
    Fitness *fitfun = new Fitness(NULL, dim, lower_limit, upper_limit);
    DeOptimizer *opt = new DeOptimizer(runid, fitfun, dim, seed, popsize,
            INT_MAX, keep, -DBL_MAX, F, CR);
    env->ReleaseDoubleArrayElements(jinit, init, 0);
    env->ReleaseDoubleArrayElements(jupper, upper, 0);
    env->ReleaseDoubleArrayElements(jlower, lower, 0);
    return (intptr_t) opt;
}

/*
 * Class:     fcmaes_core_Jni
 * Method:    destroyDE
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_fcmaes_core_Jni_destroyDE(JNIEnv *env, jclass cls, intptr_t ptr) {
    DeOptimizer* opt = (DeOptimizer*)ptr;
    Fitness* fitfun = opt->getFitfun();
    delete fitfun;
    delete opt;
}

/*
 * Class:     fcmaes_core_Jni
 * Method:    askDE
 * Signature: (J)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_fcmaes_core_Jni_askDE(JNIEnv *env,
        jclass cls, intptr_t ptr) {
    DeOptimizer *opt = (DeOptimizer*) ptr;
    int dim = opt->getDim();
    jdoubleArray jx = env->NewDoubleArray(dim + 1);
    double x[dim + 1];
    int p;
    vec args = opt->ask_one(p);
    for (int i = 0; i < dim; i++)
        x[i] = args[i];
    x[dim] = p;
    env->SetDoubleArrayRegion(jx, 0, dim + 1, x);
    return jx;
}

/*
 * Class:     fcmaes_core_Jni
 * Method:    tellDE
 * Signature: (J[DDI)I
 */
JNIEXPORT jint JNICALL Java_fcmaes_core_Jni_tellDE(JNIEnv *env, jclass cls,
		intptr_t ptr, jdoubleArray jx, jdouble y, jint p) {
    DeOptimizer *opt = (DeOptimizer*) ptr;
    int dim = opt->getDim();
    double *x = env->GetDoubleArrayElements(jx, JNI_FALSE);
    vec args(dim);
    for (int i = 0; i < dim; i++)
        args[i] = x[i];
    opt->tell_one(y, args, p);
    env->ReleaseDoubleArrayElements(jx, x, 0);
    return opt->getStop();
}
