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
#include <ctime>
#include <random>
#include "pcg_random.hpp"
#include "call_java.hpp"
#include "biteopt.h"

using namespace std;

typedef Eigen::Matrix<double, Eigen::Dynamic, 1> vec;
typedef Eigen::Matrix<int, Eigen::Dynamic, 1> ivec;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> mat;

typedef double (*callback_type)(int, double[]);

namespace biteopt {

// wrapper around the Fitness function, scales according to boundaries

class Fitness {

public:

    Fitness(CallJava *pfunc, const vec &lower_limit, const vec &upper_limit) {
        func = pfunc;
        lower = lower_limit;
        upper = upper_limit;
        evaluationCounter = 0;
        if (lower.size() > 0) // bounds defined
            scale = (upper - lower);
    }

    double eval(const double *const p) {
        int n = lower.size();
        double parg[n];
        for (int i = 0; i < n; i++)
            parg[i] = p[i];
        double res = func->evalJava1(n, parg);
        evaluationCounter++;
        return res;
    }

    int getEvaluations() {
        return evaluationCounter;
    }

    void getMinValues(double *const p) const {
        for (int i = 0; i < lower.size(); i++)
            p[i] = lower[i];
    }

    void getMaxValues(double *const p) const {
        for (int i = 0; i < upper.size(); i++)
            p[i] = upper[i];
    }

private:
    CallJava *func;
    vec lower;
    vec upper;
    long evaluationCounter;
    vec scale;
};

class BiteOptimizer: public CBiteOptDeep {

public:

    BiteOptimizer(long runid_, Fitness *fitfun_, int dim_, double *init_,
            int seed_, int M_, int stallCriterion_, int maxEvaluations_, double stopfitness_) {
        // runid used to identify a specific run
        runid = runid_;
        // fitness function to minimize
        fitfun = fitfun_;
        // Number of objective variables/problem dimension
        dim = dim_;
        // Depth to use, 1 for plain CBiteOpt algorithm, >1 for CBiteOptDeep. Expected range is [1; 36].
        M = M_ > 0 ? M_ : 1;
        // terminate if stallCriterion_*128*evaluations stalled, if <= 0 not used
        stallCriterion = stallCriterion_ > 0 ? stallCriterion_ : 0;
        // maximal number of evaluations allowed.
        maxEvaluations = maxEvaluations_ > 0 ? maxEvaluations_ : 50000;
        // Number of iterations already performed.
        // Limit for fitness value.
        stopfitness = stopfitness_;
        // stop criteria
        stop = 0;

        iterations = 0;
        bestY = DBL_MAX;
        rnd.init(seed_);
        updateDims(dim_, M);
        init(rnd, init_);
    }

    virtual void getMinValues(double *const p) const {
        fitfun->getMinValues(p);
    }

    virtual void getMaxValues(double *const p) const {
        fitfun->getMaxValues(p);
    }

    virtual double optcost(const double *const p) {
        return fitfun->eval(p);
    }

    vec getBestX() {
        vec bestX = vec(dim);
        const double *bx = getBestParams();
        for (int i = 0; i < dim; i++)
            bestX[i] = bx[i];
        return bestX;
    }

    double getBestValue() {
        return getBestCost();
    }

    double getIterations() {
        return iterations;
    }

    double getStop() {
        return stop;
    }

    void doOptimize() {

        // -------------------- Generation Loop --------------------------------
        for (iterations = 1; fitfun->getEvaluations() < maxEvaluations;
                iterations++) {
            int stallCount = optimize(rnd);
            if (getBestCost() < stopfitness) {
                stop = 1;
                break;
            }
            if (stallCriterion > 0 && stallCount > stallCriterion*128*dim) {
                stop = 2;
                break;
            }
        }
    }

private:
    long runid;
    Fitness *fitfun;
    int M; // deepness
    int stallCriterion; // terminate if f stallCriterion*128*evaluations stalled, if <= 0 not used
    int dim;
    int maxEvaluations;
    double stopfitness;
    int iterations;
    double bestY;
    int stop;
    vec bestX;
    CBiteRnd rnd;
};

}

using namespace biteopt;

/*
 * Class:     fcmaes_core_Jni
 * Method:    optimizeBite
 * Signature: (Lfcmaes/core/Fitness;[D[D[DIDIIJI)I
 */
JNIEXPORT jint JNICALL Java_fcmaes_core_Jni_optimizeBite(JNIEnv *env,
        jclass cls, jobject func, jdoubleArray jlower, jdoubleArray jupper,
        jdoubleArray jinit, jint maxEvals, jdouble stopfitness, jint M,
        jint stallLimit, jlong seed, jint runid) {

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
    Fitness fitfun(&callJava, lower_limit, upper_limit);

    BiteOptimizer opt(runid, &fitfun, dim, init, seed, M, stallLimit, maxEvals,
            stopfitness);

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

