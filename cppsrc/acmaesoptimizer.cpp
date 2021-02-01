// Copyright (c) Dietmar Wolz.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory.

// Eigen based implementation of active CMA-ES
// derived from http://cma.gforge.inria.fr/cmaes.m which follows
// https://www.researchgate.net/publication/227050324_The_CMA_Evolution_Strategy_A_Comparing_Review
// Requires Eigen version >= 3.3.90 because new slicing capabilities are used, see
// https://eigen.tuxfamily.org/dox-devel/group__TutorialSlicingIndexing.html
// requires https://github.com/imneme/pcg-cpp

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <iostream>
#include <random>
#include <float.h>
#include <stdint.h>
#include <ctime>
#include "pcg_random.hpp"
#include "call_java.hpp"

using namespace std;

typedef Eigen::Matrix<double, Eigen::Dynamic, 1> vec;
typedef Eigen::Matrix<int, Eigen::Dynamic, 1> ivec;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> mat;

typedef void (*callback_parallel)(int, int, double[], double[]);
typedef bool (*is_terminate_type)(long, int, double); // runid, iter, value -> isTerminate

namespace acmaes {

static vec zeros(int n) {
    return Eigen::MatrixXd::Zero(n, 1);
}

static ivec inverse(const ivec &indices) {
    ivec inverse = ivec(indices.size());
    for (int i = 0; i < indices.size(); i++)
        inverse(indices(i)) = i;
    return inverse;
}

static vec sequence(double start, double end, double step) {
    int size = (int) ((end - start) / step + 1);
    vec d(size);
    double value = start;
    for (int r = 0; r < size; r++) {
        d(r) = value;
        value += step;
    }
    return d;
}

struct IndexVal {
    int index;
    double val;
};

static bool compareIndexVal(IndexVal i1, IndexVal i2) {
    return (i1.val < i2.val);
}

static ivec sort_index(const vec &x) {
    int size = x.size();
    IndexVal ivals[size];
    for (int i = 0; i < size; i++) {
        ivals[i].index = i;
        ivals[i].val = x[i];
    }
    std::sort(ivals, ivals + size, compareIndexVal);
    return Eigen::MatrixXi::NullaryExpr(size, 1, [&ivals](int i) {
        return ivals[i].index;
    });
}

static normal_distribution<> gauss_01 = std::normal_distribution<>(0, 1);

static Eigen::MatrixXd normal(int dx, int dy, pcg64 &rs) {
    return Eigen::MatrixXd::NullaryExpr(dx, dy, [&]() {
        return gauss_01(rs);
    });
}

static Eigen::MatrixXd normalVec(int dim, pcg64 &rs) {
    return Eigen::MatrixXd::NullaryExpr(dim, 1, [&]() {
        return gauss_01(rs);
    });
}

// wrapper around the fitness function, scales according to boundaries

class Fitness {

public:

    Fitness(CallJava *func_par_, const vec &lower_limit, const vec &upper_limit,
            bool normalize_) {
        func_par = func_par_;
        lower = lower_limit;
        upper = upper_limit;
        normalize = normalize_ && lower.size() > 0;
        evaluationCounter = 0;
        if (lower.size() > 0) { // bounds defined
            scale = 0.5 * (upper - lower);
            typx = 0.5 * (upper + lower);
        }
    }

    vec getClosestFeasible(const vec &X) const {
        if (lower.size() > 0) {
            if (normalize)
                return X.cwiseMin(1.0).cwiseMax(-1.0);
            else
                return X.cwiseMin(upper).cwiseMax(lower);
        }
        return X;
    }

    void values(const mat &popX, vec &ys) {
        int popsize = popX.cols();
        int n = popX.rows();
        double pargs[popsize * n];
        double res[popsize];
        for (int p = 0; p < popX.cols(); p++) {
            vec x = decode(getClosestFeasible(popX.col(p)));
            for (int i = 0; i < n; i++)
                pargs[p * n + i] = x[i];
        }
        func_par->evalJava(popsize, n, pargs, res);
        for (int p = 0; p < popX.cols(); p++)
            ys[p] = res[p];
        evaluationCounter += popsize;
    }

    vec encode(const vec &X) const {
        if (normalize)
            return (X - typx).array() / scale.array();
        else
            return X;
    }

    vec decode(const vec &X) const {
        if (normalize)
            return (X.array() * scale.array()).matrix() + typx;
        else
            return X;
    }

    int getEvaluations() {
        return evaluationCounter;
    }

private:
    CallJava *func_par;
    vec lower;
    vec upper;
    long evaluationCounter;
    vec scale;
    vec typx;
    bool normalize;
};

class AcmaesOptimizer {

public:

    AcmaesOptimizer(long runid_, Fitness *fitfun_, int popsize_, int mu_,
            const vec &guess_, const vec &inputSigma_, int maxIterations_,
            int maxEvaluations_, double accuracy_, double stopfitness_,
            int update_gap_, is_terminate_type isTerminate_, long seed) {
        // runid used in isTerminate callback to identify a specific run at different iteration
        runid = runid_;
        // fitness function to minimize
        fitfun = fitfun_;
        // initial guess for the arguments of the fitness function
        guess = guess_;
        // accuracy = 1.0 is default, > 1.0 reduces accuracy
        accuracy = accuracy_;
        // callback to check if to terminate
        isTerminate = isTerminate_;
        // number of objective variables/problem dimension
        dim = guess_.size();
        // population size, offspring number. The primary strategy parameter to play
        // with, which can be increased from its default value. Increasing the
        // population size improves global search properties in exchange to speed.
        // Speed decreases, as a rule, at most linearly with increasing population
        // size. It is advisable to begin with the default small population size.
        if (popsize_ > 0)
            popsize = popsize_;
        else
            popsize = 4 + int(3. * log(dim));
        // individual sigma values - initial search volume. inputSigma determines
        // the initial coordinate wise standard deviations for the search. Setting
        // SIGMA one third of the initial search region is appropriate.
        if (inputSigma_.size() == 1)
            inputSigma = vec::Constant(dim, inputSigma_[0]);
        else
            inputSigma = inputSigma_;
        // overall standard deviation - search volume.
        sigma = inputSigma.maxCoeff();
        // termination criteria
        // maximal number of evaluations allowed.
        maxEvaluations = maxEvaluations_;
        // maximal number of iterations allowed.
        maxIterations = maxIterations_;
        // limit for fitness value.
        stopfitness = stopfitness_;
        // stop if x-changes larger stopTolUpX.
        stopTolUpX = 1e3 * sigma;
        // stop if x-change smaller stopTolX.
        stopTolX = 1e-11 * sigma * accuracy;
        // stop if fun-changes smaller stopTolFun.
        stopTolFun = 1e-12 * accuracy;
        // stop if back fun-changes smaller stopTolHistFun.
        stopTolHistFun = 1e-13 * accuracy;
        // selection strategy parameters
        // number of parents/points for recombination.
        mu = mu_ > 0 ? mu_ : popsize / 2;
        // array for weighted recombination.
        weights = (log(sequence(1, mu, 1).array()) * -1.) + log(mu + 0.5);
        double sumw = weights.sum();
        double sumwq = weights.squaredNorm();
        weights *= 1. / sumw;
        // variance-effectiveness of sum w_i x_i.
        mueff = sumw * sumw / sumwq;

        // dynamic strategy parameters and constants
        // cumulation constant.
        cc = (4. + mueff / dim) / (dim + 4. + 2. * mueff / dim);
        // cumulation constant for step-size.
        cs = (mueff + 2.) / (dim + mueff + 3.);
        // damping for step-size.
        damps = (1. + 2. * std::max(0., sqrt((mueff - 1.) / (dim + 1.)) - 1.))
                * max(0.3,
                        1. - // modification for short runs
                                dim
                                        / (1e-6
                                                + min(maxIterations,
                                                        maxEvaluations
                                                                / popsize)))
                + cs; // minor increment
        // learning rate for rank-one update.
        ccov1 = 2. / ((dim + 1.3) * (dim + 1.3) + mueff);
        // learning rate for rank-mu update'
        ccovmu = min(1. - ccov1,
                2. * (mueff - 2. + 1. / mueff)
                        / ((dim + 2.) * (dim + 2.) + mueff));
        // expectation of ||N(0,I)|| == norm(randn(N,1)).
        chiN = sqrt(dim) * (1. - 1. / (4. * dim) + 1 / (21. * dim * dim));
        ccov1Sep = min(1., ccov1 * (dim + 1.5) / 3.);
        ccovmuSep = min(1. - ccov1, ccovmu * (dim + 1.5) / 3.);
        // lazy covariance update gap
        lazy_update_gap =
                update_gap_ >= 0 ?
                        update_gap_ :
                        1.0 / (ccov1 + ccovmu + 1e-23) / dim / 10.0;
        // CMA internal values - updated each generation
        // objective variables.
        xmean = fitfun->encode(guess);
        // evolution path.
        pc = zeros(dim);
        // evolution path for sigma.
        ps = zeros(dim);
        // norm of ps, stored for efficiency.
        normps = ps.norm();
        // coordinate system.
        B = Eigen::MatrixXd::Identity(dim, dim);
        // diagonal of sqrt(D), stored for efficiency.
        diagD = inputSigma / sigma;
        diagC = diagD.cwiseProduct(diagD);
        // B*D, stored for efficiency.
        BD = B.cwiseProduct(diagD.transpose().replicate(dim, 1));
        // covariance matrix.
        C = B * (Eigen::MatrixXd::Identity(dim, dim) * B.transpose());
        // number of iterations.
        iterations = 1;
        // size of history queue of best values.
        historySize = 10 + int(3. * 10. * dim / popsize);
        // stop criteria
        stop = 0;
        // best value so far
        bestValue = DBL_MAX;
        // best parameters so far
        bestX = guess;
        // history queue of best values.
        fitnessHistory = vec::Constant(historySize, DBL_MAX);
        fitnessHistory(0) = bestValue;
        rs = new pcg64(seed);
    }

    ~AcmaesOptimizer() {
        delete rs;
    }

    // param zmean weighted row matrix of the gaussian random numbers generating the current offspring
    // param xold xmean matrix of the previous generation
    // return hsig flag indicating a small correction

    bool updateEvolutionPaths(const vec &zmean, const vec &xold) {
        ps = ps * (1. - cs) + ((B * zmean) * sqrt(cs * (2. - cs) * mueff));
        normps = ps.norm();
        bool hsig = normps / sqrt(1. - pow(1. - cs, 2. * iterations)) / chiN
                < 1.4 + 2. / (dim + 1.);
        pc *= (1. - cc);
        if (hsig)
            pc += (xmean - xold) * (sqrt(cc * (2. - cc) * mueff) / sigma);
        return hsig;
    }

    // param hsig flag indicating a small correction
    // param bestArx fitness-sorted matrix of the argument vectors producing the current offspring
    // param arz unsorted matrix containing the gaussian random values of the current offspring
    // param arindex indices indicating the fitness-order of the current offspring
    // param xold xmean matrix of the previous generation

    double updateCovariance(bool hsig, const mat &bestArx, const mat &arz,
            const ivec &arindex, const mat &xold) {
        double negccov = 0;
        if (ccov1 + ccovmu > 0) {
            mat arpos = (bestArx - xold.replicate(1, mu)) * (1. / sigma); // mu difference vectors
            mat roneu = pc * pc.transpose() * ccov1;
            // minor correction if hsig==false
            double oldFac = hsig ? 0 : ccov1 * cc * (2. - cc);
            oldFac += 1. - ccov1 - ccovmu;
            // Adapt covariance matrix C active CMA
            negccov = (1. - ccovmu) * 0.25 * mueff
                    / (pow(dim + 2., 1.5) + 2. * mueff);
            double negminresidualvariance = 0.66;
            // keep at least 0.66 in all directions, small popsize are most critical
            double negalphaold = 0.5; // where to make up for the variance loss,
            // prepare vectors, compute negative updating matrix Cneg
            ivec arReverseIndex = arindex.reverse();
            mat arzneg = arz(Eigen::all, arReverseIndex.head(mu));
            vec arnorms = arzneg.colwise().norm();
            ivec idxnorms = sort_index(arnorms);
            vec arnormsSorted = arnorms(idxnorms);
            ivec idxReverse = idxnorms.reverse();
            vec arnormsReverse = arnorms(idxReverse);
            arnorms = arnormsReverse.cwiseQuotient(arnormsSorted);
            vec arnormsInv = arnorms(inverse(idxnorms));
            mat sqarnw = arnormsInv.cwiseProduct(arnormsInv).transpose()
                    * weights;
            double negcovMax = (1. - negminresidualvariance) / sqarnw(0);
            if (negccov > negcovMax)
                negccov = negcovMax;
            arzneg = arzneg.cwiseProduct(
                    arnormsInv.transpose().replicate(dim, 1));
            mat artmp = BD * arzneg;
            mat Cneg = artmp * weights.asDiagonal() * artmp.transpose();
            oldFac += negalphaold * negccov;
            C = (C * oldFac) + roneu
                    + (arpos * (ccovmu + (1. - negalphaold) * negccov)
                            * weights.replicate(1, dim).cwiseProduct(
                                    arpos.transpose())) - (Cneg * negccov);
        }
        return negccov;
    }

    // Update B and diagD from C
    // param negccov Negative covariance factor.

    void updateBD(double negccov) {

        if (ccov1 + ccovmu + negccov > 0
                && (std::fmod(iterations,
                        1. / (ccov1 + ccovmu + negccov) / dim / 10.)) < 1.) {
            // to achieve O(N^2) enforce symmetry to prevent complex numbers
            mat triC = C.triangularView<Eigen::Upper>();
            mat triC1 = C.triangularView<Eigen::StrictlyUpper>();
            C = triC + triC1.transpose();
            Eigen::SelfAdjointEigenSolver<mat> sades;
            sades.compute(C);
            // diagD defines the scaling
            diagD = sades.eigenvalues();
            B = sades.eigenvectors();
            if (diagD.minCoeff() <= 0) {
                for (int i = 0; i < dim; i++)
                    if (diagD(i, 0) < 0)
                        diagD(i, 0) = 0.;
                double tfac = diagD.maxCoeff() / 1e14;
                C += Eigen::MatrixXd::Identity(dim, dim) * tfac;
                diagD += vec::Constant(dim, 1.0) * tfac;
            }
            if (diagD.maxCoeff() > 1e14 * diagD.minCoeff()) {
                double tfac = diagD.maxCoeff() / 1e14 - diagD.minCoeff();
                C += Eigen::MatrixXd::Identity(dim, dim) * tfac;
                diagD += vec::Constant(dim, 1.0) * tfac;
            }
            diagC = C.diagonal();
            diagD = diagD.cwiseSqrt(); // D contains standard deviations now
            BD = B.cwiseProduct(diagD.transpose().replicate(dim, 1));
        }
    }

    void newArgs() {
        // generate popsize offspring.
        xmean = fitfun->getClosestFeasible(xmean);
        arz = normal(dim, popsize, *rs);
        arx = mat(dim, popsize);
        for (int k = 0; k < popsize; k++) {
            vec delta = (BD * arz.col(k)) * sigma;
            arx.col(k) = fitfun->getClosestFeasible(xmean + delta);
        }
        fitness = vec(popsize);
    }

    vec ask_one() {
        // ask for one new argument vector.
        vec arz1 = normalVec(dim, *rs);
        vec delta = (BD * arz1) * sigma;
        vec arx1 = fitfun->getClosestFeasible(xmean + delta);
        return fitfun->decode(arx1);
    }

    int tell_one(double y, const vec &x) {
        //tell function value for a argument list retrieved by ask_one().
        if (told == 0) {
            fitness = vec(popsize);
            arx = mat(dim, popsize);
            arz = mat(dim, popsize);
        }
        fitness[told] = isfinite(y) ? y : DBL_MAX;
        arx.col(told) = fitfun->encode(x);
        told++;

        if (told >= popsize) {
            xmean = fitfun->getClosestFeasible(xmean);
            try {
                arz = (BD.inverse()
                        * ((arx - xmean.replicate(1, popsize)) / sigma));
            } catch (std::exception &e) {
                arz = normal(dim, popsize, *rs);
            }
            updateCMA();
            told = 0;
            iterations += 1;
        }
        return stop;
    }

    void updateCMA() {
        // sort by fitness and compute weighted mean into xmean
        ivec arindex = sort_index(fitness);
        // calculate new xmean, this is selection and recombination
        vec xold = xmean; // for speed up of Eq. (2) and (3)
        ivec bestIndex = arindex.head(mu);
        mat bestArx = arx(Eigen::all, bestIndex);
        xmean = bestArx * weights;
        mat bestArz = arz(Eigen::all, bestIndex);
        mat zmean = bestArz * weights;
        bool hsig = updateEvolutionPaths(zmean, xold);
        // adapt step size sigma
        sigma *= exp(min(1.0, (normps / chiN - 1.) * cs / damps));
        double bestFitness = fitness(arindex(0));
        double worstFitness = fitness(arindex(arindex.size() - 1));
        if (bestValue > bestFitness) {
            bestValue = bestFitness;
            bestX = fitfun->decode(bestArx.col(0));
            if (isfinite(stopfitness) && bestFitness < stopfitness) {
                stop = 1;
                return;
            }
        }
        if (iterations >= last_update + lazy_update_gap) {
            last_update = iterations;
            double negccov = updateCovariance(hsig, bestArx, arz, arindex,
                    xold);
            updateBD(negccov);
            // handle termination criteria
            vec sqrtDiagC = diagC.cwiseSqrt();
            vec pcCol = pc;
            for (int i = 0; i < dim; i++) {
                if (sigma * (max(abs(pcCol[i]), sqrtDiagC[i])) > stopTolX)
                    break;
                if (i >= dim - 1)
                    stop = 2;
            }
            if (stop > 0)
                return;
            for (int i = 0; i < dim; i++)
                if (sigma * sqrtDiagC[i] > stopTolUpX)
                    stop = 3;
            if (stop > 0)
                return;
        }
        double historyBest = fitnessHistory.minCoeff();
        double historyWorst = fitnessHistory.maxCoeff();
        if (iterations > 2
                && max(historyWorst, worstFitness)
                        - min(historyBest, bestFitness) < stopTolFun) {
            stop = 4;
            return;
        }
        if (iterations > fitnessHistory.size()
                && historyWorst - historyBest < stopTolHistFun) {
            stop = 5;
            return;
        }
        // condition number of the covariance matrix exceeds 1e14
        if (diagD.maxCoeff() / diagD.minCoeff() > 1e7 * 1.0 / sqrt(accuracy)) {
            stop = 6;
            return;
        }
        if (isTerminate != NULL && isTerminate(runid, iterations, bestValue)) {
            stop = 7;
            return;
        }
        // adjust step size in case of equal function values (flat fitness)
        if (bestValue == fitness[arindex[(int) (0.1 + popsize / 4.)]]) {
            sigma *= exp(0.2 + cs / damps);
        }
        if (iterations > 2
                && max(historyWorst, bestFitness)
                        - std::min(historyBest, bestFitness) == 0) {
            sigma *= ::exp(0.2 + cs / damps);
        }
        // store best in history
        for (int i = 1; i < fitnessHistory.size(); i++)
            fitnessHistory[i] = fitnessHistory[i - 1];
        fitnessHistory[0] = bestFitness;
    }

    void doOptimize() {

        // -------------------- Generation Loop --------------------------------
        for (iterations = 1;
                iterations <= maxIterations
                        && fitfun->getEvaluations() < maxEvaluations;
                iterations++) {
            // generate and evaluate popsize offspring
            newArgs();
            fitfun->values(arx, fitness);
            for (int k = 0; k < popsize; k++) {
                if (!isfinite(fitness[k]))
                    fitness[k] = DBL_MAX;
            }
            updateCMA();
            if (stop != 0)
                return;
        }
    }

    vec getBestX() {
        return bestX;
    }

    double getBestValue() {
        return bestValue;
    }

    double getIterations() {
        return iterations;
    }

    int getStop() {
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
    vec guess;
    double accuracy;
    is_terminate_type isTerminate;
    int popsize; // population size
    vec inputSigma;
    int dim;
    int maxIterations;
    int maxEvaluations;
    double stopfitness;
    double stopTolUpX;
    double stopTolX;
    double stopTolFun;
    double stopTolHistFun;
    int mu; //
    vec weights;
    double mueff; //
    double sigma;
    double cc;
    double cs;
    double damps;
    double ccov1;
    double ccovmu;
    double chiN;
    double ccov1Sep;
    double ccovmuSep;
    double lazy_update_gap = 0;
    vec xmean;
    vec pc;
    vec ps;
    double normps;
    mat B;
    mat BD;
    mat diagD;
    mat C;
    vec diagC;
    mat arz;
    mat arx;
    vec fitness;
    int iterations = 0;
    int last_update = 0;
    vec fitnessHistory;
    int historySize;
    double bestValue;
    vec bestX;
    int stop;
    int told = 0;
    pcg64 *rs;
};
}

using namespace acmaes;

/*
 * Class:     fcmaes_core_Jni
 * Method:    optimizeACMA
 * Signature: (Lfcmaes/core/Fitness;[D[D[D[DIIDIIDJIII)I
 */
JNIEXPORT jint JNICALL Java_fcmaes_core_Jni_optimizeACMA(JNIEnv *env,
        jclass cls, jobject func, jdoubleArray jlower, jdoubleArray jupper,
        jdoubleArray jsigma, jdoubleArray jinit, jint maxIter, jint maxEvals,
        jdouble stopfitness, jint popsize, jint mu, jdouble accuracy,
        jlong seed, jint runid, jint normalize, jint update_gap) {

    double *init = env->GetDoubleArrayElements(jinit, JNI_FALSE);
    double *lower = env->GetDoubleArrayElements(jlower, JNI_FALSE);
    double *upper = env->GetDoubleArrayElements(jupper, JNI_FALSE);
    double *sigma = env->GetDoubleArrayElements(jsigma, JNI_FALSE);
    int dim = env->GetArrayLength(jinit);

    vec guess(dim), lower_limit(dim), upper_limit(dim), inputSigma(dim);
    bool useLimit = false;
    for (int i = 0; i < dim; i++) {
        guess[i] = init[i];
        inputSigma[i] = sigma[i];
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
    Fitness fitfun(&callJava, lower_limit, upper_limit, normalize != 0);
    AcmaesOptimizer opt(runid, &fitfun, popsize, mu, guess, inputSigma, maxIter,
            maxEvals, accuracy, stopfitness, update_gap, NULL, seed);
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
        env->ReleaseDoubleArrayElements(jsigma, sigma, 0);
        return fitfun.getEvaluations();

    } catch (std::exception &e) {
        cout << e.what() << endl;
        return fitfun.getEvaluations();
    }
    return 0;
}

/*
 * Class:     fcmaes_core_Jni
 * Method:    initCmaes
 * Signature: ([D[D[D[DIIDJIII)J
 */
JNIEXPORT intptr_t JNICALL Java_fcmaes_core_Jni_initCmaes(JNIEnv *env, jclass cls,
        jdoubleArray jlower, jdoubleArray jupper, jdoubleArray jsigma,
        jdoubleArray jinit, jint popsize, jint mu, jdouble accuracy, jlong seed,
        jint runid, jint normalize, jint update_gap) {

    double *init = env->GetDoubleArrayElements(jinit, JNI_FALSE);
    double *lower = env->GetDoubleArrayElements(jlower, JNI_FALSE);
    double *upper = env->GetDoubleArrayElements(jupper, JNI_FALSE);
    double *sigma = env->GetDoubleArrayElements(jsigma, JNI_FALSE);
    int dim = env->GetArrayLength(jinit);

    vec guess(dim), lower_limit(dim), upper_limit(dim), inputSigma(dim);
    bool useLimit = false;
    for (int i = 0; i < dim; i++) {
        guess[i] = init[i];
        inputSigma[i] = sigma[i];
        lower_limit[i] = lower[i];
        upper_limit[i] = upper[i];
        useLimit |= (lower[i] != 0);
        useLimit |= (upper[i] != 0);
    }
    if (useLimit == false) {
        lower_limit.resize(0);
        upper_limit.resize(0);
    }
    Fitness *fitfun = new Fitness(NULL, lower_limit, upper_limit,
            normalize != 0);
    AcmaesOptimizer *opt = new AcmaesOptimizer(runid, fitfun, popsize, mu,
            guess, inputSigma, 0, 0, accuracy, -DBL_MAX, update_gap, NULL,
            seed);
    env->ReleaseDoubleArrayElements(jinit, init, 0);
    env->ReleaseDoubleArrayElements(jupper, upper, 0);
    env->ReleaseDoubleArrayElements(jlower, lower, 0);
    env->ReleaseDoubleArrayElements(jsigma, sigma, 0);
    return (intptr_t) opt;
}

/*
 * Class:     fcmaes_core_Jni
 * Method:    destroyCmaes
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_fcmaes_core_Jni_destroyCmaes(JNIEnv *env, jclass cls, intptr_t ptr) {
    AcmaesOptimizer* opt = (AcmaesOptimizer*)ptr;
    Fitness* fitfun = opt->getFitfun();
    delete fitfun;
    delete opt;
}

/*
 * Class:     fcmaes_core_Jni
 * Method:    askCmaes
 * Signature: (J)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_fcmaes_core_Jni_askCmaes(JNIEnv *env,
        jclass cls, intptr_t ptr) {
    AcmaesOptimizer *opt = (AcmaesOptimizer*) ptr;
    int dim = opt->getDim();
    jdoubleArray jx = env->NewDoubleArray(dim);
    double x[dim];
    vec args = opt->ask_one();
    for (int i = 0; i < dim; i++)
        x[i] = args[i];
    env->SetDoubleArrayRegion(jx, 0, dim, x);
    return jx;
}

/*
 * Class:     fcmaes_core_Jni
 * Method:    tellCmaes
 * Signature: (J[DD)I
 */
JNIEXPORT jint JNICALL Java_fcmaes_core_Jni_tellCmaes(JNIEnv *env, jclass cls,
		intptr_t ptr, jdoubleArray jx, jdouble y) {
    AcmaesOptimizer *opt = (AcmaesOptimizer*) ptr;
    int dim = opt->getDim();
    double *x = env->GetDoubleArrayElements(jx, JNI_FALSE);
    vec args(dim);
    for (int i = 0; i < dim; i++)
        args[i] = x[i];
    opt->tell_one(y, args);
    env->ReleaseDoubleArrayElements(jx, x, 0);
    return opt->getStop();
}

