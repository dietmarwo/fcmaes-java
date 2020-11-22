// Copyright (c)  Mingcheng Zuo, Dietmar Wolz.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory.

// Eigen based implementation of differential evolution (GCL-DE) derived from
// "A case learning-based differential evolution algorithm for global optimization of interplanetary trajectory design,
//  Mingcheng Zuo, Guangming Dai, Lei Peng, Maocai Wang, Zhengquan Liu", https://doi.org/10.1016/j.asoc.2020.106451

#include <Eigen/Core>
#include <iostream>
#include <float.h>
#include <ctime>
#include <random>
#include "pcg_random.hpp"
#include "call_java.hpp"

using namespace std;

typedef Eigen::Matrix<double, Eigen::Dynamic, 1> vec;
typedef Eigen::Matrix<int, Eigen::Dynamic, 1> ivec;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> mat;

typedef void (*callback_parallel)(int, int, double[], double[]);

namespace cl_differential_evolution {

static uniform_real_distribution<> distr_01 = std::uniform_real_distribution<>(
		0, 1);
static normal_distribution<> gauss_01 = std::normal_distribution<>(0, 1);

static double normreal(pcg64 *rs, double mu, double sdev) {
	return gauss_01(*rs) * sdev + mu;
}

static vec zeros(int n) {
	return Eigen::MatrixXd::Zero(n, 1);
}

static vec constant(int n, double val) {
	return Eigen::MatrixXd::Constant(n, 1, val);
}

static Eigen::MatrixXd uniformVec(int dim, pcg64 &rs) {
	return Eigen::MatrixXd::NullaryExpr(dim, 1, [&]() {
		return distr_01(rs);
	});
}

static Eigen::MatrixXd uniform(int dx, int dy, pcg64 &rs) {
	return Eigen::MatrixXd::NullaryExpr(dx, dy, [&]() {
		return distr_01(rs);
	});
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

static ivec sort_index(const vec &y, int* indices, int size) {
	IndexVal ivals[size];
	for (int i = 0; i < size; i++) {
		ivals[i].index = indices[i];
		ivals[i].val = y[indices[i]];
	}
	std::sort(ivals, ivals + size, compareIndexVal);
	return Eigen::MatrixXi::NullaryExpr(size, 1, [&ivals](int i) {
		return ivals[i].index;
	});
}

// wrapper around the Fitness function, scales according to boundaries

class Fitness {

public:

	Fitness(CallJava *func_par_, const vec &lower_limit,
			const vec &upper_limit) {
		func_par = func_par_;
		lower = lower_limit;
		upper = upper_limit;
		evaluationCounter = 0;
		if (lower.size() > 0) // bounds defined
			scale = (upper - lower);
	}

	vec norm(vec &X){
		return (X - lower).array() / scale.array();
	}

	vec getClosestFeasible(const vec &X) const {
		if (lower.size() > 0) {
			return X.cwiseMin(upper).cwiseMax(lower);
		}
		return X;
	}

	void values(const mat &popX, vec &ys) {
		int popsize = popX.cols();
		int n = popX.rows();
		double pargs[popsize*n];
		double res[popsize];
		for (int p = 0; p < popX.cols(); p++) {
			for (int i = 0; i < n; i++)
				pargs[p * n + i] = popX(i, p);
		}
		func_par->evalJava(popsize, n, pargs, res);
		for (int p = 0; p < popX.cols(); p++)
			ys[p] = res[p];
		evaluationCounter += popsize;
	}

	bool feasible(int i, double x) {
		return x >= lower[i] && x <= upper[i];
	}

	vec uniformX(pcg64 &rs) {
		vec rv = uniformVec(lower.size(), rs);
		return (rv.array() * scale.array()).matrix() + lower;
	}

	double uniformXi(int i, pcg64 &rs) {
		return lower[i] + scale[i] * distr_01(rs);
	}

	int getEvaluations() {
		return evaluationCounter;
	}

	vec lower;
	vec upper;

private:
	CallJava *func_par;
	long evaluationCounter;
	vec scale;
};

class ClDeOptimizer {

public:

	ClDeOptimizer(long runid_, Fitness *fitfun_, int dim_, int seed_,
			int popsize_, int maxEvaluations_, double pbest_,
			double stopfitness_,  double K1_, double K2_) {
		// runid used to identify a specific run
		runid = runid_;
		// fitness function to minimize
		fitfun = fitfun_;
		// Number of objective variables/problem dimension
		dim = dim_;
		// Population size
		popsize0 = popsize_ > 0 ? popsize_ : int(dim*8.5+150);
		// maximal number of evaluations allowed.
		maxEvaluations = maxEvaluations_;
		// use low value 0 < pbest <= 1 to narrow search.
		pbest0 = pbest_;
		// Limit for fitness value.
		stopfitness = stopfitness_;
		K1 = K1_;
		K2 = K2_;
		// stop criteria
		stop = 0;
		rs = new pcg64(seed_);
		init();
	}

	~ClDeOptimizer() {
		delete rs;
	}

	double rnd01() {
		return distr_01(*rs);
	}

	int rndInt(int max) {
		return (int) (max * distr_01(*rs));
	}

	void doOptimize() {

		double CR, F;
		vector<vec> sp;

		double stage = 0.5;

		int popsize = popsize0;
		double pbest = pbest0;


		// -------------------- Generation Loop --------------------------------

		for (iterations = 1;; iterations++) {
			// sort population
			ivec sindex = sort_index(nextY);
			popY = nextY(sindex, Eigen::all);
			popX = nextX(Eigen::all, sindex);

			bestX = popX.col(0);
			bestY = popY[0];

			if (isfinite(stopfitness) && bestY < stopfitness) {
				stop = 1;
				return;
			}

			if (fitfun->getEvaluations() >= maxEvaluations)
				return;

			double evals = float(fitfun->getEvaluations()) / maxEvaluations;
			if (evals > stage)
				pbest = pbest0;
			else {
				for (double per = 1.0; per >= 0; per = per - 0.05)
					if (evals*evals < per && evals*evals >= per - 0.05)
						pbest = 1 - per;
			}
			popsize = min(popsize0, max(7*dim, int(popsize0 - (popsize0 - 50)*evals)));

		    mat local_upper(dim, popsize);
		    mat local_lower(dim, popsize);
		    double bound_info[popsize];
			for (int p = 0; p < popsize; p++) {
				vec X = popX.col(p);
				local_upper.col(p) = fitfun->upper.cwiseMin(X);
			    local_lower.col(p) = fitfun->lower.cwiseMax(X);
			    vec norm = fitfun->norm(X);
			    double avnorm = norm.sum() / popsize;
			    bound_info[p] = 0;
			    for (int j = 0; j < dim; j++)
			    	bound_info[p] += abs(norm[j] - avnorm);
			    bound_info[p] /= popsize;
			}

			for (int p = 0; p < popsize; p++) {
				int r1, r2, r3;
				do {
					r1 = rndInt(popsize);
				} while (r1 == p);
				do {
					r2 = rndInt(int(popsize * pbest));
				} while (r2 == p || r2 == r1);
				do {
					r3 = rndInt(popsize + sp.size());
				} while (r3 == p || r3 == r2 || r3 == r1);
				int jr = rndInt(dim);

				if (iterations % 2 == 1)
//					CR = 1;
					CR = normreal(rs, 0.95, 0.01);
				else
//					CR = 0;
					CR = normreal(rs, 0.0, 0.01);

				mat mutationX;
				vec mutationY;
				int mutationI[3];
				if (evals > stage) {
					mutationI[0] = r1;
					mutationI[1] = r2;
					mutationI[2] = r3 < popsize ? r3 : r3-popsize;
					ivec mindex = sort_index(nextY, mutationI, 3);
					mutationY = nextY(mindex, Eigen::all);
					mutationX = nextX(Eigen::all, mindex);
					F = 2*(mutationY[1]-mutationY[0])/(mutationY[2]-mutationY[0]);
				} else {
					if (rnd01() < 0.5)
						F  = normreal(rs, 0.1, 0.04);
					else
						F  = normreal(rs, 1.0, 1.0);
					if (F < 0 || F > 1)
						F = rnd01();
				}
				vec ui = popX.col(p);
				vec ub = local_upper.col(p);
				vec lb = local_lower.col(p);
				for (int j = 0; j < dim; j++) {
					if (j == jr || rnd01() < CR) {
						if (bound_info[j] > 0.1 && evals < K1 && rnd01() > K2)
							ui[j] = ub[j] + lb[j] - popX(j, r1);
						else {
							if (evals > stage) {
								ui[j] = mutationX(j, 0)
										+ F * (mutationX(j, 1) - mutationX(j, 2));
							} else {
								if (r3 < popsize)
									ui[j] = popX(j, r1)
											+ F * (popX(j, r2) - popX(j, r3));
								else
									ui[j] = popX(j, r1)
											+ F * ((popX)(j, r2) - sp[r3 - popsize][j]);
							}
						}
						if (!fitfun->feasible(j, ui[j]))
							ui[j] = fitfun->uniformXi(j, *rs);
					}
				}
				nextX.col(p) = ui;
			}
			fitfun->values(nextX, nextY);
			for (int p = 0; p < popsize; p++) {
				if (nextY[p] < popY[p]) {
					if (sp.size() < popsize)
						sp.push_back(popX.col(p));
					else
						sp[rndInt(popsize)] = popX.col(p);
				} else {    // no improvement, copy from parent
					nextX.col(p) = popX.col(p);
					nextY[p] = popY[p];
				}
			}
		}
	}

	void init() {
		popCR = zeros(popsize0);
		popF = zeros(popsize0);
		nextX = mat(dim, popsize0);
		for (int p = 0; p < popsize0; p++)
			nextX.col(p) = fitfun->uniformX(*rs);
		nextY = vec(popsize0);
		fitfun->values(nextX, nextY);
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

private:
	long runid;
	Fitness *fitfun;
	int popsize0; // population size
	int dim;
	int maxEvaluations;
	double pbest0;
	double stopfitness;
	int iterations;
	double bestY;
	vec bestX;
	int stop;
	double K1;
	double K2;
	pcg64 *rs;
	mat popX;
	vec popY;
	mat nextX;
	vec nextY;
	vec popCR;
	vec popF;
};
}

using namespace cl_differential_evolution;

/*
 * Class:     fcmaes_core_Jni
 * Method:    optimizeCLDE
 * Signature: (Lfcmaes/core/Fitness;[D[D[DIDIDDDJI)I
 */
JNIEXPORT jint JNICALL Java_fcmaes_core_Jni_optimizeCLDE
  (JNIEnv* env, jclass cls, jobject func, jdoubleArray jlower, jdoubleArray jupper, jdoubleArray jinit, 
  jint maxEvals, jdouble stopfitness, jint popsize, jdouble pbest, jdouble K1, jdouble K2, jlong seed, jint runid) {
  	// init java function callback
  	double* init = env->GetDoubleArrayElements(jinit, JNI_FALSE);
	double* lower = env->GetDoubleArrayElements(jlower, JNI_FALSE);
	double* upper = env->GetDoubleArrayElements(jupper, JNI_FALSE);
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
	ClDeOptimizer opt(runid, &fitfun, dim, seed, popsize, maxEvals, pbest,
			stopfitness, K1, K2);
	try {
		opt.doOptimize();
		vec bestX = opt.getBestX();
		double bestY = opt.getBestValue();

		for (int i = 0; i < dim; i++)
			init[i] = bestX[i];

		env->SetDoubleArrayRegion (jinit, 0, dim, (jdouble*)init);
		env->ReleaseDoubleArrayElements(jinit, init, 0);
		env->ReleaseDoubleArrayElements(jupper, upper, 0);
		env->ReleaseDoubleArrayElements(jlower, lower, 0);
		return fitfun.getEvaluations();
		
	} catch (std::exception& e) {
		cout << e.what() << endl;
		return fitfun.getEvaluations();
	}
	return 0;
  }

