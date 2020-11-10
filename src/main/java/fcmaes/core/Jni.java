/* Copyright (c) Dietmar Wolz.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory.
 */

package fcmaes.core;

import java.io.IOException;
import java.util.Arrays;

import com.nativeutils.NativeUtils;

import fcmaes.core.Optimizers.CMA;
import fcmaes.core.Optimizers.Optimizer;
import fcmaes.core.Optimizers.Result;

/**
 * Native optimization functions and ODE integrators. 
 */

public class Jni {
	
	static {
		try {
			NativeUtils.loadLibraryFromJar(("/natives/"+System.mapLibraryName("fcmaeslib"))); 
		} catch (IOException e1) {
			throw new RuntimeException(e1);
		}
	}
		
	public static native int optimizeACMA(Fitness func, double[] lower, double[] upper, double[] sigma, double[] guess, 
			  int maxIter, int maxEvals, double stopValue, int popsize, int mu, double accuracy, long seed, int runid);

	public static native int optimizeHawks(Fitness func, double[] lower, double[] upper, double[] guess, 
			  int maxEvals, double stopfitness, int popsize, long seed, int runid);

	public static native int optimizeDE(Fitness func, double[] lower, double[] upper, double[] guess, 
			  int maxEvals, double stopfitness, int popsize, double keep, double F, double CR, long seed, int runid);

	public static native int optimizeDE2(Fitness func, double[] lower, double[] upper, double[] guess, 
			  int maxEvals, double stopfitness, int popsize, double keep, double K1, double K2, long seed, int runid);

	public static native int optimizeLDE(Fitness func, double[] lower, double[] upper, double[] guess, double[] sigma, 
		    int maxEvals, double stopfitness, int popsize, double keep, double F, double CR, long seed, int runid);

	public static native int optimizeGCLDE(Fitness func, double[] lower, double[] upper, double[] guess, 
			  int maxEvals, double stopfitness, int popsize, double pbest, double F0, double CR0, long seed, int runid);

	public static native int optimizeCLDE(Fitness func, double[] lower, double[] upper, double[] guess, 
			  int maxEvals, double stopfitness, int popsize, double pbest, double K1, double K2, long seed, int runid);

	public static native int optimizeLCLDE(Fitness func, double[] lower, double[] upper, double[] guess, double[] sigma, 
			  int maxEvals, double stopfitness, int popsize, double pbest, double F0, double CR0, long seed, int runid);

	public static native int optimizeDA(Fitness func, double[] lower, double[] upper, double[] guess, 
			  int maxEvals, int use_local_search, long seed, int runid);

	public static native int integrateF8(double[] y, double w, double dt, double step);
	
	public static native int integratePVthrust(double[] tpv, double veff, 
		    double ux, double uy, double uz, double dt, double step);

	public static native int integratePVgtocX(double[] tpv, double dt, double step);

	public static native int integratePVCtoc11(double[] tpv, double dt, double step, boolean dopri);

	public static native int integratePVCtoc11R(double[] tpv, double dt, double step, boolean dopri);

}
