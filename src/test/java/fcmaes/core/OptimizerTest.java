/* Copyright (c) Dietmar Wolz.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory.
 * 
 * derived from 
 * https://github.com/Hipparchus-Math/hipparchus/blob/master/hipparchus-optim/src/test/java/org/hipparchus/optim/nonlinear/scalar/noderiv/CMAESOptimizerTest.java
 */

package fcmaes.core;

import java.util.Arrays;
import java.util.Random;

import org.hipparchus.util.FastMath;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;

import fcmaes.core.Optimizers.Bite;
import fcmaes.core.Optimizers.CLDE;
import fcmaes.core.Optimizers.CMA;
import fcmaes.core.Optimizers.CMAAT;
import fcmaes.core.Optimizers.CMAPAR;
import fcmaes.core.Optimizers.CSMA;
import fcmaes.core.Optimizers.DA;
import fcmaes.core.Optimizers.DE;
import fcmaes.core.Optimizers.DEAT;
import fcmaes.core.Optimizers.DEPAR;
import fcmaes.core.Optimizers.GCLDE;
import fcmaes.core.Optimizers.Hawks;
import fcmaes.core.Optimizers.Optimizer;
import fcmaes.core.Optimizers.Result;

/**
 * Test for {@link Optimizer}.
 */
@RunWith(RetryRunner.class)

public class OptimizerTest {

	static final int DIM = 13;
	static final int POPSIZE = 31;

	@Test
	@Retry(6)
	public void testRosenCma() {
		double[] guess = point(DIM, 0.5);
		double[] sigma = point(DIM, 0.3);
		double[] lower = point(DIM, -1);
		double[] upper = point(DIM, 1);
		Result expected = new Result(0, 0.0, point(DIM, 1.0));
		Optimizer opt = new CMA();

		doTest(new Rosen(DIM), opt, lower, upper, sigma, guess,
				100000, -Double.MAX_VALUE, POPSIZE, 1e-6, 1e-12, expected);
	}
	
    @Test
    @Retry(6)
    public void testRosenCmaAskTell() {
        double[] guess = point(DIM, 0.5);
        double[] sigma = point(DIM, 0.3);
        double[] lower = point(DIM, -1);
        double[] upper = point(DIM, 1);
        Result expected = new Result(0, 0.0, point(DIM, 1.0));
        Optimizer opt = new CMAAT();

        doTest(new Rosen(DIM), opt, lower, upper, sigma, guess,
                100000, -Double.MAX_VALUE, POPSIZE, 1e-6, 1e-12, expected);
    }

	@Test
	@Retry(6)
	public void testRosenCmaParallelEval() {
		double[] guess = point(DIM, 0.5);
		double[] sigma = point(DIM, 0.3);
		double[] lower = point(DIM, -1);
		double[] upper = point(DIM, 1);
		Result expected = new Result(0, 0.0, point(DIM, 1.0));
		Optimizer opt = new CMA();
		Fitness fitness = new Rosen(DIM);
		fitness._parallelEval = true;
		doTest(fitness, opt, lower, upper, sigma, guess,
				100000, -Double.MAX_VALUE, POPSIZE, 1e-6, 1e-12, expected);
	}

    @Test
    @Retry(6)
    public void testRosenCmaParallelEvalAskTell() {
        double[] guess = point(DIM, 0.5);
        double[] sigma = point(DIM, 0.3);
        double[] lower = point(DIM, -1);
        double[] upper = point(DIM, 1);
        Result expected = new Result(0, 0.0, point(DIM, 1.0));
        Optimizer opt = new CMAPAR();
        Fitness fitness = new Rosen(DIM);
        fitness._parallelEval = true;
        doTest(fitness, opt, lower, upper, sigma, guess, 100000, -Double.MAX_VALUE, POPSIZE, 1e-6, 1e-12, expected);
    }

	@Test
	@Retry(1)
	public void testRosenCmaParallel() {
		double[] guess = point(DIM, 1);
		double[] sigma = point(DIM, 0.3);
		double[] lower = point(DIM, -1);
		double[] upper = point(DIM, 2);
		Result expected = new Result(0, 0.0, point(DIM, 1.0));
		Optimizer opt = new CMA();

		doTestParallel(8, new Rosen(DIM), opt, lower, upper, sigma, guess,
				100000, -Double.MAX_VALUE, POPSIZE, 1e-6, 1e-12, expected);
	}

	@Test
	@Retry(3)
	public void testRosenCmaCoordinated() {
		double[] guess = point(DIM, 1);
		double[] lower = point(DIM, -1);
		double[] upper = point(DIM, 2);
		Result expected = new Result(0, 0.0, point(DIM, 1.0));
		Optimizer opt = new CMA();

		doTestCoordinated(100, new Rosen(DIM), opt, lower, upper, guess,
				1000, 1e-6, 1e-12, expected);
	}

	@Test
	@Retry(3)
	public void testRosenDE() {
		double[] guess = point(DIM, 1);
		double[] sigma = point(DIM, 0.3);
		double[] lower = point(DIM, -1);
		double[] upper = point(DIM, 2);
		Result expected = new Result(0, 0.0, point(DIM, 1.0));
		Optimizer opt = new DE();

		doTest(new Rosen(DIM), opt, lower, upper, sigma, guess,
				100000, -Double.MAX_VALUE, POPSIZE, 1e-6, 1e-12, expected);
	}
	
    @Test
    @Retry(3)
    public void testRosenDEAskTell() {
        double[] guess = point(DIM, 1);
        double[] sigma = point(DIM, 0.3);
        double[] lower = point(DIM, -1);
        double[] upper = point(DIM, 2);
        Result expected = new Result(0, 0.0, point(DIM, 1.0));
        Optimizer opt = new DEAT();

        doTest(new Rosen(DIM), opt, lower, upper, sigma, guess, 100000, -Double.MAX_VALUE, POPSIZE, 1e-6, 1e-12,
                expected);
    }
    
    @Test
    @Retry(3)
    public void testRosenDEParallelEvalAskTell() {
        double[] guess = point(DIM, 1);
        double[] sigma = point(DIM, 0.3);
        double[] lower = point(DIM, -1);
        double[] upper = point(DIM, 2);
        Result expected = new Result(0, 0.0, point(DIM, 1.0));
        Optimizer opt = new DEPAR();

        doTest(new Rosen(DIM), opt, lower, upper, sigma, guess, 100000, -Double.MAX_VALUE, POPSIZE, 1e-6, 1e-12,
                expected);
    }
	
	@Test
	@Retry(6)
	public void testRosenCLDE() {
		double[] guess = point(DIM, 1);
		double[] sigma = point(DIM, 0.3);
		double[] lower = point(DIM, -1);
		double[] upper = point(DIM, 2);
		Result expected = new Result(0, 0.0, point(DIM, 1.0));
		Optimizer opt = new CLDE();

		doTest(new Rosen(DIM), opt, lower, upper, sigma, guess,
				100000, -Double.MAX_VALUE, POPSIZE, 1e-1, 1e-2, expected);
	}

	@Test
	@Retry(3)
	public void testRosenGCLDE() {
		double[] guess = point(DIM, 1);
		double[] sigma = point(DIM, 0.3);
		double[] lower = point(DIM, -1);
		double[] upper = point(DIM, 2);
		Result expected = new Result(0, 0.0, point(DIM, 1.0));
		Optimizer opt = new GCLDE();

		doTest(new Rosen(DIM), opt, lower, upper, sigma, guess,
				100000, -Double.MAX_VALUE, POPSIZE, 1e-6, 1e-12, expected);
	}

	@Test
	@Retry(1)
	public void testRosenDEParallel() {
		double[] guess = point(DIM, 1);
		double[] sigma = point(DIM, 0.3);
		double[] lower = point(DIM, -1);
		double[] upper = point(DIM, 2);
		Result expected = new Result(0, 0.0, point(DIM, 1.0));
		Optimizer opt = new DE();

		doTestParallel(8, new Rosen(DIM), opt, lower, upper, sigma, guess,
				100000, -Double.MAX_VALUE, POPSIZE, 1e-6, 1e-12, expected);
	}
	
	@Test
	@Retry(3)
	public void testRosenDA() {
		double[] guess = point(DIM, 1);
		double[] sigma = point(DIM, 0.3);
		double[] lower = point(DIM, -1);
		double[] upper = point(DIM, 2);
		Result expected = new Result(0, 0.0, point(DIM, 1.0));
		Optimizer opt = new DA();

		doTest(new Rosen(DIM), opt, lower, upper, sigma, guess,
				100000, -Double.MAX_VALUE, POPSIZE, 1e-6, 1e-12, expected);
	}

	@Test
	@Retry(3)
	public void testRosenBite() {
		double[] guess = point(DIM, 1);
		double[] sigma = point(DIM, 0.3);
		double[] lower = point(DIM, -1);
		double[] upper = point(DIM, 2);
		Result expected = new Result(0, 0.0, point(DIM, 1.0));
		Optimizer opt = new Bite();

		doTest(new Rosen(DIM), opt, lower, upper, sigma, guess,
				100000, -Double.MAX_VALUE, POPSIZE, 1e-6, 1e-12, expected);
	}

	@Test
	@Retry(3)
	public void testRosenCSMA() {
		double[] guess = point(DIM, 1);
		double[] sigma = point(DIM, 0.3);
		double[] lower = point(DIM, -1);
		double[] upper = point(DIM, 2);
		Result expected = new Result(0, 0.0, point(DIM, 1.0));
		Optimizer opt = new CSMA();

		doTest(new Rosen(DIM), opt, lower, upper, sigma, guess,
				100000, -Double.MAX_VALUE, POPSIZE, 1e-6, 1e-12, expected);
	}

	@Test
	@Retry(3)
	public void testRosenDAParallel() {
		double[] guess = point(DIM, 1);
		double[] sigma = point(DIM, 0.3);
		double[] lower = point(DIM, -1);
		double[] upper = point(DIM, 2);
		Result expected = new Result(0, 0.0, point(DIM, 1.0));
		Optimizer opt = new DA();

		doTestParallel(8, new Rosen(DIM), opt, lower, upper, sigma, guess,
				100000, -Double.MAX_VALUE, POPSIZE, 1e-6, 1e-12, expected);
	}

	@Test
	@Retry(3)
	public void testRosenHawks() {
		double[] guess = point(DIM, 1);
		double[] sigma = point(DIM, 0.3);
		double[] lower = point(DIM, -1);
		double[] upper = point(DIM, 2);
		Result expected = new Result(0, 0.0, point(DIM, 1.0));
		Optimizer opt = new Hawks();

		doTest(new Rosen(DIM), opt, lower, upper, sigma, guess,
				100000, -Double.MAX_VALUE, POPSIZE, 5e5, 5e-3, expected);
	}

	@Test
	@Retry(1)
	public void testRosenHawksParallel() {
		double[] guess = point(DIM, 1);
		double[] sigma = point(DIM, 0.3);
		double[] lower = point(DIM, -1);
		double[] upper = point(DIM, 2);
		Result expected = new Result(0, 0.0, point(DIM, 1.0));
		Optimizer opt = new Hawks();

		doTestParallel(8, new Rosen(DIM), opt, lower, upper, sigma, guess,
				100000, -Double.MAX_VALUE, POPSIZE, 1e5, 1e-3, expected);
	}

	@Test
	@Retry(3)
	public void testEllipseCma() {
		double[] guess = point(DIM, 1.0);
		double[] sigma = point(DIM, 0.1);
		double[] lower = point(DIM, -1);
		double[] upper = point(DIM, 2);
		Result expected = new Result(0, 0.0, point(DIM, 0.0));
		Optimizer opt = new CMA();

		doTest(new Elli(DIM, 1e3), opt, lower, upper, sigma, guess,
				100000, -Double.MAX_VALUE, POPSIZE, 1e-5, 1e-10, expected);
	}

	@Test
	@Retry(3)
	public void testElliRotatedCma() {
		double[] guess = point(DIM, 1.0);
		double[] sigma = point(DIM, 0.1);
		double[] lower = point(DIM, -1);
		double[] upper = point(DIM, 2);
		Result expected = new Result(0, 0.0, point(DIM, 0.0));
		Optimizer opt = new CMA();

		doTest(new ElliRotated(DIM, 1e3), opt, lower, upper, sigma, guess,
				100000, -Double.MAX_VALUE, POPSIZE, 1e-6, 1e-10, expected);
	}

	@Test
	@Retry(3)
	public void testCigarCma() {
		double[] guess = point(DIM, 0.5);
		double[] sigma = point(DIM, 0.1);
		double[] lower = point(DIM, -1);
		double[] upper = point(DIM, 1);
		Result expected = new Result(0, 0.0, point(DIM, 0.0));
		Optimizer opt = new CMA();

		doTest(new Cigar(DIM, 1e3), opt, lower, upper, sigma, guess,
				100000, -Double.MAX_VALUE, POPSIZE, 1e-5, 1e-11, expected);
	}
	
	@Test
	@Retry(3)
	public void testTwoAxesCma() {
		double[] guess = point(DIM, 0.5);
		double[] sigma = point(DIM, 0.1);
		double[] lower = point(DIM, -1);
		double[] upper = point(DIM, 1);
		Result expected = new Result(0, 0.0, point(DIM, 0.0));
		Optimizer opt = new CMA();

		doTest(new TwoAxes(DIM, 1e3), opt, lower, upper, sigma, guess,
				100000, -Double.MAX_VALUE, POPSIZE, 1e-5, 1e-11, expected);
	}

	@Test
	@Retry(3)
	public void testCigTab() {
		double[] guess = point(DIM, 0.5);
		double[] sigma = point(DIM, 0.3);
		double[] lower = point(DIM, -1);
		double[] upper = point(DIM, 1);
		Result expected = new Result(0, 0.0, point(DIM, 0.0));
		Optimizer opt = new CMA();

		doTest(new CigTab(DIM, 1e4), opt, lower, upper, sigma, guess,
				100000, -Double.MAX_VALUE, POPSIZE, 5e-5, 1e-10, expected);
	}

	@Test
	@Retry(3)
	public void testSphere() {
		double[] guess = point(DIM, 0.5);
		double[] sigma = point(DIM, 0.3);
		double[] lower = point(DIM, -1);
		double[] upper = point(DIM, 1);
		Result expected = new Result(0, 0.0, point(DIM, 0.0));
		Optimizer opt = new CMA();

		doTest(new Sphere(DIM), opt, lower, upper, sigma, guess,
				100000, -Double.MAX_VALUE, POPSIZE, 1e-5, 1e-11, expected);
	}

	@Test
	@Retry(3)
	public void testTablet() {
		double[] guess = point(DIM, 0.5);
		double[] sigma = point(DIM, 0.3);
		double[] lower = point(DIM, -1);
		double[] upper = point(DIM, 1);
		Result expected = new Result(0, 0.0, point(DIM, 0.0));
		Optimizer opt = new CMA();

		doTest(new Tablet(DIM, 1e3), opt, lower, upper, sigma, guess,
				100000, -Double.MAX_VALUE, POPSIZE, 1e-5, 1e-11, expected);
	}

	@Test
	@Retry(3)
	public void testDiffPow() {
		double[] guess = point(DIM, 0.5);
		double[] sigma = point(DIM, 0.1);
		double[] lower = point(DIM, -1);
		double[] upper = point(DIM, 1);
		Result expected = new Result(0, 0.0, point(DIM, 0.0));
		Optimizer opt = new CMA();

		doTest(new DiffPow(DIM), opt, lower, upper, sigma, guess,
				100000, -Double.MAX_VALUE, POPSIZE, 2e-1, 1e-8, expected);
	}

	@Test
	@Retry(3)
	public void testSsDiffPow() {
		double[] guess = point(DIM, 0.5);
		double[] sigma = point(DIM, 0.1);
		double[] lower = point(DIM, -1);
		double[] upper = point(DIM, 1);
		Result expected = new Result(0, 0.0, point(DIM, 0.0));
		Optimizer opt = new CMA();

		doTest(new SsDiffPow(DIM), opt, lower, upper, sigma, guess,
				100000, -Double.MAX_VALUE, POPSIZE, 1e-1, 1e-4, expected);
	}

	@Test
	@Retry(3)
	public void testAckley() {
		double[] guess = point(DIM, 0.5);
		double[] sigma = point(DIM, 0.1);
		double[] lower = point(DIM, -1);
		double[] upper = point(DIM, 1);
		Result expected = new Result(0, 0.0, point(DIM, 0.0));
		Optimizer opt = new CMA();

		doTest(new Ackley(DIM, 1), opt, lower, upper, sigma, guess,
				100000, -Double.MAX_VALUE, POPSIZE, 1e-5, 1e-9, expected);
	}

	@Test
	@Retry(3)
	public void testRastrigin() {
		double[] guess = point(DIM, 0.1);
		double[] sigma = point(DIM, 0.1);
		double[] lower = point(DIM, -1);
		double[] upper = point(DIM, 1);
		Result expected = new Result(0, 0.0, point(DIM, 0.0));
		Optimizer opt = new CMA();

		doTest(new Rastrigin(DIM, 1, 10), opt, lower, upper, sigma, guess,
				100000, -Double.MAX_VALUE, POPSIZE, 1e-6, 1e-11, expected);
	}

	/**
	 * @param fit         	Function to optimize.
	 * @param opt         	Optimizer used.
	 * @param lower     	lower point limit.
	 * @param upper     	upper point limit.
	 * @param sigma        	Individual input sigma.
	 * @param guess     	Starting point.
	 * @param maxEvals 		Maximum number of evaluations.
	 * @param stopVal      	Termination criteria for optimization.
	 * @param popsize       Population size used for offspring.
	 * @param xTol       	Tolerance for checking that the optimum is correct.
	 * @param yTol          Tolerance relative error on the objective function.
	 * @param expected      Expected point / value.
	 */
	private void doTest(Fitness fit, Optimizer opt, double[] lower, double[] upper, double[] sigma, double[] guess,
			int maxEvals, double stopVal, int popsize, double xTol, double yTol, Result expected) {
		Result result = opt.minimize(fit, lower, upper, sigma, guess, maxEvals, stopVal, popsize);		
		Assert.assertArrayEquals(expected.X, result.X, xTol);
		Assert.assertEquals(expected.y, result.y, yTol);
		Assert.assertTrue(result.evals > 0);
	}

	/**
	 * @param runs         	number of parallel optimization runs.
	 * @param fit         	Function to optimize.
	 * @param opt         	Optimizer used.
	 * @param lower     	lower point limit.
	 * @param upper     	upper point limit.
	 * @param sigma        	Individual input sigma.
	 * @param guess     	Starting point.
	 * @param maxEvals 		Maximum number of evaluations.
	 * @param stopVal      	Termination criteria for optimization.
	 * @param popsize       Population size used for offspring.
	 * @param xTol       	Tolerance for checking that the optimum is correct.
	 * @param yTol          Tolerance relative error on the objective function.
	 * @param expected      Expected point / value.
	 */
	private void doTestParallel(int runs, Fitness fit, Optimizer opt, double[] lower, double[] upper, double[] sigma, double[] guess,
			int maxEvals, double stopVal, int popsize, double xTol, double yTol, Result expected) {
		Result result = opt.minimizeN(runs, fit, lower, upper, sigma, guess, maxEvals, stopVal, popsize, 0, null);		
		Assert.assertArrayEquals(expected.X, result.X, xTol);
		Assert.assertEquals(expected.y, result.y, yTol);
		Assert.assertTrue(result.evals > 0);
	}

	/**
	 * @param runs         	number of parallel optimization runs.
	 * @param fit         	Function to optimize.
	 * @param opt         	Optimizer used.
	 * @param lower     	lower point limit.
	 * @param upper     	upper point limit.
	 * @param sigma        	Individual input sigma.
	 * @param guess     	Starting point.
	 * @param maxEvals 		Maximum number of evaluations.
	 * @param stopVal      	Termination criteria for optimization.
	 * @param popsize       Population size used for offspring.
	 * @param xTol       	Tolerance for checking that the optimum is correct.
	 * @param yTol          Tolerance relative error on the objective function.
	 * @param expected      Expected point / value.
	 */
	private void doTestCoordinated(int runs, Fitness fit, Optimizer opt, double[] lower, double[] upper, double[] guess,
			int startEvals, double xTol, double yTol, Result expected) {
	    Result result = CoordRetry.optimize(runs, fit, opt, guess, Double.MAX_VALUE, startEvals, false);
		Assert.assertArrayEquals(expected.X, result.X, xTol);
		Assert.assertEquals(expected.y, result.y, yTol);
		Assert.assertTrue(result.evals > 0);
	}

	private static double[] point(int n, double value) {
		double[] ds = new double[n];
		Arrays.fill(ds, value);
		return ds;
	}

	private static class Sphere extends Fitness {

		public Sphere(int dim) {
			super(dim);
		}

		public Sphere create() {
			return new Sphere(_dim);
		}

		public double eval(double[] x) {
			double f = 0;
			for (int i = 0; i < x.length; ++i)
				f += x[i] * x[i];
			return f;
		}
	}

	private static class Cigar extends Fitness {
		private double factor;

		Cigar(int dim, double axisratio) {
			super(dim);
			factor = axisratio * axisratio;
		}
		
		public Cigar create() {
			return new Cigar(_dim, factor);
		}

		public double eval(double[] x) {
			double f = x[0] * x[0];
			for (int i = 1; i < x.length; ++i)
				f += factor * x[i] * x[i];
			return f;
		}
	}

	private static class Tablet extends Fitness {
		private double factor;

		Tablet(int dim, double axisratio) {
			super(dim);
			factor = axisratio * axisratio;
		}
		
		public Tablet create() {
			return new Tablet(_dim, 1e3);
		}

		public double eval(double[] x) {
			double f = factor * x[0] * x[0];
			for (int i = 1; i < x.length; ++i)
				f += x[i] * x[i];
			return f;
		}
	}

	private static class CigTab extends Fitness {
		private double factor;

		CigTab(int dim, double axisratio) {
			super(dim);
			factor = axisratio;
		}
		
		public CigTab create() {
			return new CigTab(_dim, factor);
		}
		
		public double eval(double[] x) {
			int end = x.length - 1;
			double f = x[0] * x[0] / factor + factor * x[end] * x[end];
			for (int i = 1; i < end; ++i)
				f += x[i] * x[i];
			return f;
		}
	}

	private static class TwoAxes extends Fitness {

		private double factor;

		TwoAxes(int dim, double axisratio) {
			super(dim);
			factor = axisratio * axisratio;
		}
		
		public TwoAxes create() {
			return new TwoAxes(_dim, factor);
		}

		public double eval(double[] x) {
			double f = 0;
			for (int i = 0; i < x.length; ++i)
				f += (i < x.length / 2 ? factor : 1) * x[i] * x[i];
			return f;
		}
	}

	private static class ElliRotated extends Fitness {
		private Basis B = new Basis();
		private double factor;

		ElliRotated(int dim, double axisratio) {
			super(dim);
			factor = axisratio * axisratio;
		}
		
		public ElliRotated create() {
			return new ElliRotated(_dim, factor);
		}

		public double eval(double[] x) {
			double f = 0;
			x = B.Rotate(x);
			for (int i = 0; i < x.length; ++i)
				f += FastMath.pow(factor, i / (x.length - 1.)) * x[i] * x[i];
			return f;
		}
	}

	private static class Elli extends Fitness {

		double factor;

		Elli(int dim, double axisratio) {
			super(dim);
			factor = axisratio * axisratio;
		}
		
		public Elli create() {
			return new Elli(_dim, factor);
		}
	
		public double eval(double[] x) {
			double f = 0;
			for (int i = 0; i < x.length; ++i)
				f += FastMath.pow(factor, i / (x.length - 1.)) * x[i] * x[i];
			return f;
		}
	}

	private static class MinusElli extends Fitness {

		Elli elli;
		double factor;
		
		MinusElli(int dim, double axisratio) {
			super(dim);
			factor = axisratio * axisratio;
			elli = new Elli(_dim, axisratio);
		}
		
		public MinusElli create() {
			return new MinusElli(_dim, factor);
		}

		public double eval(double[] x) {
			return 1.0 - (elli.eval(x));
		}
	}

	private static class DiffPow extends Fitness {

		public DiffPow(int dim) {
			super(dim);
		}

		public DiffPow create() {
			return new DiffPow(_dim);
		}

		public double eval(double[] x) {
			double f = 0;
			for (int i = 0; i < x.length; ++i)
				f += FastMath.pow(FastMath.abs(x[i]), 2. + 10 * (double) i / (x.length - 1.));
			return f;
		}
	}

	private static class SsDiffPow extends Fitness {

		DiffPow diffPow;
		
		public SsDiffPow(int dim) {
			super(dim);
			diffPow = new DiffPow(dim);
		}

		public SsDiffPow create() {
			return new SsDiffPow(_dim);
		}

		public double eval(double[] x) {
			double f = FastMath.pow(diffPow.eval(x), 0.25);
			return f;
		}
	}

	private static class Rosen extends Fitness {

		public Rosen(int dim) {
			super(dim);
		}

		public Rosen create() {
			return new Rosen(_dim);
		}

		public double[] lower() {
			return point(_dim, -1);
		}
		
		public double[] upper() {
			return point(_dim, 1);
		}

		public double eval(double[] x) {
			double f = 0;
			for (int i = 0; i < x.length - 1; ++i)
				f += 1e2 * (x[i] * x[i] - x[i + 1]) * (x[i] * x[i] - x[i + 1]) + (x[i] - 1.) * (x[i] - 1.);
			return f;
		}
	}

	private static class Ackley extends Fitness {
		
		private double axisratio;
		
		Ackley(int dim, double axisratio) {
			super(dim);
			this.axisratio = axisratio;
		}
		
		public Ackley create() {
			return new Ackley(_dim, axisratio);
		}

		public double eval(double[] x) {
			double f = 0;
			double res2 = 0;
			double fac = 0;
			for (int i = 0; i < x.length; ++i) {
				fac = FastMath.pow(axisratio, (i - 1.) / (x.length - 1.));
				f += fac * fac * x[i] * x[i];
				res2 += FastMath.cos(2. * FastMath.PI * fac * x[i]);
			}
			f = (20. - 20. * FastMath.exp(-0.2 * FastMath.sqrt(f / x.length)) + FastMath.exp(1.)
					- FastMath.exp(res2 / x.length));
			return f;
		}
	}

	private static class Rastrigin extends Fitness {

		private double axisratio;
		private double amplitude;

		Rastrigin(int dim, double axisratio, double amplitude) {
			super(dim);
			this.axisratio = axisratio;
			this.amplitude = amplitude;
		}
		
		public Rastrigin create() {
			return new Rastrigin(_dim, axisratio, amplitude);
		}

		public double eval(double[] x) {
			double f = 0;
			double fac;
			for (int i = 0; i < x.length; ++i) {
				fac = FastMath.pow(axisratio, (i - 1.) / (x.length - 1.));
				if (i == 0 && x[i] < 0)
					fac *= 1.;
				f += fac * fac * x[i] * x[i] + amplitude * (1. - FastMath.cos(2. * FastMath.PI * fac * x[i]));
			}
			return f;
		}
	}

	private static class Basis {
		double[][] basis;
		Random rand = new Random(2); // use not always the same basis

		double[] Rotate(double[] x) {
			GenBasis(x.length);
			double[] y = new double[x.length];
			for (int i = 0; i < x.length; ++i) {
				y[i] = 0;
				for (int j = 0; j < x.length; ++j)
					y[i] += basis[i][j] * x[j];
			}
			return y;
		}

		void GenBasis(int DIM) {
			if (basis != null ? basis.length == DIM : false)
				return;

			double sp;
			int i, j, k;

			/* generate orthogonal basis */
			basis = new double[DIM][DIM];
			for (i = 0; i < DIM; ++i) {
				/* sample components gaussian */
				for (j = 0; j < DIM; ++j)
					basis[i][j] = rand.nextGaussian();
				/* substract projection of previous vectors */
				for (j = i - 1; j >= 0; --j) {
					for (sp = 0., k = 0; k < DIM; ++k)
						sp += basis[i][k] * basis[j][k]; /* scalar product */
					for (k = 0; k < DIM; ++k)
						basis[i][k] -= sp * basis[j][k]; /* substract */
				}
				/* normalize */
				for (sp = 0., k = 0; k < DIM; ++k)
					sp += basis[i][k] * basis[i][k]; /* squared norm */
				for (k = 0; k < DIM; ++k)
					basis[i][k] /= FastMath.sqrt(sp);
			}
		}
	}
}
