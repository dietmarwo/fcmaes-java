/* Copyright (c) Dietmar Wolz.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory.
 */

package fcmaes.examples;

import org.hipparchus.ode.ODEIntegrator;
import org.hipparchus.ode.ODEState;
import org.hipparchus.ode.OrdinaryDifferentialEquation;
import org.hipparchus.ode.nonstiff.DormandPrince853Integrator;

import fcmaes.core.Fitness;
import fcmaes.core.Jni;
import fcmaes.core.Optimizers.DECMA;
import fcmaes.core.Optimizers.Optimizer;
import fcmaes.core.Optimizers.Result;
import fcmaes.core.Utils;

/*
 * This example is taken from https://mintoc.de/index.php/F-8_aircraft
 * The F-8 aircraft control problem is based on a very simple aircraft model. 
 * The control problem was introduced by Kaya and Noakes and aims at controlling 
 * an aircraft in a time-optimal way from an initial state to a terminal state.
 * 
 * The code in "public static void main(String[] args)" solves the problem
 * in a fraction of a second on a modern 16-core CPU. 
 * Can you provide a faster parallel algorithm?
 */

public class F8 extends Fitness {

    @Override
    public double[] lower() {
        return Utils.array(_dim, 0);
    }

    @Override
    public double[] upper() {
        return Utils.array(_dim, 2);
    }

    public F8(int dim) {
        super(dim);
    }

    static double ksi = 0.05236;
    static double ksi_2 = ksi * ksi;
    static double ksi_3 = ksi * ksi_2;

    static class F8Equations implements OrdinaryDifferentialEquation {

        double w;

        F8Equations(double w) {
            this.w = w;
        }

        public int getDimension() {
            return 3;
        }

        public double[] computeDerivatives(final double t, final double[] y) {
            double y0 = y[0];
            double y0_2 = y0 * y0;
            double y0_3 = y0_2 * y0;

            double y1 = y[1];
            double y1_2 = y1 * y1;

            double y2 = y[2];

            return new double[] {
                    -0.877 * y0 + y2 - 0.088 * y0 * y2 + 0.47 * y0_2 - 0.019 * y1_2 - y0_2 * y2 + 3.846 * y0_3
                            + 0.215 * ksi - 0.28 * y0_2 * ksi + 0.47 * y0 * ksi_2 - 0.63 * ksi_3
                            - (0.215 * ksi - 0.28 * y0_2 * ksi - 0.63 * ksi_3) * 2 * w,
                    y2,
                    -4.208 * y0 - 0.396 * y2 - 0.47 * y0_2 - 3.564 * y0_3 + 20.967 * ksi - 6.265 * y0_2 * ksi
                            + 46 * y0 * ksi_2 - 61.4 * ksi_3
                            - (20.967 * ksi - 6.265 * y0_2 * ksi - 61.4 * ksi_3) * 2 * w };
        }
    }

    public double evalJava(double[] X) {
        ODEIntegrator dp853 = new DormandPrince853Integrator(1.0e-8, 100.0, 1.0e-10, 1.0e-10);
        F8Equations ode = new F8Equations(0);

        double t = 0.;
        double[] y = new double[] { 0.4655, 0., 0. };
        ODEState state = new ODEState(t, y);
        int n = X.length;
        for (int i = 0; i < n; i++) {
            if (X[i] == 0)
                continue;
            // bang-bang type switches starting with w(t) = 1.
            ode.w = (i + 1) % 2;
            state = dp853.integrate(ode, state, t + X[i]);
            t = state.getTime();
            y = state.getPrimaryState();
        }
        double val0 = Utils.sum(X);
        double penalty = Utils.sumAbs(y);
        double val = 0.1 * val0 + penalty;
        if (val < _bestY)
            System.out.println(val0 + " " + penalty + " " + val + " " + _evals);
        return val;
    }

    public double evalCpp(double[] X) {
        double[] y = new double[] { 0.4655, 0., 0. };
        int n = X.length;
        for (int i = 0; i < n; i++) {
            if (X[i] == 0)
                continue;
            // bang-bang type switches starting with w(t) = 1.
            double w = (i + 1) % 2;
            Jni.integrateF8(y, w, X[i], 0.1);
        }
        double val0 = Utils.sum(X);
        double penalty = Utils.sumAbs(y);
        double val = 0.1 * val0 + penalty;
        if (val < _bestY)
            System.out.println(val0 + " " + penalty + " " + val + " " + _evals);
        return val;
    }

    public double eval(double[] X) {

        try {
            return evalCpp(X);
//			return evalJava(X);
        } catch (Exception ex) {
            return 1E10; // fail
        }
    }

    public F8 create() {
        return new F8(_dim);
    }

    public static void main(String[] args) {

        Utils.startTiming();

        int dim = 6;
        double[] lower = Utils.array(dim, 0);
        double[] upper = Utils.array(dim, 2);
        double[] sigma = Utils.array(dim, 0.1);

        int maxEvals = 400000;
        int popsize = 31;

        double stopVal = 0.380;

        Utils.startTiming();
        Fitness func = new F8(dim);
        func._stopVal = stopVal;
		Optimizer opt = new DECMA();
		Result res = opt.minimizeN(32, func, lower, upper, sigma, null, maxEvals, stopVal, popsize, 0);
//        Result res = opt.minimize(func, lower, upper, sigma, guess, maxEvals, stopVal, popsize);
        System.out.println("best = " + res.y 
                + ", time = " + 0.001*Utils.measuredMillis() + " sec, evals = " + res.evals);
    }

}
