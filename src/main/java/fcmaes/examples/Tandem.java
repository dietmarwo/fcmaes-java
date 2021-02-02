package fcmaes.examples;

import java.io.FileNotFoundException;
import java.io.IOException;

import fcmaes.core.CoordRetry;
import fcmaes.core.Fitness;
import fcmaes.core.Log;
import fcmaes.core.Optimizers.DECMA;
import fcmaes.core.Optimizers.DEGCLDECMA;
import fcmaes.core.Optimizers.Optimizer;
import fcmaes.core.Optimizers.Result;
import fcmaes.core.Utils;

public class Tandem extends GtopProblem {

    /*
     * This example is taken from https://www.esa.int/gsp/ACT/projects/gtop/
     * See also http://www.midaco-solver.com/index.php/about/benchmarks/gtopx
     * where you find implementations in different programming languages. 
     * 
     * "coord(new DECMA(), runs))" solves the problem on average in about 330 seconds 
     * on a modern 16-core CPU (AMD 5950x).
     * Can you provide a faster parallel algorithm?
     * 
     * Note that Tandem is a problem were DEGCLDECMA() could be 
     * stronger than DECMA() (but slightly slower) for coordinated retry. 
     * 
     * Note that according to https://www.esa.int/gsp/ACT/projects/gtop/tandem_con
     * it took nearly 5 years until Paul Musegaas from TU Delft in 2013 found the 
     * best solution for the constraint EVEES variant of this problem used here. 
     * The best solution found before (from Dario Izzo) scored 1.7% lower.  
     */

    @Override
    public double[] lower() {
        return new double[] { 5475, 2.5, 0, 0, 20, 20, 20, 20, 0.01, 0.01, 0.01, 0.01, 1.05, 1.05, 1.05, -Math.PI,
                -Math.PI, -Math.PI };
    }

    @Override
    public double[] upper() {
        return new double[] { 9132, 4.9, 1, 1, 2500, 2500, 2500, 2500, 0.99, 0.99, 0.99, 0.99, 10, 10, 10, Math.PI,
                Math.PI, Math.PI };
    }

    public Tandem() {
        super(18);
    }

    public Tandem create() {
        return new Tandem();
    }

    @Override
    public double eval(double[] point) {
        try {
            int[] seq = new int[] { 3, 2, 3, 3, 6 };
            return Jni.tandem_C(point, seq);
        } catch (Exception ex) {
            return 1E10;
        }
    }
 
    double limitVal() {
        return -300.0;
    }
 
    double stopVal() {
        return -1500.46 / stopValFac();
    }

    public static void main(String[] args) throws FileNotFoundException {
        Log.setLog();
        Utils.startTiming();
        Optimizer opt = new DECMA();
//        Optimizer opt = new DEGCLDECMA();
        new Tandem().test(100, opt, 20000);
    }

}
