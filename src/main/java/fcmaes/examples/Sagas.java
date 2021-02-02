package fcmaes.examples;

import java.io.FileNotFoundException;
import java.io.IOException;

import fcmaes.core.CoordRetry;
import fcmaes.core.Fitness;
import fcmaes.core.Log;
import fcmaes.core.Optimizers.DECMA;
import fcmaes.core.Optimizers.Optimizer;
import fcmaes.core.Optimizers.Result;
import fcmaes.core.Utils;

public class Sagas extends GtopProblem {

    /*
     * This example is taken from https://www.esa.int/gsp/ACT/projects/gtop/
     * See also http://www.midaco-solver.com/index.php/about/benchmarks/gtopx
     * where you find implementations in different programming languages. 
     * 
     * "coord(new DECMA(), runs))" solves the problem on average in about 15 seconds 
     * on a modern 16-core CPU (AMD 5950x).
     * Can you provide a faster parallel algorithm in any language?
     */

    @Override
    public double[] lower() {
        return new double[] { 7000, 0, 0, 0, 50, 300, 0.01, 0.01, 1.05, 8, -Math.PI, -Math.PI };
    }

    @Override
    public double[] upper() {
        return new double[] { 9100, 7, 1, 1, 2000, 2000, 0.9, 0.9, 7, 500, Math.PI,  Math.PI };
    }

    public Sagas() {
        super(12);
    }

    public Sagas create() {
        return new Sagas();
    }

    @Override
    public double eval(double[] point) {
        try {
            return Jni.sagas_C(point);
        } catch (Exception ex) {
            return 1E10;
        }
    }
    
    double limitVal() {
        return 100;
    }

    double stopVal() {
        return 18.1877 * stopValFac();
    }

    public static void main(String[] args) throws FileNotFoundException {
        Log.setLog();
        Utils.startTiming();
        Optimizer opt = new DECMA();
        new Sagas().test(100, opt, 2000);
    }

}
