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

public class Rosetta extends GtopProblem {

    /*
     * This example is taken from https://www.esa.int/gsp/ACT/projects/gtop/
     * See also http://www.midaco-solver.com/index.php/about/benchmarks/gtopx
     * where you find implementations in different programming languages. 
     * 
     * "coord(new DECMA(), runs))" solves the problem on average in about 15 seconds 
     * on a modern 16-core CPU (AMD 5950x).
     * Can you provide a faster parallel algorithm?
     */

    @Override
    public double[] lower() {
        return new double[] { 1460, 3, 0, 0, 300, 150, 150, 300, 700, 0.01, 0.01, 0.01, 0.01, 0.01, 1.05, 1.05, 1.05,
                1.05, -Math.PI, -Math.PI, -Math.PI, -Math.PI };
    }

    @Override
    public double[] upper() {
        return new double[] { 1825, 5, 1, 1, 500, 800, 800, 800, 1850, 0.9, 0.9, 0.9, 0.9, 0.9, 9, 9, 9, 9, Math.PI,
                Math.PI, Math.PI, Math.PI };
    }

    public Rosetta() {
        super(22);
    }

    public Rosetta create() {
        return new Rosetta();
    }

    @Override
    public double eval(double[] point) {
        try {
            return Jni.rosetta_C(point);
        } catch (Exception ex) {
            return 1E10;
        }
    }
    
    double limitVal() {
        return 20.0;
    }
 
    double stopVal() {
        return stopValFac()*1.3433;
    }
   
    public static void main(String[] args) throws FileNotFoundException {
        Log.setLog();
        Utils.startTiming();
        Optimizer opt = new DECMA();
        new Rosetta().test(100, opt, 4000);
    }

}
