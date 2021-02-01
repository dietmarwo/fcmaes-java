package fcmaes.examples;

import fcmaes.core.CoordRetry;
import fcmaes.core.Fitness;
import fcmaes.core.Utils;
import fcmaes.core.Optimizers.DECMA;
import fcmaes.core.Optimizers.Optimizer;
import fcmaes.core.Optimizers.Result;

public class GtopProblem extends Fitness {

    public GtopProblem(int dim) {
        super(dim);
    }   

    double limitVal() {
        return Double.POSITIVE_INFINITY;
    }
 
    double stopVal() {
        return Double.POSITIVE_INFINITY;
    }

    double stopValFac() {
        return 1.005;
    }

    Result optimize(int num, int maxEvals) {
        Optimizer opt = new DECMA();
        double[] sdev = Utils.array(_dim, 0.07);
        return minimizeN(num, opt, lower(), upper(), null, sdev, 10000, stopVal(), 31, limitVal());
    }

    Result coord(Optimizer opt, int num) {
        return CoordRetry.optimize(num, this, opt, null, limitVal(), stopVal(), 1500, true);
    }
    
    void test(int retries, Optimizer opt, int num) {
        System.out.println("Testing coordinated retry " + opt.getClass().getName().split("\\$")[1] + " " + 
                this.getClass().getName().split("\\.")[2] + " stopVal = " + Utils.r(stopVal()));
        for (int i = 0; i < retries; i++) {
            Utils.startTiming();
            Result res = coord(opt, num);
            System.out.println(i + ": " + 
                    "best = " + res.y + ", time = " + 0.001 * Utils.measuredMillis() + 
                    " sec, evals = " + res.evals);        }
    }


}
