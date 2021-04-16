package fcmaes.temporal.core;

import fcmaes.core.CoordRetry;
import fcmaes.core.Fitness;
import fcmaes.core.Optimizers;
import fcmaes.core.Threads;
import io.temporal.activity.Activity;
import io.temporal.activity.ActivityExecutionContext;
import io.temporal.activity.ActivityInfo;
import io.temporal.client.WorkflowClient;
import org.apache.commons.lang3.ArrayUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * Activity implementation for distributed smart/coordinated parallel optimization retry.
 */
class SmartActivityImpl extends CoordRetry implements SmartActivity {

    WorkflowClient client;
    SmartWorkflow workflow;

    SmartActivityImpl(WorkflowClient client) {
        this.client = client;
    }

    // local smart/coordinated parallel retry.
    public String optimize(int index, Map<String, String> params) {
        try {
            String fitnessClass = params.get("fitnessClass");
            String optimizerClass = params.get("optimizerClass");
            int runs = Integer.parseInt(params.get("runs"));
            int startEvals = Integer.parseInt(params.get("startEvals"));
            int popSize = Integer.parseInt(params.get("popSize"));
            double stopVal = Double.parseDouble(params.get("stopVal"));
            double limitVal = Double.parseDouble(params.get("limit"));
            boolean log = false;

            Fitness fit = Utils.buildFitness(fitnessClass);
            Optimizers.Optimizer opt = Utils.buildOptimizer(optimizerClass);

            System.out.println("optimize " + index + " " + fitnessClass + " " + optimizerClass);
            ActivityExecutionContext ctx = Activity.getExecutionContext();
            ActivityInfo info = ctx.getInfo();
            System.out.printf(
                    "\nActivity started: WorkflowID: %s RunID: %s", info.getWorkflowId(), info.getRunId());

            workflow = client.newWorkflowStub(SmartWorkflow.class, info.getWorkflowId());
            Fitness[] store = new Fitness[500];
            Optimize retry = new Optimize(
                    runs, fit, null, store, limitVal, stopVal,
                    startEvals, 0, 0, opt, popSize, log, workflow);
            Threads threads = new Threads(retry, 8);
            threads.start();
            threads.join();
            retry.sort();
            retry.dump(retry.countRuns());
        } catch (Exception ex) {
            ex.printStackTrace();
        }
        return "success";
    }

    // Adapts CoordRetry.Optimize to exchange optimization results with the workflow
    static class Optimize extends CoordRetry.Optimize {

        long _lastTransferTime = 0;
        SmartWorkflow workflow;

        public Optimize(int runs, Fitness fit, double[] guess, Fitness[] store, double limitVal,
                        double stopVal, int startEvals,
                        int maxEvalFac, int checkInterval, Optimizers.Optimizer opt, int popsize, boolean log,
                        SmartWorkflow workflow) {
            super(runs, fit, guess, store, limitVal, stopVal, startEvals,
                    maxEvalFac, checkInterval, opt, popsize, log);
            this.workflow = workflow;
        }

        public Optimize(String fitnessClass, Fitness[] store, boolean log) {
            super(Utils.buildFitness(fitnessClass), store, log);
        }

        // Calls super.sort() and then exchanges optimization results with the workflow.
        // Data exchange is delayed and restricted to a specific time frame to reduce traffic.
        @Override
        public void sort() {
            super.sort();
            long now = System.currentTimeMillis();
            if (workflow != null && _lastTransferTime + 10000 < now) {
                //communicate with temporal server
                List<Double> ys = new ArrayList<Double>();
                List<List<Double>> xs = new ArrayList<List<Double>>();
                sinceLastTransfer(_lastTransferTime, ys, xs);
                // receive data from workflow
                List<List<Double>> xsOthers = workflow.getFitness(_lastTransferTime);
                System.out.println("query " + (now - _lastTransferTime) + " " + xsOthers.size());
                // send data to workflow
                workflow.storeFitness(ys, xs);
                System.out.println("signal " + ys.size());
                _lastTransferTime = now;
                int last = xsOthers.size() - 1;
                if (last > 0)
                    add(xsOthers.get(last), xsOthers.subList(0, last));
            }
        }

        // integrates optimization results received from the workflow
        public void add(List<Double> ys, List<List<Double>> xs) {
            for (int i = 0; i < ys.size(); i++) {
                double[] x = ArrayUtils.toPrimitive(xs.get(i).toArray(Double[]::new));
                Fitness fit = new Fitness(x.length);
                fit._bestX = x;
                fit._bestY = ys.get(i);
                addResult(countRuns(), fit, Double.MAX_VALUE);
            }
        }

        // limits optimization results to a specific time frame since the last transfer.
        void sinceLastTransfer(long minTime, List<Double> ys, List<List<Double>> xs) {
            for (int i = 0; i < _numStored; i++) {
                Fitness fit = (Fitness) _store[i];
                if (fit._time > minTime) {
                    ys.add(fit._bestY);
                    xs.add(Arrays.asList(ArrayUtils.toObject(fit._bestX)));
                }
            }
        }

        // extracts fitness values from the local fitness store.
        List<Double> getYs() {
            List<Double> ys = new ArrayList<Double>();
            for (int i = 0; i < _numStored; i++) {
                Fitness fit = (Fitness) _store[i];
                ys.add(fit._bestY);
            }
            return ys;
        }

        // extracts argument vectors from the local fitness store.
        List<List<Double>> getXs() {
            List<List<Double>> xs = new ArrayList<List<Double>>();
            for (int i = 0; i < _numStored; i++) {
                Fitness fit = (Fitness) _store[i];
                xs.add(Arrays.asList(ArrayUtils.toObject(fit._bestX)));
            }
            return xs;
        }
    }
}
