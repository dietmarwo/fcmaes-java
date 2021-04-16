package fcmaes.temporal.core;

import fcmaes.core.Fitness;
import fcmaes.core.Optimizers;
import io.temporal.activity.Activity;
import io.temporal.activity.ActivityExecutionContext;
import io.temporal.activity.ActivityInfo;
import io.temporal.client.WorkflowClient;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Activity implementation for parallel distributed optimization retry.
 * Optimization results are sent to the workflow. This activity works
 * independent from the workflow - no data is received after initialization.
 */
class OptimizerActivityImpl implements OptimizerActivity {

    static long interval = 100000; // ms
    WorkflowClient client;
    OptimizerWorkflow workflow;

    // activity state - dependent on queries to the workflow and on the internal optimization
    HashMap<String, Double> ymap = new HashMap<String, Double>();
    HashMap<String, List<Double>> xmap = new HashMap<String, List<Double>>();
    HashMap<String, Long> tmap = new HashMap<String, Long>();
    long time = System.currentTimeMillis();

    public OptimizerActivityImpl(WorkflowClient client) {
        this.client = client;
    }

    // callback injected into the fitness function to send data to workflow.
    // Sending of data is delayed to reduce traffic.
    synchronized private boolean signal(String key, Double y, List<Double> x) {
        if (!ymap.containsKey(key) || y < ymap.get(key)) {
            ymap.put(key, y);
            xmap.put(key, x);
            tmap.put(key, System.currentTimeMillis());
        }
        if (System.currentTimeMillis() - time > interval) {
            int i = 0;
            for (String k : tmap.keySet()) {
                if (tmap.get(k) > time) {
                    // send data to workflow
                    workflow.optimum(k, ymap.get(k), xmap.get(k));
                    i++;
                }
            }
            System.out.println("" + i + " signals sent");
            time = System.currentTimeMillis();
        }
        return false;
    }

    // local parallel retry.
    public String optimize(int index, Map<String, String> params) {
        try {
            String fitnessClass = params.get("fitnessClass");
            String optimizerClass = params.get("optimizerClass");
            int runs = Integer.parseInt(params.get("runs"));
            int maxEvals = Integer.parseInt(params.get("maxEvals"));
            int popSize = Integer.parseInt(params.get("popSize"));
            double stopVal = Double.parseDouble(params.get("stopVal"));
            double limit = Double.parseDouble(params.get("limit"));

            Fitness fit = Utils.buildFitness(fitnessClass);
            Optimizers.Optimizer opt = Utils.buildOptimizer(optimizerClass);

            System.out.println("optimize " + index + " " + fitnessClass + " " + optimizerClass);
            ActivityExecutionContext ctx = Activity.getExecutionContext();
            ActivityInfo info = ctx.getInfo();
            System.out.printf(
                    "\nActivity started: WorkflowID: %s RunID: %s", info.getWorkflowId(), info.getRunId());
            this.workflow =
                    client.newWorkflowStub(OptimizerWorkflow.class, info.getWorkflowId());
            fit._callBack = (key, y, x) -> signal(key, y, x);
            fit.minimizeN(runs, opt, null, maxEvals, stopVal, popSize, limit);
        } catch (Exception ex) {
            ex.printStackTrace();
        }
        return "success";
    }
}
