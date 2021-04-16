package fcmaes.temporal.core;

import io.temporal.activity.ActivityOptions;
import io.temporal.common.RetryOptions;
import io.temporal.workflow.Async;
import io.temporal.workflow.Promise;
import io.temporal.workflow.Workflow;
import io.temporal.workflow.WorkflowInfo;

import java.net.InetAddress;
import java.net.UnknownHostException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Workflow implementation for distributed parallel optimization retry .
 * Optimization results are sent to the workflow.
 * This activity depends on data received from the workflow which is merged
 * with the internal data to represent the global state of the optimization.
 */
public class OptimizerWorkflowImpl implements OptimizerWorkflow {

    // workflow state - only dependent on the signals sent to the workflow
    ConcurrentHashMap<String, Double> ymap = new ConcurrentHashMap<String, Double>();
    ConcurrentHashMap<String, List<Double>> xmap = new ConcurrentHashMap<String, List<Double>>();
    ConcurrentHashMap<String, Long> tmap = new ConcurrentHashMap<String, Long>();

    private final OptimizerActivity optimizer =
            Workflow.newActivityStub(
                    OptimizerActivity.class,
                    ActivityOptions.newBuilder()
                            .setScheduleToCloseTimeout(Duration.ofSeconds(20000))
                            .setHeartbeatTimeout(Duration.ofSeconds(20000))
                            .setRetryOptions(
                                    RetryOptions.newBuilder()
                                            .setMaximumAttempts(100)
                                            .setInitialInterval(Duration.ofSeconds(100))
                                            .build())
                            .build());

    // start workflow
    @Override
    public Map<String, List<Double>> optimize(int num, Map<String, String> params) {
        WorkflowInfo wi = Workflow.getInfo();
        try {
            System.out.printf(
                    "\nWorkflow started: WorkflowID: %s RunID: %s, Namespace: %s, Host: %s",
                    wi.getWorkflowId(), wi.getRunId(), wi.getNamespace(), InetAddress.getLocalHost());
        } catch (UnknownHostException e) {
        }

        List<Promise<String>> promises = new ArrayList<Promise<String>>();
        for (int i = 1; i <= num; i++) {
            System.out.println("optimization " + i + " requested");
            Promise<String> pr = Async.function(optimizer::optimize, i, params);
            promises.add(pr);
        }
        for (Promise<String> pr : promises) {
            String index = pr.get();
            System.out.println("optimization " + index + " ended");
        }
        return xmap;
    }

    // signal
    @Override
    public void optimum(String key, double y, List<Double> x) {
        if (!ymap.containsKey(key) || y < ymap.get(key)) {
            ymap.put(key, y);
            xmap.put(key, x);
            tmap.put(key, System.currentTimeMillis());
            System.out.println("signal " + key + " " + y + " " + Arrays.toString(x.toArray()));
        }
    }

    // query
    @Override
    public Map<String, Double> getYMap() {
        return ymap;
    }

    // query
    @Override
    public Map<String, List<Double>> getXMap() {
        return xmap;
    }
}
