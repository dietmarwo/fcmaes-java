package fcmaes.temporal.core;

import io.temporal.workflow.QueryMethod;
import io.temporal.workflow.SignalMethod;
import io.temporal.workflow.WorkflowInterface;
import io.temporal.workflow.WorkflowMethod;

import java.util.List;
import java.util.Map;

/**
 * Workflow interface for distributed smart/coordinated parallel optimization retry.
 */
@WorkflowInterface
public interface SmartWorkflow {

    @WorkflowMethod
    List<List<Double>> optimize(int num, Map<String, String> params);

    /**
     * Receives new optimum for key.
     */
    @SignalMethod
    void storeFitness(List<Double> ys, List<List<Double>> xs);

    @QueryMethod
    List<List<Double>> getFitness(long minTime);

    @QueryMethod
    List<Double> getYs();

    @QueryMethod
    List<List<Double>> getXs();
}
