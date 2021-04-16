package fcmaes.temporal.core;

import io.temporal.workflow.QueryMethod;
import io.temporal.workflow.SignalMethod;
import io.temporal.workflow.WorkflowInterface;
import io.temporal.workflow.WorkflowMethod;

import java.util.List;
import java.util.Map;

/**
 * Workflow interface for distributed parallel optimization retry.
 */
@WorkflowInterface
public interface OptimizerWorkflow {

    @WorkflowMethod
    Map<String, List<Double>> optimize(int num, Map<String, String> params);

    /**
     * Receives new optimum for key.
     */
    @SignalMethod
    void optimum(String key, double y, List<Double> x);

    @QueryMethod
    Map<String, Double> getYMap();

    @QueryMethod
    Map<String, List<Double>> getXMap();
}
