package fcmaes.temporal.core;

import io.temporal.activity.ActivityInterface;

import java.util.Map;

/**
 * Activity interface for parallel distributed optimization retry.
 */
@ActivityInterface
public interface OptimizerActivity {
    String optimize(int index, Map<String, String> params);
}
