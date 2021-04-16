package fcmaes.temporal.core;

import io.temporal.activity.ActivityInterface;

import java.util.Map;

/**
 * Activity interface for distributed smart/coordinated parallel optimization retry.
 */
@ActivityInterface
public interface SmartActivity {
    String optimize(int index, Map<String, String> params);
}
