package fcmaes.temporal.core;

import static fcmaes.temporal.core.OptimizerRetryWorker.TASK_QUEUE;
import static org.mockito.Matchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;

import io.temporal.client.WorkflowClient;
import io.temporal.client.WorkflowOptions;
import io.temporal.testing.TestWorkflowEnvironment;
import io.temporal.worker.Worker;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

public class OptimizerWorkerTest {

    private TestWorkflowEnvironment testEnv;
    private Worker worker;
    private WorkflowClient workflowClient;

    @Before
    public void setUp() {
        testEnv = TestWorkflowEnvironment.newInstance();
        worker = testEnv.newWorker(TASK_QUEUE);
        worker.registerWorkflowImplementationTypes(OptimizerWorkflowImpl.class);
        workflowClient = testEnv.getWorkflowClient();
    }

    @After
    public void tearDown() {
        testEnv.close();
    }

    @Test
    public void testParallelRetry() {
        OptimizerActivity activities = mock(OptimizerActivity.class);
        worker.registerActivitiesImplementations(activities);
        testEnv.start();
        WorkflowOptions options = WorkflowOptions.newBuilder().setTaskQueue(TASK_QUEUE).build();
        OptimizerWorkflow workflow =
                workflowClient.newWorkflowStub(OptimizerWorkflow.class, options);
        Map<String,String> params = new HashMap<String,String>();
        params.put("fitnessClass", "fcmaes.examples.Solo");
        params.put("optimizerClass", "fcmaes.core.Optimizers$Bite(16)");
        params.put("runs", "20000");
        params.put("maxEvals", "150000");
        params.put("popSize", "31");
        params.put("stopVal", "-1E99");
        params.put("limit", "1E99");
        long startTime = testEnv.currentTimeMillis();
        workflow.optimize(1, params);
        verify(activities).optimize(eq(1), eq(params));
        long duration = testEnv.currentTimeMillis() - startTime;
        System.out.println("Duration: " + duration);
    }
}

