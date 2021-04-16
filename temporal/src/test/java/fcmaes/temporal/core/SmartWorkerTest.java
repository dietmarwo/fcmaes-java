package fcmaes.temporal.core;

import static fcmaes.temporal.core.SmartRetryWorker.TASK_QUEUE;
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

public class SmartWorkerTest {

    private TestWorkflowEnvironment testEnv;
    private Worker worker;
    private WorkflowClient workflowClient;

    @Before
    public void setUp() {
        testEnv = TestWorkflowEnvironment.newInstance();
        worker = testEnv.newWorker(TASK_QUEUE);
        worker.registerWorkflowImplementationTypes(SmartWorkflowImpl.class);
        workflowClient = testEnv.getWorkflowClient();
    }

    @After
    public void tearDown() {
        testEnv.close();
    }

    @Test
    public void testTransfer() {
        SmartActivity activities = mock(SmartActivity.class);
        worker.registerActivitiesImplementations(activities);
        testEnv.start();
        WorkflowOptions options = WorkflowOptions.newBuilder().setTaskQueue(TASK_QUEUE).build();
        SmartWorkflow workflow =
                workflowClient.newWorkflowStub(SmartWorkflow.class, options);
        long starty = testEnv.currentTimeMillis();
        //workflow.transfer("account1", "account2", "reference1", 123);
        //verify(activities).withdraw(eq("account1"), eq("reference1"), eq(123));
        //verify(activities).deposit(eq("account2"), eq("reference1"), eq(123));
        long duration = testEnv.currentTimeMillis() - starty;
        System.out.println("Duration: " + duration);
    }
}

