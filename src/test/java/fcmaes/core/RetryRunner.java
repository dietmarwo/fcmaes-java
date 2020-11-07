package fcmaes.core;

import org.junit.runners.BlockJUnit4ClassRunner;
import org.junit.runners.model.FrameworkMethod;
import org.junit.runners.model.InitializationError;
import org.junit.runners.model.Statement;

public class RetryRunner extends BlockJUnit4ClassRunner {

	public RetryRunner(final Class<?> testClass) throws InitializationError {
		super(testClass);
	}

	@Override
	public Statement methodInvoker(final FrameworkMethod method, Object test) {
		final Statement singleTryStatement = super.methodInvoker(method, test);
		return new Statement() {

			@Override
			public void evaluate() throws Throwable {
				Throwable failureReason = null;

				final Retry retry = method.getAnnotation(Retry.class);
				if (retry == null) {
					singleTryStatement.evaluate();
				} else {
					final int numRetries = retry.value();

					for (int i = 0; i < numRetries; ++i) {
						try {
							singleTryStatement.evaluate();
							return;
						} catch (Throwable t) {
							failureReason = t;
						}
					}
					throw failureReason;
				}
			}
		};
	}
}
