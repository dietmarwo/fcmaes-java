package fcmaes.temporal.core;

import fcmaes.core.Fitness;
import fcmaes.core.Optimizers;

import java.lang.reflect.Constructor;

/**
 * Utilities to create Optimizer and Fitness objects from its String representation.
 */
public class Utils {

    public static Fitness buildFitness(String fitnessClass) {
        try {
            return (Fitness) Class.forName(fitnessClass).getConstructor().newInstance();
        } catch (Exception ex) {
            System.err.println("Fitness class " + fitnessClass + " not in classpath.");
            return null;
        }
    }

    public static Optimizers.Optimizer buildOptimizer(String optimizerClass) {
        try {
            if (optimizerClass.contains("(")) {
                int arg = Integer.parseInt(optimizerClass.substring(
                        optimizerClass.indexOf("(")+1, optimizerClass.length()-1));
                optimizerClass = optimizerClass.substring(0, optimizerClass.indexOf("("));
                Constructor c = getConstructor(optimizerClass, 1);
                return (Optimizers.Optimizer) c.newInstance(arg);
            } else
                return (Optimizers.Optimizer) Class.forName(optimizerClass).getConstructor().newInstance();
        } catch (Exception ex) {
            System.err.println("Fitness class " + optimizerClass + " not in classpath.");
            return null;
        }
    }

    private static Constructor getConstructor(String className, int argNum) throws ClassNotFoundException {
        Constructor[] cs = Class.forName(className).getConstructors();
        for (Constructor c: cs)
            if (c.getParameterTypes().length == argNum) return c;
        return null;
    }

}
