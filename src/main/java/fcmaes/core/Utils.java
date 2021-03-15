/* Copyright (c) Dietmar Wolz.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory.
 */

package fcmaes.core;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

import org.hipparchus.geometry.euclidean.threed.Vector3D;
import org.hipparchus.random.RandomGenerator;

/**
 * Utilities.
 */

public class Utils {

    public static final double DAY = 24 * 3600;
    
    public static final double YEAR = 365.25;
    
    public static long _startTime = System.nanoTime();

    public static void startTiming() {
        _startTime = System.nanoTime();
    }

    public static double measuredMillis() {
        long endTime = System.nanoTime();
        return (endTime - _startTime) / 1000000.0;
    }

    public static double measuredSeconds() {
        long endTime = System.nanoTime();
        return (endTime - _startTime) / 1000000000.0;
    }

    public static ThreadLocalRandom rnd() {
        return ThreadLocalRandom.current();
    }

    public static boolean rnd(double prob) {
        return rnd().nextDouble() < prob;
    }

    public static boolean rnd(RandomGenerator r, double prob) {
        return r.nextDouble() < prob;
    }

    public static int rndInt(int min, int max) {
        return min + rnd().nextInt(max - min);
    }

    public static double rnd(RandomGenerator r, double min, double max) {
        return min + r.nextDouble() * (max - min);
    }

    public static double rnd(double min, double max) {
        return min + rnd().nextDouble() * (max - min);
    }

    public static double[] rnd(double min, double max, int n) {
        double[] rnds = new double[n];
        for (int i = 0; i < n; i++)
            rnds[i] = rnd(min, max);
        return rnds;
    }

    public static double[] rnd(double[] min, double[] max) {
        double[] rnds = new double[min.length];
        for (int i = 0; i < min.length; i++)
            rnds[i] = rnd(min[i], max[i]);
        return rnds;
    }

    public static boolean isNaN(double[] x) {
        for (double d : x)
            if (Double.isNaN(d) || Double.isInfinite(d)) 
                return true;
        return false;
    }

    public static String r(Vector3D v) {
        return "(" + r(v.getX(), 3) + ", " + r(v.getY(), 3) + ", " + r(v.getZ(), 3) + ")";
    }

    public static double r(double d) {
        return r(d, 3);
    }

    public static double r(double d, int i) {
        if (Math.abs(d) < Math.pow(0.1, i))
            return 0;
        else {
            double d1 = 1.0D;
            for (int j = 0; j < i; j++)
                d1 *= 10D;
            return (double) java.lang.Math.round(d * d1) / d1;
        }
    }

    public static String r(double[] v, int n) {
        StringBuilder buf = new StringBuilder();
        buf.append('[');
        for (int i = 0; i < v.length; i++) {
            buf.append(r(v[i], n));
            if (i < v.length - 1)
                buf.append(',');
        }
        buf.append(']');
        return buf.toString();
    }

    public static String r(int[][] v) {
        StringBuilder buf = new StringBuilder();
        buf.append('[');
        for (int i = 0; i < v.length; i++) {
            buf.append(v[i][0]);
            buf.append(':');
            buf.append(v[i][1]);
            if (i < v.length - 1)
                buf.append(',');
        }
        buf.append(']');
        return buf.toString();
    }

    public static Vector3D vector(double[] v) {
        return new Vector3D(v[0], v[1], v[2]);
    }

    public static double[] array(Vector3D v) {
        return new double[] { v.getX(), v.getY(), v.getZ() };
    }

    public static double[] array(int n, double v) {
        double[] m = new double[n];
        Arrays.fill(m, v);
        return m;
    }

    public static double norm(double[] x) {
        double sum = 0;
        for (int i = 0; i < x.length; i++)
            sum += x[i] * x[i];
        return Math.sqrt(sum);
    }

    public static double[] plus(double[] x1, double[] x2) {
        double[] m = new double[x1.length];
        for (int i = 0; i < x1.length; i++)
            m[i] = x1[i] + x2[i];
        return m;
    }

    public static double[] minus(double[] x1, double[] x2) {
        double[] m = new double[x1.length];
        for (int i = 0; i < x1.length; i++)
            m[i] = x1[i] - x2[i];
        return m;
    }

    public static double[] quot(double[] x1, double[] x2) {
        double[] m = new double[x1.length];
        for (int i = 0; i < x1.length; i++)
            m[i] = x1[i] / x2[i];
        return m;
    }

    public static double[] minusAbs(double[] x1, double[] x2) {
        double[] m = new double[x1.length];
        for (int i = 0; i < x1.length; i++)
            m[i] = Math.abs(x1[i] - x2[i]);
        return m;
    }

    public static double[] sprod(double[] v, double f) {
        double[] m = new double[v.length];
        for (int i = 0; i < v.length; i++)
            m[i] = Math.abs(v[i] * f);
        return m;
    }

    public static double[] fitting(double[] v, double min, double max) {
        double[] m = new double[v.length];
        for (int i = 0; i < v.length; i++)
            m[i] = Math.min(Math.max(v[i], min), max);
        return m;
    }

    public static double[] fitting(double[] v, double[] min, double[] max) {
        double[] m = new double[v.length];
        for (int i = 0; i < v.length; i++)
            m[i] = Math.min(Math.max(v[i], min[i]), max[i]);
        return m;
    }

    public static double[] maximum(double[] v, double d) {
        double[] m = new double[v.length];
        for (int i = 0; i < v.length; i++)
            m[i] = Math.max(v[i], d);
        return m;
    }

    public static double[] minimum(double[] v, double d) {
        double[] m = new double[v.length];
        for (int i = 0; i < v.length; i++)
            m[i] = Math.min(v[i], d);
        return m;
    }

    public static boolean lower(double[] v, double[] max) {
        for (int i = 0; i < v.length; i++)
            if (v[i] > max[i])
                return false;
        return true;
    }

    public static double[] maximum(double[] v, double[] max) {
        double[] m = new double[v.length];
        for (int i = 0; i < v.length; i++)
            m[i] = Math.max(v[i], max[i]);
        return m;
    }

    public static double[] minimum(double[] v, double[] d) {
        double[] m = new double[v.length];
        for (int i = 0; i < v.length; i++)
            m[i] = Math.min(v[i], d[i]);
        return m;
    }

    public static double sum(double[] v) {
        double sum = 0;
        for (double vi : v)
            sum += vi;
        return sum;
    }

    public static double sum(List<Double> v) {
        double sum = 0;
        for (double vi : v)
            sum += vi;
        return sum;
    }

    public static double sumAbs(double[] v) {
        double sum = 0;
        for (double vi : v)
            sum += Math.abs(vi);
        return sum;
    }

}
