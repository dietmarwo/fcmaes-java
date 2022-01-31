/* Copyright (c) Dietmar Wolz.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory.
 */

package fcmaes.core;

/**
 * Statistics utility class.
 */

public class Statistics {

    private double ai;
    private double qi;
    private int i;
    private double max = Double.NEGATIVE_INFINITY;
    private double min = Double.POSITIVE_INFINITY;

    public void clear() {
        ai = qi = i = 0;
        max = Double.NEGATIVE_INFINITY;
        min = Double.POSITIVE_INFINITY;
    }

    public double getAi() {
        return ai;
    }

    public double getQi() {
        return qi;
    }

    public int getI() {
        return i;
    }

    public double getMax() {
        return max;
    }

    public double getMin() {
        return min;
    }

    public synchronized void add(double value) {
        if (max < value)
            max = value;
        if (min > value)
            min = value;
        i++;
        if (i == 1)
            ai = value;
        else {
            qi += (i - 1) * (value - ai) * (value - ai) / i;
            ai += (value - ai) / i;
        }
    }

    public double sampleDev() {
        if (i <= 1)
            return 0;
        else
            return Math.sqrt(qi / (i - 1));
    }

    public double standardDev() {
        if (i == 0)
            return 0;
        else
            return Math.sqrt(qi / i);
    }

    public double conf() {
        if (i < 10)
            return standardDev() < 10 ? 10 : standardDev();
        else
            return standardDev() / Math.sqrt(i);
    }

    class Correlation {
        Statistics x;
        Statistics y;
        double sumX;
        double sumY;
        double sumXY;

        synchronized void add(double valueX, double valueY) {
            if (x == null) {
                x = new Statistics();
                y = new Statistics();
            }
            x.add(valueX);
            y.add(valueY);
            sumX += valueX;
            sumY += valueY;
            sumXY += valueX * valueY;
        }

        double correlation() {
            if (x.i == y.i && x.i > 1)
                return (sumXY - sumX * sumY / x.i) / ((x.i - 1) * x.sampleDev() * y.sampleDev());
            else
                return 0;
        }
    }

    public double norm(double d) {
        double sdev = standardDev();
        if (sdev == 0)
            return 0;
        else
            return (d - ai) / sdev;
    }

    public String toString() {
        return "n=" + i + " m=" + Utils.r(ai) + " sd=" + Utils.r(standardDev()) + " l=" + Utils.r(min) + " u="
                + Utils.r(max);
    }
    
    public String toString(int n) {
        return "n=" + i + " m=" + Utils.r(ai, n) + " sd=" + Utils.r(standardDev(), n) + " l=" + Utils.r(min, n) + " u="
                + Utils.r(max, n);
    }

}
