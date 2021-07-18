/* Copyright (c) Dietmar Wolz.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory.
 */

package fcmaes.examples;

import java.io.IOException;

import com.nativeutils.NativeUtils;

public class Jni {

    public static native double gtoc1_C(double[] x, int[] seq, int[] rev, double dvLaunch, double[] rp, double[] dv);

    public static native double gtoc1part_C(double[] x, int[] seq, int[] rev, double dvLaunch, double[] rp,
            double[] dv);

    public static native double gtoc1back_C(double[] x, int[] seq, int[] rev, double dvLaunch, double[] rp,
            double[] dv);

    public static native double cassini1_C(double[] x, double[] rp);

    public static native double sagas_C(double[] x);

    public static native double rosetta_C(double[] x);

    public static native double cassini2_C(double[] x);

    public static native double messenger_C(double[] x);

    public static native double messengerfull_C(double[] x);

    public static native double messengerpart_C(double[] x, int[] sequence, double[] dv);

    public static native double insertionDSM_C(double[] x, int[] sequence, double eIns, double rpIns, double[] dv);

    public static native double messengermga_C(double[] x, int[] sequence, double[] dv);

    public static native double tandem_C(double[] x, int[] sequence);

    public static double gtoc1(double[] x, int[] seq, int[] rev, double dvLaunch, double[] rp, double[] dv) {
        return gtoc1_C(x, seq, rev, dvLaunch, rp, dv);
    }

    public static double gtoc1part(double[] x, int[] seq, int[] rev, double dvLaunch, double[] rp, double[] dv) {
        return gtoc1part_C(x, seq, rev, dvLaunch, rp, dv);
    }

    public static double gtoc1back(double[] x, int[] seq, int[] rev, double dvLaunch, double[] rp, double[] dv) {
        return gtoc1back_C(x, seq, rev, dvLaunch, rp, dv);
    }

    public static double cassini1(double[] x, double[] rp) {
        return cassini1_C(x, rp);
    }

    public static double sagas(double[] x) {
        return sagas_C(x);
    }

    public static double rosetta(double[] x) {
        return rosetta_C(x);
    }

    public static double cassini2(double[] x) {
        return cassini2_C(x);
    }

    public static double messenger(double[] x) {
        return messenger_C(x);
    }

    public static double messengerfull(double[] x) {
        return messengerfull_C(x);
    }

    public static double messengerpart(double[] x, int[] sequence, double[] dv) {
        return messengerpart_C(x, sequence, dv);
    }

    public static double messengermga(double[] x, int[] sequence, double[] dv) {
        return messengermga_C(x, sequence, dv);
    }

    public static double tandem(double[] x, int[] sequence) {
        return tandem_C(x, sequence);
    }
    
    public static native double[] bounds_re_C(String problem);

    public static native double[] objectives_re_C(String problem, double[] x);
    
}
