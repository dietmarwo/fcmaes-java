package fcmaes.examples;

import java.io.FileNotFoundException;
import java.util.Arrays;

import org.jtransforms.fft.FloatFFT_2D;

import boofcv.alg.filter.misc.AverageDownSampleOps;
import boofcv.io.image.UtilImageIO;
import boofcv.struct.image.GrayF32;
import fcmaes.core.Cmaes;
import fcmaes.core.CoordRetry;
import fcmaes.core.De;
import fcmaes.core.Fitness;
import fcmaes.core.Log;
import fcmaes.core.Optimizers.Optimizer;
import fcmaes.core.Optimizers.Result;
import fcmaes.core.Utils;
import pl.edu.icm.jlargearrays.ConcurrencyUtils;

public class Interferometry extends Fitness {

    /*
     * This example code is derived from python code posted
     * on https://gitter.im/pagmo2/Lobby by Markus Märtens @CoolRunning 
     * Uses https://github.com/wendykierp/JTransforms and 
     * https://github.com/lessthanoptimal/BoofCV
     * Corresponds to the equivalent Python example 
     * https://github.com/dietmarwo/fast-cma-es/blob/master/examples/interferometry.py
     * The test image used is here: https://api.optimize.esa.int/data/interferometry/orion.jpg
     */
   
    static int n_points_;
    static int image_size_;

    static float[][] fft_;
    static float[][] im_array_;
    
    static int tabsize = 10000;
    static double dtheta = 2 * Math.PI / tabsize;
    static double[] cos_theta = new double[tabsize];
    static double[] sin_theta = new double[tabsize];
    static {
        for (int i = 0; i < tabsize; i++) {
            double theta = i * dtheta;
            cos_theta[i] = Math.cos(theta);
            sin_theta[i] = Math.sin(theta);        
        }
    }
    
    @Override
    public double[] lower() {
        return Utils.array(_dim, -1);
    }

    @Override
    public double[] upper() {
        return Utils.array(_dim, 1);
    }

    public Interferometry(int n_points) {
        super(2*n_points);
    }
        
    public Interferometry(String img, int n_points, int image_size) {
        super(2*n_points);
        n_points_ = n_points;
        image_size_ = image_size;
        fft_ = get_fft(img);
    }

    public Interferometry create() {
        return new Interferometry(n_points_);
    }
    
    static double bestY_ = Double.MAX_VALUE;
    static double evals_ = 0;

    @Override
    public double eval(double[] point) {
        try {
            float[][] observed = get_observed(n_points_, fft_, point);
            FloatFFT_2D fft = new FloatFFT_2D(image_size_, image_size_);
            fft.complexInverse(observed, true);
            double sum = 0;
            for (int i = 0; i < image_size_; i++)
                for (int j = 0; j < image_size_; j++) {
                    double diff = observed[i][2*j] - im_array_[i][j]; // use real, ignore img
                    sum += diff*diff;
                }
            double val = sum/(image_size_*image_size_); // mean_squared_error
            evals_++;
            if (val < bestY_) {
                bestY_ = val;
                System.out.println("" + evals_ + " " + val + " " + Utils.measuredSeconds() + " s " + 
                        Arrays.toString(point));
            }
            return val;
        } catch (Exception ex) {
            return 1E10;
        }
    }
    
    double limitVal() {
        return 1E99;
    }

    double stopVal() {
        return -1E99;
    }
        
    float[][] get_fft(String img) {
        GrayF32 image = UtilImageIO.loadImage(img, GrayF32.class);
        GrayF32 out = new GrayF32(image_size_, image_size_);
        AverageDownSampleOps.down(image, out);
        float[] data = out.getData();
        float[][] im_fft = new float[image_size_][2*image_size_];
        im_array_ = new float[image_size_][image_size_];
        for (int i = 0; i < image_size_; i++) {
            for (int j = 0; j < image_size_; j++) {
                float ub = data[i*image_size_+j]; 
                im_fft[i][j] = ub;
                im_array_[i][j] = ub;
            }
        }
        FloatFFT_2D fft = new FloatFFT_2D(image_size_, image_size_);
        fft.realForwardFull(im_fft);
        return im_fft;
    }
    
    static int to_int(double v) {
        return (int)Math.floor(v);
    }
    
    static float[][] get_observed(int n, float[][] im_fft, double[] args) {
        int r = im_fft.length;
        int c = im_fft[0].length/2;
        double l = 0.01;
        double[] x = Arrays.copyOfRange(args, 0, n);
        double[] y = Arrays.copyOfRange(args, n, 2*n);
        int nq = n*n;
        double[] lx = new double[nq]; 
        double[] ly = new double[nq];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                lx[i*n+j] = x[i] - x[j];
                ly[i*n+j] = y[i] - y[j];
            }
        int[][] obs_uv_matrix = new int[r][c]; 
        double q = Math.pow(2, 2.5);
        for (int i = 0; i < tabsize; i++)
            for (int j = 0; j < nq; j++) {
                double cti = cos_theta[i];
                double sti = sin_theta[i];
                int full_re_u = to_int((lx[j]*cti + ly[j]*sti)/l);
                int full_re_v = to_int((lx[j]*-sti + ly[j]*cti)/l);
                int xij = to_int(full_re_u * r / q * l);
                int yij = to_int(full_re_v * r / q * l);  
                if (xij < 0) xij += r;
                if (yij < 0) yij += c;                              
                obs_uv_matrix[xij][yij] = 1;
            }
        float[][] observed = new float[r][2*c];
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                if (obs_uv_matrix[i][j] == 1) {
                    observed[i][2*j] = im_fft[i][2*j]; // real
                    observed[i][2*j+1] = im_fft[i][2*j+1]; // imag
                }
            }
        }
        return observed;
    }

    // Differential Evolution parallel function evaluation
    void de_parallel(int max_evals) {
        Result res = De.minimize_parallel(this, lower(), upper(), null, max_evals, 0, 32, 200, 0.5, 0.9,
                Utils.rnd().nextLong(), 0, 32);
        System.out.println(res.evals + ": " + _bestY + " time = " + Utils.measuredMillis());
    }

    // CMA-ES parallel function evaluation
    void cma_parallel(int max_evals) {
        Result res = Cmaes.minimize_parallel(this, lower(), upper(), null, null, max_evals, 0, 32, 16, 1,
                Utils.rnd().nextLong(), 0, 1, -1, 32);
        System.out.println(res.evals + ": " + _bestY + " time = " + Utils.measuredMillis());
    }

    // parallel retry
    void retry(Optimizer opt, int retries) {
        System.out.println("Testing parallel retry " + opt.getClass().getName().split("\\$")[1] + " " + 
                this.getClass().getName().split("\\.")[2] + " stopVal = " + Utils.r(stopVal()));
        double[] sdev = Utils.array(_dim, 0.07);
        Utils.startTiming();
        Result res = minimizeN(retries, opt, lower(), upper(), null, sdev, 100000, stopVal(), 31, limitVal());
        System.out.println("best = " + res.y + ", time = " + Utils.measuredSeconds() + 
                " sec, evals = " + res.evals + " x = " + Arrays.toString(res.X));
    }
    
    // smart retry
    void smart(Optimizer opt, int retries) {
        System.out.println("Testing smart retry " + opt.getClass().getName().split("\\$")[1] + " " + 
                this.getClass().getName().split("\\.")[2] + " stopVal = " + Utils.r(stopVal()));
            Utils.startTiming();
            Result res = CoordRetry.optimize(retries, this, opt, null, limitVal(), stopVal(), 1500, true);
            System.out.println("best = " + res.y + ", time = " + Utils.measuredSeconds() + 
                    " sec, evals = " + res.evals + " x = " + Arrays.toString(res.X));
    }
    
    static void checkGoodResult() {
        double[] x = new double[]{-0.5525744962819084, -0.43883468127578895, -0.6785398184930235, -0.08114253160789063, 
                -0.8411258059974988, -0.6056851931956526, -0.5663895555289944, 0.26568007012429795, -0.058986786386966486, 
                -0.6311239695091586, -1.0, 0.9997028893794696, 0.9994590383759787, 0.9279325579572855, 0.981547346917406, 
                0.9999999999995614, 0.9961605447057663, 0.9999999984827094, 0.9349754944331856, 0.676027417863097, 
                0.9998915752821322, 1.0};
        Interferometry inter = new Interferometry("img/orion.jpg", 11, 512);
        double val = inter.eval(x);
        System.out.println("fval = " + val + " x = " + Arrays.toString(x));
    }

    static void optimize(String img, int n_points, int image_size) throws FileNotFoundException {
        Log.setLog();
        ConcurrencyUtils.setNumberOfThreads(1);
        ConcurrencyUtils.setConcurrentThreshold(1000000);
        Utils.startTiming();
        Interferometry inter = new Interferometry(img, n_points, image_size);
        
        // Differential Evolution parallel function evaluation
        inter.de_parallel(100000);

        // CMA-ES parallel function evaluation
//        inter.cma_parallel(100000);
        
        // parallel retry Differential Evolution
//        inter.retry(new DE(), 128);
        
        // parallel retry DE->CMA sequence
//        inter.retry(new DECMA(), 128);
        
        // smart retry Differential Evolution
//        inter.smart(new DE(), 10000);
        
        // smart retry DE->CMA sequence 
//        inter.smart(new DECMA(), 10000);
    }    
     
    public static void main(String[] args) throws FileNotFoundException {
        optimize("img/orion.jpg", 11, 512);
//        checkGoodResult();
    }

}
