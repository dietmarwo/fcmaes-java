#include "fcmaes_core_Jni.h"

// call java

class CallJava {

public:

   CallJava (
    	const jobject &jfunc,
     	JNIEnv* jenv) : func(jfunc), env(jenv) {
        env->GetJavaVM(&jvm);
    }

    void evalJava(int popsize, int n, const double* arx, double* result) {
        jclass func_class = env->GetObjectClass(func);
        jmethodID mid = env->GetMethodID(func_class, "values", "([D)[D");	  
        int size = popsize * n;
        jdoubleArray jarg = env->NewDoubleArray(size);

        double* args = env->GetDoubleArrayElements(jarg, JNI_FALSE);
        for (int i = 0; i < size; i++)
            args[i] = arx[i];	
        env->SetDoubleArrayRegion (jarg, 0, size, (jdouble*)args);

        // call java cost function
        jobject jores = env->CallObjectMethod(func, mid, jarg);
        jdoubleArray jres = reinterpret_cast<jdoubleArray>(jores);
        double* res = env->GetDoubleArrayElements(jres, JNI_FALSE);
        for (int r = 0; r < popsize; r++)		
            result[r] = res[r];
        env->ReleaseDoubleArrayElements(jarg, args, 0);
        env->ReleaseDoubleArrayElements(jres, res, 0);
    }

    double evalJava1(int n, const double* arx) {
        jclass func_class = env->GetObjectClass(func);
        jmethodID mid = env->GetMethodID(func_class, "value", "([D)D");
        jdoubleArray jarg = env->NewDoubleArray(n);
        double* args = env->GetDoubleArrayElements(jarg, JNI_FALSE);
        for (int i = 0; i < n; i++)
            args[i] = arx[i];
        env->SetDoubleArrayRegion (jarg, 0, n, (jdouble*)args);
        // call java cost function
        jdouble res = env->CallDoubleMethod(func, mid, jarg);
        env->ReleaseDoubleArrayElements(jarg, args, 0);
        return res;
   }
   
   void evalJavaMo(int dim, int nobj, const double* arx, double* result) {
        jclass func_class = env->GetObjectClass(func);
        jmethodID mid = env->GetMethodID(func_class, "movalue", "([D)[D");
        jdoubleArray jarg = env->NewDoubleArray(dim);
        double* args = env->GetDoubleArrayElements(jarg, JNI_FALSE);
        for (int i = 0; i < dim; i++)
            args[i] = arx[i];
        env->SetDoubleArrayRegion (jarg, 0, dim, (jdouble*)args);
        // call java cost function
        jobject jores = env->CallObjectMethod(func, mid, jarg);
        jdoubleArray jres = reinterpret_cast<jdoubleArray>(jores);
        double* res = env->GetDoubleArrayElements(jres, JNI_FALSE);
        for (int r = 0; r < nobj; r++)		
            result[r] = res[r];
        env->ReleaseDoubleArrayElements(jarg, args, 0);
        env->ReleaseDoubleArrayElements(jres, res, 0);
   }

   void logJava(int cols, int xsize, int ysize,
		   const double* xdata, const double* ydata) {
        jclass func_class = env->GetObjectClass(func);
        jmethodID mid = env->GetMethodID(func_class, "log", "(I[D[D)V");
        jdoubleArray jx = env->NewDoubleArray(xsize);
        double* xe = env->GetDoubleArrayElements(jx, JNI_FALSE);
        for (int i = 0; i < xsize; i++)
        	xe[i] = xdata[i];
		env->SetDoubleArrayRegion(jx, 0, xsize, (jdouble*) xe);
        jdoubleArray jy = env->NewDoubleArray(ysize);
        double* ye = env->GetDoubleArrayElements(jy, JNI_FALSE);
        for (int i = 0; i < ysize; i++)
        	ye[i] = ydata[i];
		env->SetDoubleArrayRegion(jy, 0, ysize, (jdouble*) ye);
        jobject jores = env->CallObjectMethod(func, mid, cols, jx, jy);
        env->ReleaseDoubleArrayElements(jx, xe, 0);
        env->ReleaseDoubleArrayElements(jy, ye, 0);
   }

   void printJava(std::string s) {
    	int n = s.size();
		jclass func_class = env->GetObjectClass(func);
		jmethodID mid = env->GetMethodID(func_class, "print", "([B)V");
		jbyteArray jarg = env->NewByteArray(n);
		jbyte *args = env->GetByteArrayElements(jarg, JNI_FALSE);
		for (int i = 0; i < n; i++)
			args[i] = s[i];
		env->SetByteArrayRegion(jarg, 0, n, (jbyte*) args);
		env->CallVoidMethod(func, mid, jarg);
		env->ReleaseByteArrayElements(jarg, args, 0);
	}

    void attachCurrentThread() {
        jvm->AttachCurrentThread((void **) &env, NULL);
    }

    void detachCurrentThread() {
        jvm->DetachCurrentThread();
    }

private:
   jobject func;
   JNIEnv* env;
   JavaVM* jvm;
};


