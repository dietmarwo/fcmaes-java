#include "fcmaes_core_Jni.h"

// call java

class CallJava {

public:

   CallJava (
    	const jobject jfunc,
     	JNIEnv* jenv)  
    {
    	func = jfunc;
     	env = jenv;
    }

    void evalJava(int popsize, int n, double* arx, double* result) {
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

    double evalJava1(int n, double* arx) {
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

private:
   jobject func;
   JNIEnv* env;
};


