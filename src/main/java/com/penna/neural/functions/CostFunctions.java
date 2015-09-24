package com.penna.neural.functions;

import org.jblas.DoubleMatrix;

import com.penna.neural.common.DoubleMatrixUtils;
import com.penna.neural.functions.ActivationFunctions;

public enum CostFunctions {
    QUADRATIC {
        // (a-y) @ (a @ (1-a))
        @Override
        public DoubleMatrix derivative(DoubleMatrix activations, DoubleMatrix output, ActivationFunctions actFunc) {
            DoubleMatrix tmp = activations.sub(output);
            //System.out.println(" activ - output : \n" + DoubleMatrixUtils.toString(tmp));
            DoubleMatrix activ = actFunc.derivative(activations);
            //System.out.println(" activ deriv : \n" + DoubleMatrixUtils.toString(activ));
            DoubleMatrix ret = (activations.sub(output)).mul(activ);
            //System.out.println(" (activ - output)*(activ*(1-activ)) : \n" + DoubleMatrixUtils.toString(ret));
            return ret;
        }
    },
    CROSS_ENTROPY {
        // (a-y)
        @Override
        public DoubleMatrix derivative(DoubleMatrix activations, DoubleMatrix output,
                ActivationFunctions actFunc) {
            return activations.sub(output);
        }
    };
    public abstract DoubleMatrix derivative(DoubleMatrix activations, DoubleMatrix output,
            ActivationFunctions actFunc);
    
    public static void main(String[] arg) {
        
        // TESTING QUADRATIC WITH SIGMOID
        // should yield [-13.125, 9.375]
        DoubleMatrix activ = new DoubleMatrix(new double[][] { { 3.5 }, { -1.5 } });
        DoubleMatrix res = new DoubleMatrix(new double[][] { { 2.0 }, { 1.0 } });
        DoubleMatrix x = CostFunctions.QUADRATIC.derivative(activ, res, ActivationFunctions.SIGMOID);
        System.out.println(DoubleMatrixUtils.toString(x));
    }
}
