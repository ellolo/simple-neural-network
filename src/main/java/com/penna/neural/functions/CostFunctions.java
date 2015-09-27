package com.penna.neural.functions;

import org.jblas.DoubleMatrix;
import com.penna.neural.functions.ActivationFunctions;
import com.penna.neural.utils.DoubleMatrixUtils;

/**
 * Cost functions that can be used in the neural network.
 * 
 * @author mpennacchiotti
 *
 */
public enum CostFunctions {
    /**
     * Quadratic cost function
     */
    QUADRATIC {
        // Computes derivative of the cost function, as: (a-y) @ (a @ (1-a)), where 'a' is the
        //activation, 'y' the labels. 
        @Override
        public DoubleMatrix derivative(DoubleMatrix activations, DoubleMatrix output, ActivationFunctions actFunc) {
            DoubleMatrix tmp = activations.sub(output);
            DoubleMatrix activ = actFunc.derivative(activations);
            DoubleMatrix ret = (activations.sub(output)).mul(activ);
            return ret;
        }
    },
    /**
     * Cross entropy cost function
     */
    CROSS_ENTROPY {
        // Computes derivative of the cost function, as: (a-y), where 'a' is the activation, 'y'
        //the labels. 
        @Override
        public DoubleMatrix derivative(DoubleMatrix activations, DoubleMatrix output,
                ActivationFunctions actFunc) {
            return activations.sub(output);
        }
    };
    
    /**
     * Computes the derivative of the cost function at the output layer with respect to the zeta
     * of the neuron.
     *  
     * @param activations the activations of the output layer
     * @param output the expected labels of the instance
     * @param actFunc the activation function of the neural network 
     * @return the derivative
     */
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
