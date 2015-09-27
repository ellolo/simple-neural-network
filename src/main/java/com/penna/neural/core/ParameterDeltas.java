package com.penna.neural.core;

import org.jblas.DoubleMatrix;

/**
 * This class represents the output of the backpropagation algorithm, i.e. the
 * derivative of the cost function with respect to the system parameters,
 * weights and the biases, for all layers. These derivatives are used to update
 * the parameters at each gradient descent epoch.
 * 
 * @author mpennacchiotti
 * 
 */
public class ParameterDeltas {
    DoubleMatrix[] deltaWeights;
    DoubleMatrix[] deltaBiases;

    ParameterDeltas(DoubleMatrix[] deltaWeights, DoubleMatrix[] deltaBiases) {
        this.deltaWeights = deltaWeights;
        this.deltaBiases = deltaBiases;
    }
}
