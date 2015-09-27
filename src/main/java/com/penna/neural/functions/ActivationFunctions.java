package com.penna.neural.functions;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

/**
 * Activation functions that can be used in the neural network.
 * 
 * @author mpennacchiotti
 * 
 */
public enum ActivationFunctions {
    /**
     * Sigmoid activation function
     */
    SIGMOID {
        // Activates the function as: a = 1 / (1 + exp(-z)) where 'z' are the
        // zeta values for the
        // layer.
        @Override
        public DoubleMatrix activate(DoubleMatrix zetas) {
            DoubleMatrix oneMatrix = DoubleMatrix.ones(zetas.rows);
            DoubleMatrix oneMatrix1 = DoubleMatrix.ones(zetas.rows);
            DoubleMatrix sigmoid = oneMatrix.div(oneMatrix1.add(MatrixFunctions.exp(zetas.neg())));
            return sigmoid;
        }

        // Deriviative of the activation function as a @ (1-a).
        @Override
        public DoubleMatrix derivative(DoubleMatrix activations) {
            DoubleMatrix ret = activations.mul(DoubleMatrix.ones(activations.rows, 1).sub(
                    activations));
            return ret;
        }
    },
    /**
     * Tanh activation function
     */
    TANH {
        // Activates the function as: a = tanh(z).
        @Override
        public DoubleMatrix activate(DoubleMatrix zetas) {
            return MatrixFunctions.tanh(zetas);
        }

        // Derivative of the activation function 1 - a^2.
        @Override
        public DoubleMatrix derivative(DoubleMatrix activations) {
            return DoubleMatrix.ones(activations.rows, 1).sub(activations.mul(activations));
        }
    };

    /**
     * Applies the activation function to the zeta of the layer's neurons
     * 
     * @param zetas the zetas of the layer
     * @return the activation values for the layer
     */
    public abstract DoubleMatrix activate(DoubleMatrix zetas);

    /**
     * Computes the derivative of the activation function with respect to a
     * layer's activations
     * 
     * @param activations the layer's activations
     * @return derivative of the activation `for the layer
     */
    public abstract DoubleMatrix derivative(DoubleMatrix activations);
}
