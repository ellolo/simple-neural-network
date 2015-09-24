package com.penna.neural.functions;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import com.penna.neural.common.DoubleMatrixUtils;

public enum ActivationFunctions {
    SIGMOID {
        @Override
        public DoubleMatrix activate(DoubleMatrix zetas) {
            DoubleMatrix oneMatrix = DoubleMatrix.ones(zetas.rows);
            DoubleMatrix oneMatrix1 = DoubleMatrix.ones(zetas.rows);
            // a = 1 / (1 + e^-z)
            DoubleMatrix sigmoid = oneMatrix.div(oneMatrix1.add(MatrixFunctions.exp(zetas.neg())));
            return sigmoid;
        }

        // a @ (1-a) where a is the vector of activation for the current level
        @Override
        public DoubleMatrix derivative(DoubleMatrix activations) {
            DoubleMatrix ret = activations.mul(DoubleMatrix.ones(activations.rows, 1).sub(activations));
            //System.out.println(" activ * (1 - activ) : \n" + DoubleMatrixUtils.toString(ret));
            return ret;
        }
    },
    TANH {
        @Override
        public DoubleMatrix activate(DoubleMatrix zetas) {
            return MatrixFunctions.tanh(zetas);
        }

        // 1-a^2 case value : Boolean => Json.toJson(if (value) "1" else "0")
        // R model uses 1 and 0 factor levels
        @Override
        public DoubleMatrix derivative(DoubleMatrix activations) {
            return DoubleMatrix.ones(activations.rows, 1).sub(activations.mmul(activations));
        }
    };
    public abstract DoubleMatrix activate(DoubleMatrix zetas);

    public abstract DoubleMatrix derivative(DoubleMatrix activations);
}
