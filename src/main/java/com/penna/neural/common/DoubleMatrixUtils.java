package com.penna.neural.common;

import org.jblas.DoubleMatrix;

public class DoubleMatrixUtils {

    public static String toString(DoubleMatrix matrix) {
        return matrix.toString("%+10.3f", "[", "]", " ", "]\n[");
    }
}
