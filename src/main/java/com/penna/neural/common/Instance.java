package com.penna.neural.common;

import org.jblas.DoubleMatrix;

public class Instance {

    private DoubleMatrix features;
    private DoubleMatrix labels;

    public Instance(DoubleMatrix in, DoubleMatrix out) {
        features = in;
        labels = out;
    }

    public DoubleMatrix getFeatures() {
        return features;
    }

    public DoubleMatrix getLabels() {
        return labels;
    }
}
