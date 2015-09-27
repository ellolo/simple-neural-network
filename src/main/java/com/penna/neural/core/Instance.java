package com.penna.neural.core;

import org.jblas.DoubleMatrix;

import com.penna.neural.exceptions.NoLabelException;
import com.penna.neural.utils.DoubleMatrixUtils;

/**
 * An Instance is an input object to the neural network. If the instance has
 * labels, then it can be used to train or test the neural network. For binary
 * classification, labels are a 1x1 matrix. For n-class classification they are
 * typically a nx1 matrix.
 * 
 * @author mpennacchiotti
 * 
 */
public class Instance {

    private DoubleMatrix features;
    private DoubleMatrix labels;

    /**
     * Constructs an instance with features and labels.
     * 
     * @param features the instance features
     * @param labels the instance labels
     */
    public Instance(DoubleMatrix features, DoubleMatrix labels) {
        this.features = features;
        this.labels = labels;
    }

    /**
     * Constructs an instance with only features.
     * 
     * @param features the instance features
     */
    public Instance(DoubleMatrix features) {
        this.features = features;
    }

    public DoubleMatrix getFeatures() {
        return features;
    }

    /**
     * Returns the labels of the instance.
     * 
     * @return the labels of the instance
     * @throws NoLabelException if the instance does not have a label
     */
    public DoubleMatrix getLabels() throws NoLabelException {
        if (!isLabelled()) {
            throw new NoLabelException("labels are not set");
        }
        return labels;
    }

    /**
     * Tests if instance has labels.
     * 
     * @return true if the instance is labelled
     */
    public boolean isLabelled() {
        if (labels == null) {
            return false;
        }
        return true;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("features\n");
        sb.append(DoubleMatrixUtils.toString(features));
        sb.append("\nlabels\n");
        sb.append(DoubleMatrixUtils.toString(labels));
        return sb.toString();
    }
}
