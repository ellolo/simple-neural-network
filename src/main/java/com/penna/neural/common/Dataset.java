package com.penna.neural.common;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import org.jblas.DoubleMatrix;

public class Dataset{// implements Iterable<Instance> {

    private List<Instance> instances;

    public Dataset() {
        instances = new ArrayList<Instance>();
    }

    public Dataset(List<Instance> instances) {
        this.instances = instances;
    }

    /**
     * Random constructor
     */
    public Dataset(int inputLayerSize, int outputLayerSize, int numInstances) {
        instances = new ArrayList<Instance>();
        for (int i = 0; i < numInstances; i++) {
            DoubleMatrix input = DoubleMatrix.randn(inputLayerSize, 1);
            DoubleMatrix output = DoubleMatrix.randn(outputLayerSize, 1);
            instances.add(new Instance(input, output));
        }
    }

    public void add(Instance instance) {
        instances.add(instance);
    }

    public void shuffle() {
        Collections.shuffle(instances);
    }

    public int size() {
        return instances.size();
    }

    public Iterator<Instance> getInstances() {
        return instances.iterator();
    }

    public Iterator<Instance> iterator() {
        return getInstances();
    }

    public Dataset getSubSet(int startIdx, int endIdx) {
        return new Dataset(instances.subList(startIdx, endIdx));
    }

    public Instance getInstance(int i) {
        return instances.get(i);
    }
}
