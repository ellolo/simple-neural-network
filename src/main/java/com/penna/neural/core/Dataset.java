package com.penna.neural.core;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.logging.Logger;

import org.jblas.DoubleMatrix;

import com.penna.neural.utils.MnistUtils;

public class Dataset implements Iterable<Instance> {
    
    private static final Logger LOGGER = Logger.getLogger(MnistUtils.class.getName());
    
    private List<Instance> instances;

    public Dataset() {
        instances = new ArrayList<Instance>();
    }

    public Dataset(List<Instance> instances) {
        this.instances = instances;
    }

    /**
     * Constructs a dataset with random unlabelled instances.
     * 
     * @param numFeatures number of instances' features
     * @param numInstances number of instances to the dataset
     */
    public Dataset(int numFeatures, int numInstances) {
        instances = new ArrayList<Instance>();
        for (int i = 0; i < numInstances; i++) {
            DoubleMatrix features = DoubleMatrix.randn(numFeatures, 1);
            instances.add(new Instance(features));
        }
    }
    
    /**
     * Constructs a dataset with random labelled instances.
     * 
     * @param numFeatures number of instances' features
     * @param numLabels number of instances'  labels
     * @param numInstances number of instances to the dataset
     */
    public Dataset(int numFeatures, int numLabels, int numInstances) {
        instances = new ArrayList<Instance>();
        for (int i = 0; i < numInstances; i++) {
            DoubleMatrix features = DoubleMatrix.randn(numFeatures, 1);
            DoubleMatrix labels = DoubleMatrix.randn(numLabels, 1);
            instances.add(new Instance(features, labels));
        }
    }

    public Instance getInstance(int i) {
        return instances.get(i);
    }
    
    public void add(Instance instance) {
        instances.add(instance);
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

    /**
     * Shuffle the order of the instances in the dataset.
     */
    public void shuffle() {
        Collections.shuffle(instances);
    }
    
    /**
     * Get a subset of this dataset, by specifying the start and end index from which to subset
     * from.
     * 
     * @param startIdx start index of the subset
     * @param endIdx end index of the subset
     * @return the subset dataset
     */
    public Dataset getSubSet(int startIdx, int endIdx) {
        return new Dataset(instances.subList(startIdx, endIdx));
    }
    
    /**
     * Removed unlabelled instances from the dataset.
     */
    public void removeUnlabelledInstance() {
        int numRemovedInstances = 0;
        for  (int i = 0; i < instances.size(); i++) {
            if (!instances.get(i).isLabelled()) {
                instances.remove(i);
                numRemovedInstances++;
            }
        }
        LOGGER.info("Removed " + numRemovedInstances + " instances");
    }
}
