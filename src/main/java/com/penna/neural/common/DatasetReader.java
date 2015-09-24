package com.penna.neural.common;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.logging.Logger;

import org.jblas.DoubleMatrix;

/**
 * The DataSetReader class contains static methods to read datasets for the
 * neural network.
 * 
 * @author mpennacchiotti
 * 
 */
public class DatasetReader {

    private static final Logger LOGGER = Logger.getLogger(DatasetReader.class.getName());

    /**
     * Reads Mnist dataset (as in http://yann.lecun.com/exdb/mnist/) storing the
     * data in a GoldSet. The Mnist dataset contains images of digits (from 0 to
     * 9) represented as matrixes of pixels. Each image comes with its label,
     * i.e. the digit that it represents.
     * 
     * Each instance of the minst dataset is constitued by an image and la
     * Reuses code from pen-ui project (https://code.google.com/p/pen-ui/)
     * 
     * @param labelFile
     *            file containing labels of the dataset instances
     * @param imageFile
     *            file containing features of the dataset instances
     * @return a dataset structure
     * @throws IOException
     */
    public static Dataset readMNISTdata(String labelFile, String imageFile) throws IOException {

        DataInputStream labels = new DataInputStream(new FileInputStream(labelFile));
        DataInputStream images = new DataInputStream(new FileInputStream(imageFile));

        try {
            LOGGER.info("Reading MNIST dataset");
            Dataset dataset = new Dataset();

            // check file consistency
            int magicNumber = labels.readInt();
            if (magicNumber != 2049) {
                System.err.println("Label file has wrong magic number: " + magicNumber
                        + " (should be 2049)");
                System.exit(-1);
            }
            magicNumber = images.readInt();
            if (magicNumber != 2051) {
                System.err.println("Image file has wrong magic number: " + magicNumber
                        + " (should be 2051)");
                System.exit(-1);
            }
            int numLabels = labels.readInt();
            int numImages = images.readInt();
            int numRows = images.readInt();
            int numCols = images.readInt();
            if (numLabels != numImages) {
                System.err.println("Image file and label file do not contain the same number of"
                        + "entries.");
                System.err.println("  Label file contains: " + numLabels);
                System.err.println("  Image file contains: " + numImages);
                System.exit(-1);
            }

            // read dataset
            int numLabelsRead = 0;
            while (labels.available() > 0 && numLabelsRead < numLabels) {
                // label of the image is in [0,9]
                int label = (int) labels.readByte();
                numLabelsRead++;
                double[] image = new double[numCols * numRows];
                int idx = 0;
                int pixel;
                for (int colIdx = 0; colIdx < numCols; colIdx++) {
                    for (int rowIdx = 0; rowIdx < numRows; rowIdx++) {
                        pixel = images.readUnsignedByte();
                        image[idx] = pixel;
                        idx++;
                    }
                }
                // add instance (image plus label) in dataset
                DoubleMatrix labelMatrix = DoubleMatrix.zeros(10, 1);
                labelMatrix.put(label, 1);
                DoubleMatrix dImage = new DoubleMatrix(image);
                /*System.out.println("\n*************");
                System.out.println(label);
                int ct = 0;
                for (int i = 0; i < numRows; i++){
                    for (int j = 0; j < numCols; j++){
                        System.out.print(image[ct] + " ");
                        ct++;
                    }
                    System.out.print("\n");
                }*/
                
                Instance instance = new Instance(dImage, labelMatrix);
                dataset.add(instance);
                if ((numLabelsRead % 1000) == 0) {
                    LOGGER.info(" read " + numLabelsRead + " of " + numLabels + " instances");
                }
            }
            LOGGER.info("Completed: read " + numLabelsRead + " instances");
            LOGGER.severe("Completed: read " + numLabelsRead + " instances");
            return dataset;
        } finally {
            labels.close();
            images.close();
        }
    }

    public static void main(String[] argv) throws IOException {
        String labelFile = "/Users/mpennacchiotti/dev/deep-learning/neural-net/data/mnist/training/train-labels-idx1-ubyte";
        String imageFile = "/Users/mpennacchiotti/dev/deep-learning/neural-net/data/mnist/training/train-images-idx3-ubyte";
        readMNISTdata(labelFile, imageFile);
    }
}
