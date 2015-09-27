package com.penna.neural.evaluation;

import java.util.List;
import org.jblas.DoubleMatrix;
import com.penna.neural.exceptions.NoLabelException;
import com.penna.neural.utils.DoubleMatrixUtils;

/**
 * This class collects various evaluation metrics for neural network
 * @author mpennacchiotti
 *
 */
public class EvaluationMetrics {

    /**
     * Computes accuracy of predictions with respect to a gold standard true labels. Accuracy is
     * computed as the fraction of predictions whose label with maximum value matches the label
     * with maximum value in the gold set.
     *  
     * @param predictions the prediction labels
     * @param golds the corresponding gold standard true labels
     * @return the accuracy of the predictions
     * @throws NoLabelException if prediction list and gold list do not match in length
     */
    public static double accuracy(List<DoubleMatrix> predictions, List<DoubleMatrix> golds) throws NoLabelException {
        if (predictions.size() < 1 || predictions.size() != golds.size()) {
            throw new NoLabelException("Predictions and gold standard labels are not valid");
        }
        double accuracy = 0;
        for (int i = 0; i < predictions.size(); i++){
            int predictedLabel = predictions.get(i).argmax();
            int trueLabel = golds.get(i).argmax();
            if (predictedLabel == trueLabel) {
                accuracy++;
            }
        }
        return accuracy / predictions.size();
    }
    
    /**
     * Computes average cosine between a prediction and a gold label, across all examples in input.
     *  
     * @param predictions the prediction labels
     * @param golds the corresponding gold standard true labels
     * @return the average cosine between predicted and true labels
     * @throws NoLabelException if prediction list and gold list do not match in length
     */
    public static double averageCosine(List<DoubleMatrix> predictions, List<DoubleMatrix> golds) throws NoLabelException {
        if (predictions.size() < 1 || predictions.size() != golds.size()){
            throw new NoLabelException("Predictions and gold standard labels are not valid");
        }
        double cosine = 0;
        int totPredictions = 0;
        for (int i = 0; i < predictions.size(); i++){
            DoubleMatrix predictedLabel = predictions.get(i);
            DoubleMatrix trueLabel = golds.get(i);
            if (predictedLabel.columns != 1 || predictedLabel.rows !=  trueLabel.rows || predictedLabel.columns !=  trueLabel.columns) {
                throw new NoLabelException("Predictions and gold standard labels are not valid");
            }
            if (predictedLabel.norm2() != 0 ||  trueLabel.norm2() !=0) {
                cosine += predictedLabel.mul(trueLabel).sum();
                System.out.println(DoubleMatrixUtils.toString(predictedLabel));
                System.out.println("\n");
                System.out.println(DoubleMatrixUtils.toString(trueLabel));
                System.out.println("num:" + predictedLabel.mul(trueLabel).sum());
                System.out.println("den:" + predictedLabel.norm2() * trueLabel.norm2());
                cosine = cosine / (predictedLabel.norm2() * trueLabel.norm2());
                
                totPredictions++;
            }
        }
        return cosine / totPredictions;
    }
}
