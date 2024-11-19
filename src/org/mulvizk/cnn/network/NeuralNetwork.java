package org.mulvizk.cnn.network;

import org.mulvizk.cnn.data.Image;
import org.mulvizk.cnn.layers.Layer;

import java.util.ArrayList;
import java.util.List;

import static org.mulvizk.cnn.data.MatrixUtils.add;
import static org.mulvizk.cnn.data.MatrixUtils.multiply;

public class NeuralNetwork {

    List<Layer> layers;
    double scaleFactor;

    public NeuralNetwork(List<Layer> layers, double scaleFactor) {
        this.layers = layers;
        this.scaleFactor = scaleFactor;
        linkLayers();
    }

    private void linkLayers(){
        if(layers.size() <= 1){
            return;
        }else{

            for (int i = 0; i < layers.size(); i++) {
                if(i == 0){
                    layers.get(i).setNextLayer(layers.get(i + 1));

                }else if(i == layers.size() - 1){
                    layers.get(i).setPreviousLayer(layers.get(i - 1));
                }else {
                    layers.get(i).setPreviousLayer(layers.get(i - 1));
                    layers.get(i).setNextLayer(layers.get(i + 1));
                }
            }
        }

    }

    public double[] getErrors(double[] nnOutput, int correctAnswer){

        int numClasses = nnOutput.length;

        double[] expected = new double[numClasses];

        expected[correctAnswer] = 1;

        return add(nnOutput, multiply(expected, -1));

    }

    private int getMaxIndex(double[] index){

        double max = 0;
        int indx = 0;

        for (int i = 0; i < index.length; i++) {
            if (index[i] >= max){
                max = index[i];
                indx = i;
            }
        }
        return indx;
    }

    public int guess(Image image){
        List<double[][]> inList = new ArrayList<>();
        inList.add(multiply(image.getData(), (1 / scaleFactor)));

        double[] out = layers.get(0).getOutput(inList);
        int guess = getMaxIndex(out);

        return guess;
    }

    public float test (List<Image> images){
        int correct = 0;

        for (Image image : images){
            int guess = guess(image);
            if(guess == image.getLabel()){
                correct++;
            }
        }

        return ((float) correct / images.size());
    }

    public void train(List<Image> images){

        for (Image image : images){
            List<double[][]> inList = new ArrayList<>();
            inList.add(multiply(image.getData(), (1d / scaleFactor)));
            double[] out = layers.get(0).getOutput(inList);
            double[] diffLoss_diffOut = getErrors(out, image.getLabel());

            layers.get(layers.size() - 1).backPropagation(diffLoss_diffOut);
        }
    }

    public List<Layer> getLayers() {
        return layers;
    }
}
