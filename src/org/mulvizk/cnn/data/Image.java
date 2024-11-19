package org.mulvizk.cnn.data;

public class Image {

    private final double[][] data;
    private final int label;

    public Image(double[][] data, int label) {
        this.data = data;
        this.label = label;
    }

    public double[][] getData() {
        return data;
    }

    public int getLabel() {
        return label;
    }
}
