package org.mulvizk.cnn.data;

public class MatrixUtils {


    public static double[][] add(double[][] a, double[][] b){

        double[][] output = new double[a.length][a[0].length];

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                output[i][j] = a[i][j] + b[i][j];
            }
        }
        return output;
    }


    public static double[] add(double[] a, double[] b){

        double[] output = new double[a.length];

        for (int i = 0; i < a.length; i++) {
            output[i] = a[i] + b[i];
        }
        return output;
    }

    public static double[][] multiply(double[][] a, double b){

        double[][] output = new double[a.length][a[0].length];

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                output[i][j] = a[i][j] * b;
            }
        }
        return output;
    }


    public static double[] multiply(double[] a, double b){

        double[] output = new double[a.length];

        for (int i = 0; i < a.length; i++) {
            output[i] = a[i] * b;
        }
        return output;
    }



}
