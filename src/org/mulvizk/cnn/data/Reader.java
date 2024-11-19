package org.mulvizk.cnn.data;

import java.io.*;
import java.util.*;

public class Reader {

    private static final int ROWS = 28;
    private static final int COLS = 28;

    public List<Image> readCSV(String path){
        List<Image> images = new ArrayList<>();

        try(BufferedReader reader = new BufferedReader(new FileReader(path))) {

            String line;
            while ((line = reader.readLine()) != null){
                String[] lineItems = line.split(",");

                double[][] data = new double[ROWS][COLS];
                int label = Integer.parseInt(lineItems[0]);

                int i = 1;

                for (int row = 0; row < ROWS; row++) {
                    for (int col = 0; col < COLS; col++) {
                        data[row][col] = (double) Integer.parseInt(lineItems[i]);
                        i++;
                    }
                }

                images.add(new Image(data, label));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }


        return images;
    }


}
