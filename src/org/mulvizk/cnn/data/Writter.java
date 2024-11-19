package org.mulvizk.cnn.data;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;

public class Writter {

    public static void writeCSV(double[][] data, String file) {

        try (PrintWriter writer = new PrintWriter(new File(file))) {
            StringBuilder sb = new StringBuilder();

            for (int i = 0; i < data.length; i++) {
                for (int j = 0; j < data[0].length; j++) {
                    sb.append(data[i][j] + " ");
                }
                sb.append('\n');
            }

            writer.write(sb.toString());

        } catch (FileNotFoundException e) {
            System.out.println(e.getMessage());
        }
    }
}
