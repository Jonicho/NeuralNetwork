package de.jrk.namegenderguesser;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;

import de.jrk.neuralnetwork.Matrix;
import de.jrk.neuralnetwork.NeuralNetwork;

public class Guesser {
	public static void main(String[] args) {
		if (args == null || args.length < 1) {
			System.out.println("error");
			System.exit(0);
		}
		String name = args[0];
		NeuralNetwork nn = null;
		try {
			BufferedReader br = new BufferedReader(new FileReader(new File("nn.txt")));
			String line;
			String string = "";
			while ((line = br.readLine()) != null) {
				string += line;
			}
			br.close();
			nn = NeuralNetwork.fromString(string);
		} catch (IOException e) {
			e.printStackTrace();
			System.out.println("error");
			System.exit(0);
		}

		System.out.println(nn.feedfoward(getData(name)));
	}

	public static Matrix getData(String s) {
		Matrix result;
		byte[] nC = s.toLowerCase().trim().getBytes();
		if (nC.length > 12) {
			System.out.println("error");
			System.exit(0);
		}
		double[] nB = new double[312];
		Arrays.fill(nB, 0);

		for (int i = 0; i < nC.length; i++) {
			if (nC[i] < 'a' || nC[i] > 'z') {
				System.out.println("error");
				System.exit(0);
			}
			nB[i * 26 + (nC[i] - (byte) 'a')] = 1;
		}

		result = Matrix.from2DArray(nB);
		return result;
	}
}
