package de.jrk.neuralnetwork.examples.namegenderguesser;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

import de.jrk.neuralnetwork.Matrix;
import de.jrk.neuralnetwork.NeuralNetwork;
import de.jrk.neuralnetwork.training.BackpropagationTrainer;

public class Trainer {
	public static void main(String[] args) {
		NeuralNetwork nn = null;
		boolean newNetwork = true; // change this to false if you want to continue training;
									// change it to true to create a new network
		if (newNetwork) {
			nn = new NeuralNetwork(312, 20, 10, 2);
			nn.randomize();
		} else {
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
			}
		}
		BackpropagationTrainer nnt = new BackpropagationTrainer(nn);
		ArrayList<Matrix[]> trainingData = new ArrayList<Matrix[]>();
		try {
			BufferedReader br = new BufferedReader(
					new InputStreamReader(Trainer.class.getResourceAsStream("male.txt")));
			String line;
			while ((line = br.readLine()) != null) {
				trainingData.add(getTrainingData(line, true));
			}
			br.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		try {
			BufferedReader br = new BufferedReader(
					new InputStreamReader(Trainer.class.getResourceAsStream("female.txt")));
			String line;
			while ((line = br.readLine()) != null) {
				trainingData.add(getTrainingData(line, false));
			}
			br.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		Collections.shuffle(trainingData);
		for (int i = 0; i < trainingData.size(); i++) {
			if ((double) i / trainingData.size() > 0.02) {
				nnt.addTrainingData(trainingData.get(i));
			} else {
				nnt.addValidationData(trainingData.get(i));
			}
		}
		System.out.println("Training data: " + nnt.getTrainingData().size());
		System.out.println("Validation data: " + nnt.getValidationData().size());
		nnt.setLearningRate(0.001);
		for (int i = 0; i < 1000; i++) {
			Collections.shuffle(nnt.getTrainingData());
			nnt.train();
			System.out.printf("Iteration: %s; Loss: %s; V. loss: %s\n", i, nnt.getLoss(), nnt.getValidationLoss());
			saveNN(nn);
		}
	}

	public static Matrix[] getTrainingData(String s, boolean male) {
		Matrix[] result = new Matrix[2];
		result[1] = Matrix.from2DArray(male ? 1 : 0, male ? 0 : 1);
		byte[] nC = s.toLowerCase().trim().getBytes();
		if (nC.length > 12) {
			throw new IllegalArgumentException(s + " is too long (" + nC.length + ")");
		}
		double[] nB = new double[312];
		Arrays.fill(nB, 0);

		for (int i = 0; i < nC.length; i++) {
			if (nC[i] < 'a' || nC[i] > 'z') {
				throw new IllegalArgumentException("Unsupported character: " + nC[i]);
			}
			nB[i * 26 + (nC[i] - (byte) 'a')] = 1;
		}

		result[0] = Matrix.from2DArray(nB);
		return result;
	}

	public static void saveNN(NeuralNetwork nn) {
		try {
			System.out.print("Saving network... ");
			BufferedWriter fw = new BufferedWriter(new FileWriter(new File("nn.txt")));
			fw.write(nn.toString());
			fw.flush();
			fw.close();
			System.out.println("done.");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
