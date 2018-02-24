package de.jrk.xor;

import de.jrk.neuralnetwork.Matrix;
import de.jrk.neuralnetwork.NeuralNetwork;
import de.jrk.neuralnetwork.training.BackpropagationTrainer;
import processing.core.PApplet;

public class Main extends PApplet {
	NeuralNetwork nn;
	BackpropagationTrainer bpt;

	public static void main(String[] args) {
		PApplet.main("de.jrk.xor.Main");
	}

	public void settings() {
		size(300, 300);
	}

	public void setup() {
		nn = new NeuralNetwork(2, 2, 1);
		nn.randomize();
		bpt = new BackpropagationTrainer(nn);
		bpt.setLearningRate(0.1);
		bpt.addTrainingData(new Matrix[] { Matrix.from2DArray(1, 0), Matrix.from2DArray(1) },
				new Matrix[] { Matrix.from2DArray(1, 1), Matrix.from2DArray(0) },
				new Matrix[] { Matrix.from2DArray(0, 0), Matrix.from2DArray(0) },
				new Matrix[] { Matrix.from2DArray(0, 1), Matrix.from2DArray(1) });
		noStroke();
	}

	public void draw() {
		for (int i = 0; i < 100; i++) {
			bpt.train();
		}
		int size = 20;
		for (int x = 0; x < size; x++) {
			for (int y = 0; y < size; y++) {
				fill((int) (nn.feedforward(Matrix.from2DArray(x / (double) size, y / (double) size)).get(0, 0) * 255));
				rect(x * (width / size), y * (height / size), width / size, height / size);
			}
		}
	}
}
