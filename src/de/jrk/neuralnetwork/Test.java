package de.jrk.neuralnetwork;

import de.jrk.neuralnetwork.training.BackpropagationTrainer;

public class Test {
	public static void main(String[] args) {
		NeuralNetwork nn = new NeuralNetwork(3, 4, 3);
		nn.randomize();
		BackpropagationTrainer bpt = new BackpropagationTrainer(nn);
		Matrix[][] trainingData = { { Matrix.from2DArray(0, 0, 0), Matrix.from2DArray(0, 0, 1) },
				{ Matrix.from2DArray(0, 0, 1), Matrix.from2DArray(0, 1, 0) },
				{ Matrix.from2DArray(0, 1, 0), Matrix.from2DArray(0, 1, 1) },
				{ Matrix.from2DArray(0, 1, 1), Matrix.from2DArray(1, 0, 0) },
				{ Matrix.from2DArray(1, 0, 0), Matrix.from2DArray(1, 0, 1) },
				{ Matrix.from2DArray(1, 0, 1), Matrix.from2DArray(1, 1, 0) },
				{ Matrix.from2DArray(1, 1, 0), Matrix.from2DArray(1, 1, 1) },
				{ Matrix.from2DArray(1, 1, 1), Matrix.from2DArray(0, 0, 0) }, };

		for (int i = 0; i < trainingData.length; i++) {
			if (Math.random() > 0.1) {
				bpt.addTrainingData(trainingData[i]);
			} else {
				bpt.addValidationData(trainingData[i]);
			}
		}
		bpt.setLearningRate(0.1);
		for (int i = 0; i < 1000000; i++) {
			for (int j = 0; j < trainingData.length; j++) {
				bpt.train();
			}
			if (i % 1000 == 0) {
				System.out.println(
						"Iteration " + i + "; Loss: " + bpt.getLoss() + "; V. loss: " + bpt.getValidationLoss());
			}
			if (bpt.getLoss() < 0.001) {
				System.out.println("Last iteration: " + i);
				break;
			}
		}
		System.out.println("Final loss: " + bpt.getLoss() + " Final V. loss: " + bpt.getValidationLoss());
		for (int k = 0; k < trainingData.length; k++) {
			Matrix m = Matrix.transpose(nn.feedfoward(trainingData[k][0]));

			String result = "";
			for (int i = 0; i < m.getRows(); i++) {
				if (i != 0) {
					result += "\n";
				}
				for (int j = 0; j < m.getCols(); j++) {
					if (j != 0) {
						result += ", ";
					}
					result += m.get(i, j);
				}
			}
			System.out.println(result);
		}
		for (int k = 0; k < trainingData.length; k++) {
			Matrix m = Matrix.transpose(nn.feedfoward(trainingData[k][0]));

			String result = "";
			for (int i = 0; i < m.getRows(); i++) {
				if (i != 0) {
					result += "\n";
				}
				for (int j = 0; j < m.getCols(); j++) {
					if (j != 0) {
						result += ", ";
					}
					result += Math.round(m.get(i, j));
				}
			}
			System.out.println(result);
		}
		System.out.println("NN:");
		System.out.println(nn);
	}
}
