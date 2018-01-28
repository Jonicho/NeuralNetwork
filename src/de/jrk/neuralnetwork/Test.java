package de.jrk.neuralnetwork;

public class Test {
	public static void main(String[] args) {
		NeuralNetwork nn = new NeuralNetwork(3, 4, 3);
		nn.randomize();
		NeuralNetworkTrainer nnt = new NeuralNetworkTrainer(nn);
		Matrix[][] trainingData = { { Matrix.from2DArray(0, 0, 0), Matrix.from2DArray(0, 0, 1) },
				{ Matrix.from2DArray(0, 0, 1), Matrix.from2DArray(0, 1, 0) },
				{ Matrix.from2DArray(0, 1, 0), Matrix.from2DArray(0, 1, 1) },
				{ Matrix.from2DArray(0, 1, 1), Matrix.from2DArray(1, 0, 0) },
				{ Matrix.from2DArray(1, 0, 0), Matrix.from2DArray(1, 0, 1) },
				{ Matrix.from2DArray(1, 0, 1), Matrix.from2DArray(1, 1, 0) },
				{ Matrix.from2DArray(1, 1, 0), Matrix.from2DArray(1, 1, 1) },
				{ Matrix.from2DArray(1, 1, 1), Matrix.from2DArray(0, 0, 0) }, };

		double learningRate = 0.5;
		double loss = 0;
		for (int i = 0; i < 1000000; i++) {
			for (int j = 0; j < trainingData.length; j++) {
				loss += nnt.train(trainingData[j][0], trainingData[j][1], learningRate);
			}
			loss /= trainingData.length;
			if (i % 1000 == 0) {
				System.out.println("Iteration " + i + "; Loss: " + loss);
			}
			if (loss < 0.001) {
				System.out.println("Last iteration: " + i);
				break;
			}
		}
		System.out.println("Final loss: " + loss);
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
