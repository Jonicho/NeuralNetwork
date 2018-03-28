package de.jrk.neuralnetwork.training;

import java.util.ArrayList;
import de.jrk.neuralnetwork.NeuralNetwork;

public class EvolutionalTrainer {
	private ArrayList<EvolutionalNeuralNetwork> networks;
	private int keepAmount;
	private double mutationRate;
	private double lastHighscore;

	public EvolutionalTrainer(NeuralNetwork seedNetwork, int networkAmount, int keepAmount, boolean randomize) {
		if (networkAmount < 2) {
			throw new IllegalArgumentException("The amount of networks must not be less than 2!");
		}
		if (keepAmount >= networkAmount) {
			throw new IllegalArgumentException(
					"The amount of keep networks has to be less than the amount of networks!");
		}
		this.keepAmount = keepAmount;
		networks = new ArrayList<EvolutionalNeuralNetwork>(networkAmount);
		for (int i = 0; i < networkAmount; i++) {
			networks.add(new EvolutionalNeuralNetwork(seedNetwork.getCopy()));
		}
		if (randomize) {
			for (EvolutionalNeuralNetwork neuralNetworkWithScore : networks) {
				neuralNetworkWithScore.getNeuralNetwork().randomize(1);
			}
		}
	}

	public void doIteration(NeuralNetworkTester nnt, boolean useMultiThreading) {
		Thread[] threads = new Thread[networks.size()];
		for (int i = 0; i < networks.size(); i++) {
			if (!networks.get(i).tested) {
				int index = i;
				NeuralNetwork nn = networks.get(index).getNeuralNetwork();
				threads[i] = new Thread(() -> {
					networks.get(index).setScore(nnt.test(nn));
				});
				if (useMultiThreading) {
					threads[i].start();
				} else {
					threads[i].run();
				}
			}
		}
		if (useMultiThreading) {
			for (int i = 0; i < threads.length; i++) {
				try {
					if (threads[i] != null) {
						threads[i].join();
					}
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
		}
		sortNetworks();
		lastHighscore = networks.get(0).getScore();
		generateNewNetworks();
	}

	private void generateNewNetworks() {
		for (int i = keepAmount; i < networks.size(); i++) {
			int randIndex = (int) (Math.random() * Math.random() * keepAmount);
			networks.set(i, new EvolutionalNeuralNetwork(networks.get(randIndex).getNeuralNetwork().getCopy()));
			networks.get(i).mutate(mutationRate);
		}
	}

	public void resetTested() {
		for (EvolutionalNeuralNetwork evolutionalNeuralNetwork : networks) {
			evolutionalNeuralNetwork.tested = false;
		}
	}

	private void sortNetworks() {
		networks.sort((n1, n2) -> (int) Math.signum(n2.getScore() - n1.getScore()));
	}

	public double getHighscore() {
		return lastHighscore;
	}

	public void setMutationRate(double mutationRate) {
		this.mutationRate = mutationRate;
	}

	public NeuralNetwork getBestNetwork() {
		sortNetworks();
		return networks.get(0).getNeuralNetwork().getCopy();
	}

	public ArrayList<NeuralNetwork> getNetworks() {
		ArrayList<NeuralNetwork> result = new ArrayList<NeuralNetwork>();
		for (int i = 0; i < networks.size(); i++) {
			result.add(networks.get(i).getNeuralNetwork());
		}
		return result;
	}

	public ArrayList<EvolutionalNeuralNetwork> getEvolutionalNeuralNetworks() {
		return networks;
	}

	public interface NeuralNetworkTester {
		public double test(NeuralNetwork nn);
	}

	public class EvolutionalNeuralNetwork {
		private final NeuralNetwork neuralNetwork;
		private double score;
		private boolean tested = false;

		public EvolutionalNeuralNetwork(NeuralNetwork neuralNetwork) {
			this.neuralNetwork = neuralNetwork;
		}

		public NeuralNetwork getNeuralNetwork() {
			return neuralNetwork;
		}

		public double getScore() {
			return score;
		}

		public void setScore(double score) {
			this.score = score;
			tested = true;
		}

		public void mutate(double mutationRate) {
			for (int w = 0; w < neuralNetwork.getWeights().length; w++) {
				neuralNetwork.getWeights()[w] = neuralNetwork.getWeights()[w]
						.map((x, i, j) -> x + ((Math.random() * mutationRate * 2) - mutationRate) * x);
			}
			for (int b = 0; b < neuralNetwork.getBiases().length; b++) {
				neuralNetwork.getBiases()[b] = neuralNetwork.getBiases()[b]
						.map((x, i, j) -> x + ((Math.random() * mutationRate * 2) - mutationRate) * x);
			}
		}
	}
}
