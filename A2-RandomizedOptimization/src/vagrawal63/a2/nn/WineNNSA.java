package vagrawal63.a2.nn;

/**
 * Adopted from ABAGAIL /src/opt/test/AbaloneTest.java
 * 
 * @author Hannah Lau
 * @submittedby Vivek Agrawal (vagrawal63@gatech.edu)
 * @version 1.0
 */

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.Scanner;

import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;
import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.NeuralNetworkOptimizationProblem;
import shared.DataSet;
import shared.ErrorMeasure;
import shared.Instance;
import shared.SumOfSquaresError;
import vagrawal63.a2.util.FileOperationsHelper;

public class WineNNSA {

	private static Instance[] instances = initializeInstances();
	private static Instance[] wine_train = Arrays.copyOfRange(instances, 0, 4007);
	private static Instance[] wine_test = Arrays.copyOfRange(instances, 4007, 5010);

	private static int inputLayer = 12, hiddenLayer = 12, outputLayer = 1;
	private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();

	private static ErrorMeasure measure = new SumOfSquaresError();

	private static DataSet set = new DataSet(instances);

	private static BackPropagationNetwork networks[] = new BackPropagationNetwork[1];
	private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[1];

	private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[1];
	private static String[] oaNames = { "SA" };
	private static String results = "";

	private static DecimalFormat df = new DecimalFormat("0.000");

	public static void main(String[] args) {
		for (int i = 0; i < oa.length; i++) {
			networks[i] = factory.createClassificationNetwork(new int[] { inputLayer, hiddenLayer, outputLayer });
			nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
		}

		int[] epochs = { 20, 100, 500, 1000, 2500, 5000, 10000};
		// int[] epochs = { 2, 5 };
		String fileName1 = "WineNN_SA_Train.csv";
		String fileName2 = "WineNN_SA_Test.csv";
		String dir = "Optimization";
		FileOperationsHelper.writeToFile(fileName1, dir, FileOperationsHelper.getNNTrainHyperFileHeader(), false);
		FileOperationsHelper.writeToFile(fileName2, dir, FileOperationsHelper.getNNTestHyperFileHeader(), false);

		double[] t = { 1E9, 1E10, 1E11};
		double[] cRate = {0.45, 0.55, 0.65, 0.75, 0.85, 0.95 };

		for (int epoch : epochs) {
			results = "";

			for (int i = 0; i < oa.length; i++) {
				for (int m = 0; m < t.length; m++) {
					for (int k = 0; k < cRate.length; k++) {
						oa[0] = new SimulatedAnnealing(t[m], cRate[k], nnop[i]);
						double startTime = System.currentTimeMillis();
						double endTime;
						double trainingTime;
						double testingTime;
						double correct = 0;
						double incorrect = 0;
						train(oa[i], networks[i], oaNames[i], epoch); // trainer.train();
						endTime = System.currentTimeMillis();
						trainingTime = endTime - startTime;
						trainingTime /= 1000;

						Instance optimalInstance = oa[i].getOptimal();
						networks[i].setWeights(optimalInstance.getData());

						double predicted, actual;
						startTime = System.currentTimeMillis();
						for (int j = 0; j < wine_train.length; j++) {
							networks[i].setInputValues(wine_train[j].getData());
							networks[i].run();
							predicted = Double.parseDouble(wine_train[j].getLabel().toString());
							actual = Double.parseDouble(networks[i].getOutputValues().toString());
							double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

						}
						endTime = System.currentTimeMillis();
						testingTime = endTime - startTime;
						testingTime /= 1000;
						results = "";
						results = oaNames[i] + "," + epoch + "," + "(" +t[m] + " - "+ cRate[k] + ")" + "," + df.format(correct / (correct + incorrect))
								+ "," + df.format(trainingTime) + "," + df.format(testingTime);
						FileOperationsHelper.writeToFile(fileName1, dir, results, true);
						System.out.println("======== ** NN Simulated Annealing ** ========");
						System.out.println("Epoch        : " + epoch);
						System.out.println("Train Result: " + results);

						startTime = System.currentTimeMillis();
						correct = 0;
						incorrect = 0;
						for (int j = 0; j < wine_test.length; j++) {
							networks[i].setInputValues(wine_test[j].getData());
							networks[i].run();
							actual = Double.parseDouble(wine_test[j].getLabel().toString());
							predicted = Double.parseDouble(networks[i].getOutputValues().toString());
							double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
						}
						endTime = System.currentTimeMillis();
						testingTime = endTime - startTime;
						testingTime /= 1000;
						results = "";
						results = oaNames[i] + "," + epoch + "," + "(" +t[m] + " - "+ cRate[k] + ")" + "," + df.format(correct / (correct + incorrect))
								+ "," + df.format(testingTime);
						FileOperationsHelper.writeToFile(fileName2, dir, results, true);
						System.out.println("Test Result: " + results);
					}
				}
			}
		}
	}

	private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName, int iteration) {

		int trainingIterations = iteration;
		for (int i = 0; i < trainingIterations; i++) {
			oa.train();
			double error = 0;
			for (int j = 0; j < instances.length; j++) {
				network.setInputValues(instances[j].getData());
				network.run();

				Instance output = instances[j].getLabel(), example = new Instance(network.getOutputValues());
				example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
				error += measure.value(output, example);
			}
		}
	}

	private static Instance[] initializeInstances() {
		double[][][] attributes = new double[5011][][];
		try {
			BufferedReader br = new BufferedReader(new FileReader(new File("winequality.csv")));
			for (int i = 0; i < attributes.length; i++) {
				Scanner scan = new Scanner(br.readLine());
				scan.useDelimiter(",");

				attributes[i] = new double[2][];
				attributes[i][0] = new double[12]; // 12 attributes
				attributes[i][1] = new double[1];

				for (int j = 0; j < 12; j++)
					attributes[i][0][j] = Double.parseDouble(scan.next());
				attributes[i][1][0] = Double.parseDouble(scan.next());
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		Instance[] instances = new Instance[attributes.length];

		for (int i = 0; i < instances.length; i++) {
			instances[i] = new Instance(attributes[i][0]);
			instances[i].setLabel(new Instance(attributes[i][1][0]));
		}
		return instances;
	}
}
