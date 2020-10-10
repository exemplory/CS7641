package vagrawal63.a2.opt;

import java.util.Arrays;
import java.util.Random;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.SwapNeighbor;
import opt.example.TravelingSalesmanCrossOver;
import opt.example.TravelingSalesmanEvaluationFunction;
import opt.example.TravelingSalesmanRouteEvaluationFunction;
import opt.example.TravelingSalesmanSortEvaluationFunction;
import opt.ga.CrossoverFunction;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.SwapMutation;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;
import vagrawal63.a2.util.FileOperationsHelper;

/**
 * Adopted from ABAGAIL /src/opt/test/TravelingSalesmanTest.java
 * 
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @submittedby Vivek Agrawal (vagrawal63@gatech.edu)
 * @version 1.0
 */

public class TravellingSalesman {

	/** The n value */
	private static final int N = 50;

	public TravellingSalesman() {
		// TODO Auto-generated constructor stub
	}

	public void fit(int numberOfRuns) {
		// adding timing calculations, iterations and multiple runs
		double startTime, totalTime;
		int[] epochs = { 20, 100, 500, 1000, 2500, 5000, 7500, 10000, 25000, 50000 };
		String results;
		String fileName = "TravellingSalesmanResult.csv";
		String dir = "Optimization";
		FileOperationsHelper.writeToFile(fileName, dir, FileOperationsHelper.getFileHeader(), false);
		for (int epoch : epochs) {

			double sumRHC = 0;
			double sumSA = 0;
			double sumGA = 0;
			double sumMIMIC = 0;

			double timeRHC = 0.0;
			double timeSA = 0.0;
			double timeGA = 0.0;
			double timeMIMIC = 0.0;
			Random random = new Random(903471711);
			for (int j = 0; j < numberOfRuns; j++) {
				// create the random points
				double[][] points = new double[N][2];
				for (int i = 0; i < points.length; i++) {
					points[i][0] = random.nextDouble();
					points[i][1] = random.nextDouble();
				}
				// for rhc, sa, and ga we use a permutation based encoding
				TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);
				Distribution odd = new DiscretePermutationDistribution(N);
				NeighborFunction nf = new SwapNeighbor();
				MutationFunction mf = new SwapMutation();
				CrossoverFunction cf = new TravelingSalesmanCrossOver(ef);
				HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
				GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);

				startTime = System.currentTimeMillis();
				RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
				FixedIterationTrainer fit = new FixedIterationTrainer(rhc, epoch);
				fit.train();
				// System.out.println(ef.value(rhc.getOptimal()));
				totalTime = System.currentTimeMillis() - startTime;
				timeRHC += totalTime / 1000;
				sumRHC += ef.value(rhc.getOptimal());

				startTime = System.currentTimeMillis();
				SimulatedAnnealing sa = new SimulatedAnnealing(1E12, .95, hcp);
				fit = new FixedIterationTrainer(sa, epoch);
				fit.train();
				// System.out.println(ef.value(sa.getOptimal()));
				totalTime = System.currentTimeMillis() - startTime;
				timeSA += totalTime / 1000;
				sumSA += ef.value(sa.getOptimal());

				startTime = System.currentTimeMillis();
				StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 150, 20, gap);
				fit = new FixedIterationTrainer(ga, epoch);
				fit.train();
				// System.out.println(ef.value(ga.getOptimal()));
				totalTime = System.currentTimeMillis() - startTime;
				timeGA += totalTime / 1000;
				sumGA += ef.value(ga.getOptimal());

				// for mimic we use a sort encoding
				startTime = System.currentTimeMillis();
				ef = new TravelingSalesmanSortEvaluationFunction(points);
				int[] ranges = new int[N];
				Arrays.fill(ranges, N);
				odd = new DiscreteUniformDistribution(ranges);
				Distribution df = new DiscreteDependencyTree(.1, ranges);
				ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

				MIMIC mimic = new MIMIC(200, 100, pop);
				fit = new FixedIterationTrainer(mimic, epoch);
				fit.train();
				// System.out.println(ef.value(mimic.getOptimal()));
				totalTime = System.currentTimeMillis() - startTime;
				timeMIMIC += totalTime / 1000;
				sumMIMIC += ef.value(mimic.getOptimal());

			}

			double averageRHC = sumRHC / numberOfRuns;
			double averageSA = sumSA / numberOfRuns;
			double averageGA = sumGA / numberOfRuns;
			double averageMIMIC = sumMIMIC / numberOfRuns;

			double averageTimeRHC = timeRHC / numberOfRuns;
			double averageTimeSA = timeSA / numberOfRuns;
			double averageTimeGA = timeGA / numberOfRuns;
			double averageTimeMIMIC = timeMIMIC / numberOfRuns;

			System.out.println("======== ** Travelling Salesman ** ========");
			System.out.println("Epoch        : " + epoch);
			System.out.println("RHC   Results: Average = " + averageRHC + " Time = " + averageTimeRHC);
			System.out.println("SA    Results: Average = " + averageSA + " Time = " + averageTimeSA);
			System.out.println("GA    Results: Average = " + averageGA + " Time = " + averageTimeGA);
			System.out.println("MIMIC Results: Average = " + averageMIMIC + " Time = " + averageTimeMIMIC);

			results = FileOperationsHelper.getFileData(epoch, averageRHC, averageTimeRHC, averageSA, averageTimeSA,
					averageGA, averageTimeGA, averageMIMIC, averageTimeMIMIC);
			FileOperationsHelper.writeToFile(fileName, dir, results, true);
		}
	}

	public void tune(int numberOfRuns) {
		// adding timing calculations, iterations and multiple runs
		double startTime, totalTime;
		//int[] epochs = { 20, 100, 500, 1000, 2500, 5000, 7500, 10000, 25000, 50000 };
		int[] populationSize = {25, 50, 100, 250, 500 };
		int[] toMateSize = { 10, 25, 50, 100, 150 };
		int[] toMutateSize = { 2, 10, 25, 50, 100 };
		String results;
		String fileName = "TravellingSalesmanOptimizationResult.csv";
		String dir = "Optimization";
		FileOperationsHelper.writeToFile(fileName, dir, FileOperationsHelper.getTuningFileHeader(), false);
		for (int i = 0; i < populationSize.length; i++) {
			for (int j = 0; j < toMateSize.length; j++) {
				for (int k = 0; k < toMutateSize.length; k++) {
					if (toMutateSize[k] > toMateSize[j]) {
						continue;
					}
					if (toMateSize[j] > populationSize[i]) {
						continue;
					}
					double sumGA = 0;
					double timeGA = 0.0;
					Random random = new Random(903471711);
					for (int l = 0; l < numberOfRuns; l++) {
						// create the random points
						double[][] points = new double[N][2];
						for (int m = 0; m < points.length; m++) {
							points[m][0] = random.nextDouble();
							points[m][1] = random.nextDouble();
						}
						// for rhc, sa, and ga we use a permutation based encoding
						TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);
						Distribution odd = new DiscretePermutationDistribution(N);
						MutationFunction mf = new SwapMutation();
						CrossoverFunction cf = new TravelingSalesmanCrossOver(ef);
						GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);

						startTime = System.currentTimeMillis();
						StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(populationSize[i], toMateSize[j],
								toMutateSize[k], gap);
						FixedIterationTrainer fit = new FixedIterationTrainer(ga, 5000);
						fit.train();
						// System.out.println(ef.value(ga.getOptimal()));
						totalTime = System.currentTimeMillis() - startTime;
						timeGA += totalTime / 1000;
						sumGA += ef.value(ga.getOptimal());

					}

					double averageGA = sumGA / numberOfRuns;

					double averageTimeGA = timeGA / numberOfRuns;

					System.out.println("======== ** Travelling Salesman Tuning ** ========");
					System.out.println("Population Size: " + populationSize[i]);
					System.out.println("To Mate        : " + toMateSize[j]);
					System.out.println("to Mutate      : " + toMutateSize[k]);
					System.out.println("GA    Results: Average = " + averageGA + " Time = " + averageTimeGA);

					results = "[" + populationSize[i] + " - " + toMateSize[j] + " - " + toMutateSize[k] + "],"
							+ Double.toString(averageGA) + "," + Double.toString(averageTimeGA);
					FileOperationsHelper.writeToFile(fileName, dir, results, true);
				}
			}
		}
	}

	public static void main(String[] args) {
		TravellingSalesman ts = new TravellingSalesman();
		//ts.fit(1);
		ts.tune(1);
	}

}