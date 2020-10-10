package vagrawal63.a2.opt;

import java.util.Arrays;
import vagrawal63.a2.util.FileOperationsHelper;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * Adopted from ABAGAIL /src/opt/test/ContinuousPeaksTest.java
 * 
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @submittedby Vivek Agrawal (vagrawal63@gatech.edu)
 * @version 1.0
 */
public class ContinuousPeak {
	/** The n value */
	private static final int N = 60;
	/** The t value */
	private static final int T = N / 10;

	public ContinuousPeak() {
		super();
		// TODO Auto-generated constructor stub
	}

	public void fit(int numberOfRuns) {
		// adding timing calculations, iterations and multiple runs
		double startTime, totalTime;
		//int[] epochs = { 10, 100, 1000, 2500, 5000, 10000, 25000, 50000, 100000, 200000 };
		int[] epochs = {20, 100, 500, 1000, 2500, 5000, 7500, 10000, 25000, 50000};
		String results;
		String fileName = "ContinuousPeaksResult.csv";
		String dir = "Optimization";
		FileOperationsHelper.writeToFile(fileName, dir, FileOperationsHelper.getFileHeader(), false);
		for (int epoch : epochs) {

			int sumRHC = 0;
			int sumSA = 0;
			int sumGA = 0;
			int sumMIMIC = 0;

			double timeRHC = 0.0;
			double timeSA = 0.0;
			double timeGA = 0.0;
			double timeMIMIC = 0.0;

			for (int j = 0; j < numberOfRuns; j++) {
				int[] ranges = new int[N];
				Arrays.fill(ranges, 2);
				EvaluationFunction ef = new ContinuousPeaksEvaluationFunction(T);
				Distribution odd = new DiscreteUniformDistribution(ranges);
				NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
				MutationFunction mf = new DiscreteChangeOneMutation(ranges);
				CrossoverFunction cf = new SingleCrossOver();
				Distribution df = new DiscreteDependencyTree(.1, ranges);
				HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
				GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
				ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

				startTime = System.currentTimeMillis();
				RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
				FixedIterationTrainer fit = new FixedIterationTrainer(rhc, epoch);
				fit.train();
				// System.out.println(ef.value(rhc.getOptimal()));
				totalTime = System.currentTimeMillis() - startTime;
				timeRHC += totalTime/1000;
				sumRHC += ef.value(rhc.getOptimal());

				startTime = System.currentTimeMillis();
				SimulatedAnnealing sa = new SimulatedAnnealing(1E11, .95, hcp);
				fit = new FixedIterationTrainer(sa, epoch);
				fit.train();
				// System.out.println(ef.value(sa.getOptimal()));
				totalTime = System.currentTimeMillis() - startTime;
				timeSA += totalTime/1000;
				sumSA += ef.value(sa.getOptimal());

				startTime = System.currentTimeMillis();
				StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 100, 20, gap);
				fit = new FixedIterationTrainer(ga, epoch);
				fit.train();
				// System.out.println(ef.value(ga.getOptimal()));
				totalTime = System.currentTimeMillis() - startTime;
				timeGA += totalTime/1000;
				sumGA += ef.value(ga.getOptimal());

				startTime = System.currentTimeMillis();
				MIMIC mimic = new MIMIC(200, 20, pop);
				fit = new FixedIterationTrainer(mimic, epoch);
				fit.train();
				// System.out.println(ef.value(mimic.getOptimal()));
				totalTime = System.currentTimeMillis() - startTime;
				timeMIMIC += totalTime/1000;
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

			System.out.println("======== ** Continuous Peaks ** ========");
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
		double[] t = { 1E9, 1E10, 1E11, 1E12, 1E13 };
		double[] cRate = { 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95 };
		String results;
		String fileName = "ContinuousPeaksOptimizationResult.csv";
		String dir = "Optimization";
		FileOperationsHelper.writeToFile(fileName, dir, FileOperationsHelper.getTuningFileHeader(), false);
		for (int i = 0; i < t.length; i++) {
			for (int j = 0; j < cRate.length; j++) {

				int sumSA = 0;

				double timeSA = 0.0;

				for (int k = 0; k < numberOfRuns; k++) {
					int[] ranges = new int[N];
					Arrays.fill(ranges, 2);
					EvaluationFunction ef = new ContinuousPeaksEvaluationFunction(T);
					Distribution odd = new DiscreteUniformDistribution(ranges);
					NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
					HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);

					startTime = System.currentTimeMillis();
					SimulatedAnnealing sa = new SimulatedAnnealing(t[i], cRate[j], hcp);
					FixedIterationTrainer fit = new FixedIterationTrainer(sa, 3500);
					fit.train();
					// System.out.println(ef.value(sa.getOptimal()));
					totalTime = System.currentTimeMillis() - startTime;
					timeSA += totalTime / 1000;
					sumSA += ef.value(sa.getOptimal());
				}

				double averageSA = sumSA / numberOfRuns;

				double averageTimeSA = timeSA / numberOfRuns;

				System.out.println("======== ** Continuous Peaks Tuning ** ========");
				System.out.println("Temperature  : " + t[i]);
				System.out.println("Cooling Rate : " + cRate[j]);
				System.out.println("SA    Results: Average = " + averageSA + " Time = " + averageTimeSA);

				results = "[" + t[i] + " - " + cRate[j] + "]," + Double.toString(averageSA) + "," + Double.toString(averageTimeSA);
				FileOperationsHelper.writeToFile(fileName, dir, results, true);
			}
		}
	}

	public static void main(String[] args) {
		ContinuousPeak cp = new ContinuousPeak();
		//cp.fit(1);
		cp.tune(5);
	}
}
