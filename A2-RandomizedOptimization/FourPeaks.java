package vagrawal63.a2.opt;

import java.util.Arrays;

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
import vagrawal63.a2.util.FileOperationsHelper;

/**
 * Copied from ContinuousPeaksTest
 * 
 * @version 1.0
 */
public class FourPeaks {

	/** The n value */
	private static final int N = 200;
	/** The t value */
	private static final int T = N / 5;

	public FourPeaks() {
		// TODO Auto-generated constructor stub
	}

	public void fit(int numberOfRuns) {
		// adding timing calculations, iterations and multiple runs
		double startTime, totalTime;
		int[] epochs = { 20, 100, 500, 1000, 2500, 5000, 7500, 10000, 25000, 50000 };
		String results;
		String fileName = "FourPeaksResult.csv";
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
				EvaluationFunction ef = new FourPeaksEvaluationFunction(T);
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
				// System.out.println("RHC: " + ef.value(rhc.getOptimal()));
				totalTime = System.currentTimeMillis() - startTime;
				timeRHC += totalTime / 1000;
				sumRHC += ef.value(rhc.getOptimal());

				startTime = System.currentTimeMillis();
				SimulatedAnnealing sa = new SimulatedAnnealing(1E11, .95, hcp);
				fit = new FixedIterationTrainer(sa, epoch);
				fit.train();
				// System.out.println("SA: " + ef.value(sa.getOptimal()));
				totalTime = System.currentTimeMillis() - startTime;
				timeSA += totalTime / 1000;
				sumSA += ef.value(sa.getOptimal());

				startTime = System.currentTimeMillis();
				StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 100, 10, gap);
				fit = new FixedIterationTrainer(ga, epoch);
				fit.train();
				// System.out.println("GA: " + ef.value(ga.getOptimal()));
				totalTime = System.currentTimeMillis() - startTime;
				timeGA += totalTime / 1000;
				sumGA += ef.value(ga.getOptimal());

				startTime = System.currentTimeMillis();
				MIMIC mimic = new MIMIC(200, 20, pop);
				fit = new FixedIterationTrainer(mimic, epoch);
				fit.train();
				// System.out.println("MIMIC: " + ef.value(mimic.getOptimal()));
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

			System.out.println("======== ** Four Peaks ** ========");
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
		int[] populationSize = {50, 100, 200, 500, 1000, 2500, 5000, 10000};
		int[] toKeepSize = {10, 50, 100, 250, 500, 750, 1000};
		String results;
		String fileName = "FourPeaksOptimizationResult.csv";
		String dir = "Optimization";
		FileOperationsHelper.writeToFile(fileName, dir, FileOperationsHelper.getTuningFileHeader(), false);
		for (int i = 0; i < populationSize.length; i++) {
			for (int j = 0; j < toKeepSize.length; j++) {
				if (toKeepSize[j] >= populationSize[i]) {
					continue;
				}
				int sumMIMIC = 0;

				double timeMIMIC = 0.0;

				for (int k = 0; k < numberOfRuns; k++) {
					int[] ranges = new int[N];
					Arrays.fill(ranges, 2);
					EvaluationFunction ef = new FourPeaksEvaluationFunction(T);
					Distribution odd = new DiscreteUniformDistribution(ranges);
					Distribution df = new DiscreteDependencyTree(.1, ranges);
					ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

					startTime = System.currentTimeMillis();
					MIMIC mimic = new MIMIC(populationSize[i], toKeepSize[j], pop);
					FixedIterationTrainer fit = new FixedIterationTrainer(mimic, 5000);
					fit.train();
					// System.out.println("MIMIC: " + ef.value(mimic.getOptimal()));
					totalTime = System.currentTimeMillis() - startTime;
					timeMIMIC += totalTime / 1000;
					sumMIMIC += ef.value(mimic.getOptimal());

				}

				double averageMIMIC = sumMIMIC / numberOfRuns;

				double averageTimeMIMIC = timeMIMIC / numberOfRuns;

				System.out.println("======== ** Four Peaks ** ========");
				System.out.println("Population    : " + populationSize[i]);
				System.out.println("To Keep Size : " + toKeepSize[j]);
				System.out.println("MIMIC Results: Average = " + averageMIMIC + " Time = " + averageTimeMIMIC);
				results = "[" + populationSize[i] + " - " + toKeepSize[j] + "]," + Double.toString(averageMIMIC) + ","
						+ Double.toString(averageTimeMIMIC);
				FileOperationsHelper.writeToFile(fileName, dir, results, true);
			}
		}
	}

	public static void main(String[] args) {
		FourPeaks fp = new FourPeaks();
		//fp.fit(1);
		fp.tune(1);
	}

}
