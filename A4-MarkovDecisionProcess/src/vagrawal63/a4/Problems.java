package vagrawal63.a4;

import java.util.HashMap;

import vagrawal63.a4.Hazard.HazardType;

public class Problems {

	static ProblemObject problem1() {

		String[] map = new String[] {
				"X0010000",
				"01000S10",
				"010M110S",
				"0M0010M1",
				"011L1000",
				"10L010S0",
				"0L0010L0",
				"00SS00SG",
		};
	
		/*
		 * Make sure to specify the specific number of iterations for each algorithm. If you don't
		 * do this, I'm still nice and use 100 as the default value, but that wouldn't make sense
		 * all the time.
		 */
		HashMap<Algorithm, Integer> numberOfIterations = new HashMap<Algorithm, Integer>();
		numberOfIterations.put(Algorithm.ValueIteration, 50);
		numberOfIterations.put(Algorithm.PolicyIteration, 20);
		numberOfIterations.put(Algorithm.QLearning, 500);
	
		/*
		 * These are the specific rewards for each one of the hazards. Here you can be creative and
		 * play with different values as you see fit.
		 */
		HashMap<HazardType, Double> hazardRewardsHashMap = new HashMap<HazardType, Double>();
		hazardRewardsHashMap.put(HazardType.S, -1.0);
		hazardRewardsHashMap.put(HazardType.M, -2.0);
		hazardRewardsHashMap.put(HazardType.L, -3.0);
	
		/*
		 * Notice how I specify below the specific default reward for cells with nothing on them (we
		 * want regular cells to have a small penalty that encourages our agent to find the goal),
		 * and the reward for the cell representing the goal (something nice and large so the agent
		 * is happy).
		 */
		return new ProblemObject(map, numberOfIterations, -0.1, 20, hazardRewardsHashMap);
	}

	static ProblemObject problem2() {
		String[] map = new String[] {
				"X0001100L000M110",
				"1000100010001000",
				"101100101L1010S1",
				"1000001010001000",
				"111010101010L1S1",
				"1000001000001000",
				"0010101S10101011",
				"0000101010100010",
				"0000101000100010",
				"100M00L010001000",
				"1100101M110010M1",
				"1000101000101000",
				"101010101M001011",
				"10000MM000100000",
				"1010011000101100",
				"101010ML0010000G",
		};
	
		HashMap<Algorithm, Integer> numIterationsHashMap = new HashMap<Algorithm, Integer>();
		numIterationsHashMap.put(Algorithm.ValueIteration, 100);
		numIterationsHashMap.put(Algorithm.PolicyIteration, 40);
		numIterationsHashMap.put(Algorithm.QLearning, 1000);
		
		HashMap<HazardType, Double> hazardRewardsHashMap = new HashMap<HazardType, Double>();
		hazardRewardsHashMap.put(HazardType.S, -1.0);
		hazardRewardsHashMap.put(HazardType.M, -2.0);
		hazardRewardsHashMap.put(HazardType.L, -3.0);
	
		return new ProblemObject(map, numIterationsHashMap, -0.1, 100, hazardRewardsHashMap);
	}

}
