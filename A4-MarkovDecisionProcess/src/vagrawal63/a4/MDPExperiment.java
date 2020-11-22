package vagrawal63.a4;

import java.util.List;

import burlap.behavior.policy.Policy;
import burlap.behavior.policy.PolicyUtils;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.auxiliary.performance.LearningAlgorithmExperimenter;
import burlap.behavior.singleagent.auxiliary.performance.PerformanceMetric;
import burlap.behavior.singleagent.auxiliary.performance.TrialMode;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.ValueFunctionVisualizerGUI;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.LearningAgentFactory;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.planning.Planner;
import burlap.behavior.singleagent.planning.stochastic.policyiteration.PolicyIteration;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.behavior.valuefunction.ValueFunction;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldRewardFunction;
import burlap.domain.singleagent.gridworld.state.GridAgent;
import burlap.domain.singleagent.gridworld.state.GridLocation;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.mdp.auxiliary.common.ConstantStateGenerator;
import burlap.mdp.auxiliary.common.SinglePFTF;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.oo.propositional.PropositionalFunction;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.SADomain;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.statehashing.HashableStateFactory;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import vagrawal63.a4.FileOperationsHelper;

public class MDPExperiment {

	private static int PROBLEM = 1;
	private final static Algorithm ALGO = Algorithm.QLearning;
	private final static boolean VIZ = true;

	/*
	 * This is a very cool feature of BURLAP that unfortunately only works with learning algorithms
	 * (so no ValueIteration or PolicyIteration). It is somewhat redundant to the specific analysis
	 * I implemented for all three algorithms (it computes some of the same stuff), but it shows
	 * very cool charts and it lets you export all the data to external files.
	 * 
	 * At the end, I didnt't use this much, but I'm sure some people will love it. Keep in mind that
	 * by setting this constant to true, you'll be running the QLearning experiment twice (so double
	 * the time).
	 */
	private static boolean USE_LEARNING_EXPERIMENTER = true;

	public static void main(String[] args) {

		final ProblemObject problem = PROBLEM == 1
			? Problems.problem1()
			: Problems.problem2();

		GridWorldDomain gridWorldDomain = new GridWorldDomain(problem.getWidth(), problem.getWidth());
		gridWorldDomain.setMap(problem.getMatrix());
		gridWorldDomain.setProbSucceedTransitionDynamics(0.8);
		//final double[] discountRate = {0.9, 0.95, 0.99, 0.9999};
		final double[] discountRate = {0.9};
		final String fileName = PROBLEM == 1? "P1.csv" : "P2.csv";
		final String dir = "mdp";
		FileOperationsHelper.writeToFile(fileName, dir, FileOperationsHelper.getFileHeader(), false);
		
		/*
		 * This makes sure that the algorithm finishes as soon as the agent reaches the goal. We
		 * don't want the agent to run forever, so this is kind of important.
		 * 
		 * You could set more than one goal if you wish, or you could even set hazards that end the
		 * game (and penalize the agent with a negative reward). But this is not this code...
		 */
		TerminalFunction terminalFunction = new SinglePFTF(PropositionalFunction.findPF(gridWorldDomain.generatePfs(), GridWorldDomain.PF_AT_LOCATION));

		GridWorldRewardFunction rewardFunction = new GridWorldRewardFunction(problem.getWidth(), problem.getWidth(), problem.getDefaultReward());

		/*
		 * This sets the reward for the cell representing the goal. Of course, we want this reward
		 * to be positive and juicy (unless we don't want our agent to reach the end, which will
		 * probably be mean).
		 */
		rewardFunction.setReward(problem.getGoal().x, problem.getGoal().y, problem.getGoalReward());

		/*
		 * This sets up all the rewards associated with the different hazards specified on the
		 * surface of the grid.
		 */
		for (Hazard hazard : problem.getHazards()) {
			rewardFunction.setReward(hazard.getLocation().x, hazard.getLocation().y, hazard.getReward());
		}

		gridWorldDomain.setTf(terminalFunction);
		gridWorldDomain.setRf(rewardFunction);

		OOSADomain domain = gridWorldDomain.generateDomain();

		/*
		 * This sets up the initial position of the agent, and the goal.
		 */
		GridWorldState initialState = new GridWorldState(new GridAgent(problem.getStart().x, problem.getStart().y), new GridLocation(problem.getGoal().x, problem.getGoal().y, "loc0"));

		SimpleHashableStateFactory hashingFactory = new SimpleHashableStateFactory();

		
		for (int i = 0; i< discountRate.length; i ++) {
			Analysis analysis = new Analysis();
			final double dRate = discountRate[i];
			//final double dRate = 0.99;
			switch (ALGO) {
				case ValueIteration:
					runAlgorithm(analysis, problem, domain, hashingFactory, initialState, new PlannerFactory() {

						@Override
						public Planner createPlanner(int episodeIndex, SADomain domain, HashableStateFactory hashingFactory, SimulatedEnvironment simulatedEnvironment) {
							return new ValueIteration(domain, dRate, hashingFactory, 0.001, episodeIndex);
						}
					}, ALGO);
					break;
				case PolicyIteration:
					runAlgorithm(analysis, problem, domain, hashingFactory, initialState, new PlannerFactory() {

						@Override
						public Planner createPlanner(int episodeIndex, SADomain domain, HashableStateFactory hashingFactory, SimulatedEnvironment simulatedEnvironment) {
							return new PolicyIteration(domain, dRate, hashingFactory, 0.001, problem.getNumberOfIterations(Algorithm.ValueIteration), episodeIndex);
						}
					}, ALGO);
					break;
				default:
					runAlgorithm(analysis, problem, domain, hashingFactory, initialState, new PlannerFactory() {

						@Override
						public Planner createPlanner(int episodeIndex, SADomain domain, HashableStateFactory hashingFactory, SimulatedEnvironment simulatedEnvironment) {
							QLearning agent = new QLearning(domain,dRate, hashingFactory, 0.3, 0.1);
							for (int i = 0; i < episodeIndex; i++) {
								agent.runLearningEpisode(simulatedEnvironment);
								simulatedEnvironment.resetEnvironment();
							}

							agent.initializeForPlanning(1);

							return agent;
						}
					}, ALGO);
					break;
			}

			analysis.print(discountRate[i], fileName, dir);
			//analysis.print(0.99, fileName, dir);
			
		}
	}

	/**
	 * Here is where the magic happens. In this method is where I loop through the specific number
	 * of episodes (iterations) and run the specific algorithm. To keep things nice and clean, I use
	 * this method to run all three algorithms. The specific details are specified through the
	 * PlannerFactory interface.
	 * 
	 * This method collects all the information from the algorithm and packs it in an Analysis
	 * instance that later gets dumped on the console.
	 */
	private static void runAlgorithm(Analysis analysis, ProblemObject problem, SADomain domain, HashableStateFactory hashingFactory, State initialState, PlannerFactory plannerFactory, Algorithm algorithm) {
		ConstantStateGenerator constantStateGenerator = new ConstantStateGenerator(initialState);
		SimulatedEnvironment simulatedEnvironment = new SimulatedEnvironment(domain, constantStateGenerator);
		Planner planner = null;
		Policy policy = null;
		for (int episodeIndex = 1; episodeIndex <= problem.getNumberOfIterations(algorithm); episodeIndex++) {
			long startTime = System.nanoTime();
			planner = plannerFactory.createPlanner(episodeIndex, domain, hashingFactory, simulatedEnvironment);
			policy = planner.planFromState(initialState);

			/*
			 * If we haven't converged, following the policy will lead the agent wandering around
			 * and it might never reach the goal. To avoid this, we need to set the maximum number
			 * of steps to take before terminating the policy rollout. I decided to set this maximum
			 * at the number of grid locations in our map (width * width). This should give the
			 * agent plenty of room to wander around.
			 * 
			 * The smaller this number is, the faster the algorithm will run.
			 */
			int maxNumberOfSteps = problem.getWidth() * problem.getWidth();

			Episode episode = PolicyUtils.rollout(policy, initialState, domain.getModel(), maxNumberOfSteps);
			analysis.add(episodeIndex, episode.rewardSequence, episode.numTimeSteps(), (long) (System.nanoTime() - startTime) / 1000000);
		}

		if (algorithm == Algorithm.QLearning && USE_LEARNING_EXPERIMENTER) {
			learningExperimenter(problem, (LearningAgent) planner, simulatedEnvironment);
		}

		if (VIZ && planner != null && policy != null) {
			visualize(problem, (ValueFunction) planner, policy, initialState, domain, hashingFactory, algorithm.getTitle());
		}
	}

	/**
	 * Runs a learning experiment and shows some cool charts. Apparently, this is only useful for
	 * Q-Learning, so I only call this method when Q-Learning is selected and the appropriate flag
	 * is enabled.
	 */
	private static void learningExperimenter(ProblemObject problem, final LearningAgent agent, SimulatedEnvironment simulatedEnvironment) {
		LearningAlgorithmExperimenter experimenter = new LearningAlgorithmExperimenter(simulatedEnvironment, 10, problem.getNumberOfIterations(Algorithm.QLearning), new LearningAgentFactory() {

			public String getAgentName() {
				return Algorithm.QLearning.getTitle();
			}

			public LearningAgent generateAgent() {
				return agent;
			}
		});

		/*
		 * Try different PerformanceMetric values below to display different charts.
		 */
		experimenter.setUpPlottingConfiguration(500, 250, 2, 1000, TrialMode.MOST_RECENT_AND_AVERAGE, PerformanceMetric.CUMULATIVE_STEPS_PER_EPISODE, PerformanceMetric.AVERAGE_EPISODE_REWARD);
		experimenter.startExperiment();
	}

	/**
	 * This method takes care of visualizing the grid, rewards, and specific policy on a nice
	 * BURLAP-predefined GUI. I found this very useful to understand how the algorithm was working.
	 */
	private static void visualize(ProblemObject map, ValueFunction valueFunction, Policy policy, State initialState, SADomain domain, HashableStateFactory hashingFactory, String title) {
		List<State> states = StateReachability.getReachableStates(initialState, domain, hashingFactory);
		ValueFunctionVisualizerGUI gui = GridWorldDomain.getGridWorldValueFunctionVisualization(states, map.getWidth(), map.getWidth(), valueFunction, policy);
		gui.setTitle(title);
		gui.setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);
		gui.initGUI();
	}

}
