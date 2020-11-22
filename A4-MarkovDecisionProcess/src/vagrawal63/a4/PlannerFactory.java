package vagrawal63.a4;

import burlap.behavior.singleagent.planning.Planner;
import burlap.mdp.singleagent.SADomain;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.statehashing.HashableStateFactory;

public interface PlannerFactory {
	Planner createPlanner(int episodeIndex, SADomain domain, HashableStateFactory hashingFactory, SimulatedEnvironment simulatedEnvironment);
}
