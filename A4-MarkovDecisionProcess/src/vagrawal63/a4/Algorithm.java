package vagrawal63.a4;

public enum Algorithm {
	ValueIteration("Value Iteration"),
	PolicyIteration("Policy Iteration"),
	QLearning("Q-Learning");

	private String title;

	private Algorithm(String title) {
		this.title = title;
	}

	public String getTitle() {
		return this.title;
	}
}

