package vagrawal63.a4;

public class Hazard {
	public enum HazardType {
		S, M, L
	}
	
	private Coordinates loc;
	private double reward;
	private HazardType type;

	public Hazard(int x, int y, double reward, HazardType type) {
		this.loc = new Coordinates(x, y);
		this.reward = reward;
		this.type = type;
	}
	
	public Coordinates getLocation() {
		return this.loc;
	}
	
	public double getReward() {
		return this.reward;
	}
	
	public HazardType getType() {
		return this.type;
	}
}
