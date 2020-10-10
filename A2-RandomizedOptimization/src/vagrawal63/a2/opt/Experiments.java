package vagrawal63.a2.opt;

public class Experiments {

	public Experiments() {
		// TODO Auto-generated constructor stub
	}
	
	public static void main(String[] args) {
		ContinuousPeak cp = new ContinuousPeak();
		//cp.fit(5);
		//cp.tune(5);
		
		FourPeaks fp = new FourPeaks();
		fp.fit(5);
		//fp.tune(5);

		TravellingSalesman ts = new TravellingSalesman();
		//ts.fit(5);
		//ts.tune(5);
		
	}
}
