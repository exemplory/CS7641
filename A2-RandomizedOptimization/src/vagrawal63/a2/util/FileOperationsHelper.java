package vagrawal63.a2.util;

//import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class FileOperationsHelper {

	public FileOperationsHelper() {
		// TODO Auto-generated constructor stub
	}

	public static boolean writeToFile(String fileName, String dir, String output, boolean append) {
		try {
			String outDir = dir + "/" + fileName;
			// String outDir = dir + "/" + new SimpleDateFormat("yyyyMMdd-HHmm").format(new
			// Date()) + "/" + fileName;
			// System.out.println("Dir: " + outDir);
			Path path = Paths.get(outDir);
			if (Files.notExists(path)) {
				Files.createDirectories(path.getParent());
			}
			PrintWriter fileWriter = new PrintWriter(new FileWriter(outDir, append));
			fileWriter.println(output);
			fileWriter.close();
		} catch (IOException e) {
			System.out.println("Exception in writing file:");
			e.printStackTrace();
			return false;
		}
		return true;
	}

	public static String getFileData(int epoch, double RHCVal, double RHCTime, double SAVal, double SATime,
			double GAVal, double GATime, double MIMICVal, double MIMICTime) {
		String seperator = ",";
		return Integer.toString(epoch) + seperator + Double.toString(RHCVal) + seperator + Double.toString(RHCTime)
				+ seperator + Double.toString(SAVal) + seperator + Double.toString(SATime) + seperator
				+ Double.toString(GAVal) + seperator + Double.toString(GATime) + seperator + Double.toString(MIMICVal)
				+ seperator + Double.toString(MIMICTime);

	}

	public static String getFileHeader() {
		return "Epoch,RHC_Score,RHC_Time,SA_Score,SA_Time,GA_Score,GA_Time,MIMIC_Score,MIMIC_Time";
	}
	
	public static String getTuningFileHeader() {
		return "Params,Score,Time";
	}
	
	public static String getNNTrainFileHeader() {
		return "Optimization_Algo,Iterations,Train_Accuracy,Train_Time,Test_Time" ;
	}

	public static String getNNTestFileHeader() {
		return "Optimization_Algo,Iterations,Test_Accuracy,Test_Time" ;
	}
	public static String getNNTrainHyperFileHeader() {
		return "Optimization_Algo,Hyper,Iterations,Train_Accuracy,Train_Time,Test_Time" ;
	}

	public static String getNNTestHyperFileHeader() {
		return "Optimization_Algo,Iterations,Hyper,Test_Accuracy,Test_Time" ;
	}
	
}
