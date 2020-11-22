package vagrawal63.a4;

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

	public static String getFileData(double discountRate, double avgReward, double avgSteps, double minSteps, double avgTime) {
		String seperator = ",";
		return Double.toString(discountRate) + seperator + Double.toString(avgReward) + seperator + Double.toString(avgSteps)
				+ seperator + Double.toString(minSteps) + seperator + Double.toString(avgTime);

	}

	public static String getFileHeader() {
		return "Discount_Rate,Avg_Reward,Avg_Steps,Min_Steps,Avg_Time";
	}
	
}
