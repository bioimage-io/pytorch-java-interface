package io.bioimage.modelrunner.pytorch;

import java.io.IOException;
import java.net.MalformedURLException;

import ai.djl.MalformedModelException;
import ai.djl.engine.EngineException;
import ai.djl.repository.zoo.ModelNotFoundException;

public class Utils {


	/**
	 * Print the correct message depending on the exception that happened trying
	 * to load the model
	 * 
	 * @param e the exception that occurred
	 */
	public static void managePytorchExceptions(Exception e) {
		if (e instanceof ModelNotFoundException ||
			e instanceof MalformedURLException)
		{
			System.out.println("No model was found in the folder provided.");
		}
		else if (e instanceof EngineException) {
			String err = e.getMessage();
			String os = System.getProperty("os.name").toLowerCase();
			String msg;
			if (os.contains("win") && err.contains(
				"https://github.com/awslabs/djl/blob/master/docs/development/troubleshooting.md"))
			{
				msg = "JDLL could not load the model.\n" +
					"Please install the Visual Studio 2019 redistributables and reboot" +
					"your machine to be able to use Pytorch with JDLL.\n" +
					"For more information:\n" +
					" -https://github.com/awslabs/djl/blob/master/docs/development/troubleshooting.md\n" +
					" -https://github.com/awslabs/djl/issues/126\n" +
					"If you already have installed VS2019 redistributables, the error" +
					"might be caused by a missing dependency or an incompatible Pytorch version.\n" +
					"Furthermore, the DJL Pytorch dependencies (pytorch-egine, pytorch-api and pytorch-native-auto).\n" +
					"should be compatible with each other." +
					"Please check the JDLL Wiki.";
			}
			else if ((os.contains("linux") || os.contains("unix")) && err.contains(
				"https://github.com/awslabs/djl/blob/master/docs/development/troubleshooting.md"))
			{
				msg = "JDLL could not load the model.\n" +
					"Check that there are no repeated dependencies on the jars folder.\n" +
					"The problem might be caused by a missing or repeated dependency or an incompatible Pytorch version.\n" +
					"Furthermore, the DJL Pytorch dependencies (pytorch-egine, pytorch-api and pytorch-native-auto) " +
					"should be compatible with each other.\n" +
					"If the problem persists, please check the JDLL Wiki.";
			}
			else {
				msg = "JDLL could not load the model.\n" +
					"Either the DJL Pytorch version is incompatible with the Torchscript model's " +
					"Pytorch version or the DJL Pytorch dependencies (pytorch-egine, pytorch-api and pytorch-native-auto) " +
					"are not compatible with each other.\n" +
					"Please check the JDLL Wiki.";
			}
			System.out.println(msg);
		}
		else if (e instanceof MalformedModelException) {
			String msg = "DeepImageJ could not load the model.\n" +
				"The model provided is not a correct Torchscript model.";
			System.out.println(msg);
		}
		else if (e instanceof IOException) {
			System.out.println("An error occurred accessing the model file.");
		}
	}
}
