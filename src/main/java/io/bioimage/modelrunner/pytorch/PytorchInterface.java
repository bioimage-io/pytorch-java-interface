/*-
 * #%L
 * This project complements the DL-model runner acting as the engine that works loading models 
 * 	and making inference with Java API for Pytorch.
 * %%
 * Copyright (C) 2022 - 2023 Institut Pasteur and BioImage.IO developers.
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *      http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * #L%
 */
package io.bioimage.modelrunner.pytorch;

import io.bioimage.modelrunner.engine.DeepLearningEngineInterface;
import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.pytorch.tensor.ImgLib2Builder;
import io.bioimage.modelrunner.pytorch.tensor.NDArrayBuilder;
import io.bioimage.modelrunner.system.PlatformDetection;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.tensor.shm.SharedMemoryArray;
import io.bioimage.modelrunner.utils.CommonUtils;
import net.imglib2.type.NativeType;

import java.io.File;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.net.MalformedURLException;
import java.net.URISyntaxException;
import java.net.URL;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.security.ProtectionDomain;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.stream.Collectors;

import com.google.gson.Gson;

import ai.djl.MalformedModelException;
import ai.djl.engine.EngineException;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

/**
 * This class implements an interface that allows the main plugin to interact in
 * an agnostic way with the Deep Learning model and tensors to make inference.
 * This implementation adds Pytorch support to the main program.
 * 
 * Class to that communicates with the dl-model runner, 
 * @see <a href="https://github.com/bioimage-io/model-runner-java">dlmodelrunner</a>
 * to execute Pytorch models.
 * This class implements the interface {@link DeepLearningEngineInterface} to get the 
 * agnostic {@link io.bioimage.modelrunner.tensor.Tensor}, convert them into 
 * {@link ai.djl.ndarray.NDArray}, execute a Pytorch Deep Learning model on them and
 * convert the results back to {@link io.bioimage.modelrunner.tensor.Tensor} to send them 
 * to the main program in an agnostic manner.
 * 
 * {@link ImgLib2Builder}. Creates ImgLib2 images for the backend
 *  of {@link io.bioimage.modelrunner.tensor.Tensor} from {@link ai.djl.ndarray.NDArray}
 * {@link NDArrayBuilder}. Converts {@link io.bioimage.modelrunner.tensor.Tensor} into {@link ai.djl.ndarray.NDArray}
 *  
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public class PytorchInterface implements DeepLearningEngineInterface {

	/**
	 * The Pytorch model loaded with the DJL API
	 */
	private ZooModel<NDList, NDList> model;
	
	private String modelFolder;
	
	private boolean interprocessing = false;
	
	private Process process;
	
	private List<SharedMemoryArray> shmaList;

	/**
	 * Constructor for the interface. It is going to be called from the 
	 * dlmodel-runner
	 */
	public PytorchInterface() {}
	
    /**
     * Private constructor that can only be launched from the class to create a separate
     * process to avoid the conflicts that occur in the same process between Pytorch1 and 2
     * @param doInterprocessing
     * 	whether to do interprocessing or not
     * @throws IOException if the temp dir is not found
     */
    private PytorchInterface(boolean doInterprocessing) throws IOException
    {
    	interprocessing = doInterprocessing;
    }

	/**
	 * {@inheritDoc}
	 * 
	 * Run a Pytorch model on the data provided by the {@link Tensor} input list
	 * and modifies the output list with the results obtained
	 * 
	 */
	@Override
	public void run(List<Tensor<?>> inputTensors, List<Tensor<?>> outputTensors)
		throws RunModelException
	{
		if (interprocessing) {
			runInterprocessing(inputTensors, outputTensors);
			return;
		}
		try (NDManager manager = NDManager.newBaseManager()) {
			// Create the input lists of engine tensors (NDArrays) and their
			// corresponding names
			NDList inputList = new NDList();
			List<String> inputListNames = new ArrayList<String>();
			for (Tensor<?> tt : inputTensors) {
				inputListNames.add(tt.getName());
				inputList.add(NDArrayBuilder.build(tt, manager));
			}
			// Run model
			Predictor<NDList, NDList> predictor = model.newPredictor();
			NDList outputNDArrays = predictor.predict(inputList);
			// Fill the agnostic output tensors list with data from the inference
			// result
			fillOutputTensors(outputNDArrays, outputTensors);
		}
		catch (TranslateException e) {
			e.printStackTrace();
			throw new RunModelException(e.getMessage());
		}
	}

	/**
	 * {@inheritDoc}
	 * 
     * Load a Pytorch model. 
	 */
	@Override
	public void loadModel(String modelFolder, String modelSource)
		throws LoadModelException
	{
		this.modelFolder = modelFolder;
		if (interprocessing) 
			return;
		String modelName = new File(modelSource).getName();
		modelName = modelName.substring(0, modelName.indexOf(".pt"));
		try {
			// Find the URL that corresponds to the file
			URL url = new File(modelFolder).toURI().toURL();
			// Define the location and type of the model
			Criteria<NDList, NDList> criteria = Criteria.builder()
					.setTypes(NDList.class, NDList.class)
					.optModelUrls(url.toString())
					.optModelName(modelName)
					.optProgress(new ProgressBar())
					.build();
			criteria.getTranslatorFactory();
			// Load the model using the criteria defined previously
			this.model = ModelZoo.loadModel(criteria);
		}
		catch (Exception e) {
			e.printStackTrace();
			managePytorchExceptions(e);
			throw new LoadModelException("Error loading a Pytorch model", e.getCause()
				.toString());
		}
	}

	/**
	 * {@inheritDoc}
	 * 
	 * Closes the Pytorch model and sets it to null once the model is not needed anymore.
	 * 
	 */
	@Override
	public void closeModel() {
		if (model != null) 
			model.close();
		model = null;
	}

	/**
	 * Create the list a list of output tensors agnostic to the Deep Learning
	 * engine that can be readable by the model-runner
	 * 
	 * @param outputNDArrays 
	 * 	an NDList containing NDArrays (tensors)
	 * @param outputTensors 
	 * 	the list of output tensors where the output data is going to be written to send back
	 * 	to the model runner
	 * @throws RunModelException If the number of tensors expected is not the same
	 *           as the number of Tensors outputed by the model
	 */
	public static void fillOutputTensors(NDList outputNDArrays,
		List<Tensor<?>> outputTensors) throws RunModelException
	{
		if (outputNDArrays.size() != outputTensors.size())
			throw new RunModelException(outputNDArrays.size(), outputTensors.size());
		for (int i = 0; i < outputNDArrays.size(); i++) {
			outputTensors.get(i).setData(ImgLib2Builder.build(outputNDArrays.get(i)));
		}
	}

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
	
	/**
	 * MEthod  that makes all the arrangements
	 * to create another process, communicate the model info and tensors to the other 
	 * process and then retrieve the results of the other process
	 * @param inputTensors
	 * 	tensors that are going to be run on the model
	 * @param outputTensors
	 * 	expected results of the model
	 * @throws RunModelException if there is any issue running the model
	 */
	public void runInterprocessing(List<Tensor<?>> inputTensors, List<Tensor<?>> outputTensors) throws RunModelException {
		int i = 0;
		shmaList = new ArrayList<SharedMemoryArray>();
		try {
			List<String> args = getProcessCommandsWithoutArgs();
			args.addAll(encodeInputs(inputTensors));
			args.addAll(encodeOutputs(outputTensors));
			ProcessBuilder builder = new ProcessBuilder(args);
			builder.redirectOutput(ProcessBuilder.Redirect.INHERIT);
			builder.redirectError(ProcessBuilder.Redirect.INHERIT);
	        process = builder.start();
	        int result = process.waitFor();
	        process.destroy();
	        if (result != 0)
	    		throw new RunModelException("Error executing the Tensorflow 2 model in"
	        			+ " a separate process. The process was not terminated correctly.");
	        			// TODO remove + System.lineSeparator() + readProcessStringOutput(process));
		} catch (RunModelException e) {
			closeModel();
			throw e;
		} catch (Exception e) {
			closeModel();
			throw new RunModelException(e.toString());
		}
		
		retrieveInterprocessingTensors(outputTensors);
	}
	
	
	private List<String> encodeInputs(List<Tensor<?>> inputTensors) {
		int i = 0;
		List<String> encodedInputTensors = new ArrayList<String>();
		Gson gson = new Gson();
		for (Tensor<?> tt : inputTensors) {
			shmaList.add(SharedMemoryArray.buildSHMA(tt.getData()));
			HashMap<String, Object> map = new HashMap<String, Object>();
			map.put("name", tt.getName());
			map.put("shape", tt.getShape());
			map.put("dtype", CommonUtils.getDataType(tt.getData()));
			map.put("isInput", true);
			map.put("memoryName", shmaList.get(i).getMemoryLocationName());
			encodedInputTensors.add(gson.toJson(map));
	        i ++;
		}
		return encodedInputTensors;
	}
	
	
	private List<String> encodeOutputs(List<Tensor<?>> outputTensors) {
		int i = 0;
		Gson gson = new Gson();
		List<String> encodedOutputTensors = new ArrayList<String>();
		for (Tensor<?> tt : outputTensors) {
			HashMap<String, Object> map = new HashMap<String, Object>();
			map.put("name", tt.getName());
			map.put("shape", tt.getShape());
			map.put("dtype", CommonUtils.getDataType(tt.getData()));
			map.put("isInput", false);
			if (!tt.isEmpty()) {
				shmaList.add(SharedMemoryArray.buildSHMA(tt.getData()));
				map.put("memoryName", shmaList.get(i).getMemoryLocationName());
			}
			encodedOutputTensors.add(gson.toJson(map));
	        i ++;
		}
		return encodedOutputTensors;
	}
	
	/**
	 * Create the arguments needed to execute Pytorch in another 
	 * process with the corresponding tensors
	 * @return the command used to call the separate process
	 * @throws IOException if the command needed to execute interprocessing is too long
	 * @throws URISyntaxException if there is any error with the URIs retrieved from the classes
	 */
	private List<String> getProcessCommandsWithoutArgs() throws IOException, URISyntaxException {
		String javaHome = System.getProperty("java.home");
        String javaBin = javaHome +  File.separator + "bin" + File.separator + "java";

        String modelrunnerPath = getPathFromClass(DeepLearningEngineInterface.class);
        String imglib2Path = getPathFromClass(NativeType.class);
        if (modelrunnerPath == null || (modelrunnerPath.endsWith("DeepLearningEngineInterface.class") 
        		&& !modelrunnerPath.contains(File.pathSeparator)))
        	modelrunnerPath = System.getProperty("java.class.path");
        String classpath =  modelrunnerPath + File.pathSeparator + imglib2Path + File.pathSeparator;
        ProtectionDomain protectionDomain = PytorchInterface.class.getProtectionDomain();
        String codeSource = protectionDomain.getCodeSource().getLocation().getPath();
        String f_name = URLDecoder.decode(codeSource, StandardCharsets.UTF_8.toString());
	        for (File ff : new File(f_name).getParentFile().listFiles()) {
	        	classpath += ff.getAbsolutePath() + File.pathSeparator;
	        }
        String className = PytorchInterface.class.getName();
        List<String> command = new LinkedList<String>();
        command.add(padSpecialJavaBin(javaBin));
        command.add("-cp");
        command.add(classpath);
        command.add(className);
        command.add(modelFolder);
        return command;
	}
	
	/**
	 * Method that gets the path to the JAR from where a specific class is being loaded
	 * @param clazz
	 * 	class of interest
	 * @return the path to the JAR that contains the class
	 * @throws UnsupportedEncodingException if the url of the JAR is not encoded in UTF-8
	 */
	private static String getPathFromClass(Class<?> clazz) throws UnsupportedEncodingException {
	    String classResource = clazz.getName().replace('.', '/') + ".class";
	    URL resourceUrl = clazz.getClassLoader().getResource(classResource);
	    if (resourceUrl == null) {
	        return null;
	    }
	    String urlString = resourceUrl.toString();
	    if (urlString.startsWith("jar:")) {
	        urlString = urlString.substring(4);
	    }
	    if (urlString.startsWith("file:/") && PlatformDetection.isWindows()) {
	        urlString = urlString.substring(6);
	    } else if (urlString.startsWith("file:/") && !PlatformDetection.isWindows()) {
	        urlString = urlString.substring(5);
	    }
	    urlString = URLDecoder.decode(urlString, "UTF-8");
	    File file = new File(urlString);
	    String path = file.getAbsolutePath();
	    if (path.lastIndexOf(".jar!") != -1)
	    	path = path.substring(0, path.lastIndexOf(".jar!")) + ".jar";
	    return path;
	}
	
	/**
	 * if java bin dir contains any special char, surround it by double quotes
	 * @param javaBin
	 * 	java bin dir
	 * @return impored java bin dir if needed
	 */
	private static String padSpecialJavaBin(String javaBin) {
		String[] specialChars = new String[] {" "};
        for (String schar : specialChars) {
        	if (javaBin.contains(schar) && PlatformDetection.isWindows()) {
        		return "\"" + javaBin + "\"";
        	}
        }
        return javaBin;
	}
	
	
	/**
	 * Methods to run interprocessing and be able to run Pytorch 1 and 2
	 * This method checks that the arguments are correct, retrieves the input and output
	 * tensors, loads the model, makes inference with it and finally sends the tensors
	 * to the original process
     * 
     * @param args
     * 	arguments of the program:
     * 		- Path to the model folder
     * 		- Encoded input 0
     * 		- Encoded input 1
     * 		- ...
     * 		- Encoded input n
     * 		- Encoded output 0
     * 		- Encoded output 1
     * 		- ...
     * 		- Encoded output n
     * @throws LoadModelException if there is any error loading the model
     * @throws IOException	if there is any error reading or writing any file or with the paths
     * @throws RunModelException	if there is any error running the model
     */
    public static void main(String[] args) throws LoadModelException, IOException, RunModelException {
    	// Unpack the args needed
    	if (args.length < 3)
    		throw new IllegalArgumentException("Error exectuting Pytorch, "
    				+ "at least35 arguments are required:" + System.lineSeparator()
    				+ " - Path to the model weigths." + System.lineSeparator()
    				+ " - Encoded input 1" + System.lineSeparator()
    				+ " - Encoded input 2 (if exists)" + System.lineSeparator()
    				+ " - ...." + System.lineSeparator()
    				+ " - Encoded input n (if exists)" + System.lineSeparator()
    				+ " - Encoded output 1" + System.lineSeparator()
    				+ " - Encoded output 2 (if exists)" + System.lineSeparator()
    				+ " - ...." + System.lineSeparator()
    				+ " - Encoded output n (if exists)" + System.lineSeparator()
    				);
    	String modelSource = args[0];
    	if (!(new File(modelSource).isFile())) {
    		throw new IllegalArgumentException("Argument 0 of the main method, '" + modelSource + "' "
    				+ "should be the path to the wanted .pth weights file.");
    	}
    	PytorchInterface tfInterface = new PytorchInterface(false);
    	
    	tfInterface.loadModel(new File(modelSource).getParent(), modelSource);
    	
    	HashMap<String, List<String>> map = tfInterface.getInputTensorsFileNames(args);
    	List<String> inputNames = map.get(INPUTS_MAP_KEY);
    	List<Tensor<?>> inputList = inputNames.stream().map(n -> {
									try {
										return tfInterface.retrieveInterprocessingTensorsByName(n);
									} catch (RunModelException e) {
										return null;
									}
								}).collect(Collectors.toList());
    	List<String> outputNames = map.get(OUTPUTS_MAP_KEY);
    	List<Tensor<?>> outputList = outputNames.stream().map(n -> {
									try {
										return tfInterface.retrieveInterprocessingTensorsByName(n);
									} catch (RunModelException e) {
										return null;
									}
								}).collect(Collectors.toList());
    	tfInterface.run(inputList, outputList);
    	tfInterface.createTensorsForInterprocessing(outputList);
	}
}
