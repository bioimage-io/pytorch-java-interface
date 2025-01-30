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

import io.bioimage.modelrunner.apposed.appose.Service;
import io.bioimage.modelrunner.apposed.appose.Types;
import io.bioimage.modelrunner.apposed.appose.Service.Task;
import io.bioimage.modelrunner.apposed.appose.Service.TaskStatus;
import io.bioimage.modelrunner.engine.DeepLearningEngineInterface;
import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.numpy.DecodeNumpy;
import io.bioimage.modelrunner.pytorch.shm.ShmBuilder;
import io.bioimage.modelrunner.pytorch.shm.TensorBuilder;
import io.bioimage.modelrunner.pytorch.tensor.ImgLib2Builder;
import io.bioimage.modelrunner.pytorch.tensor.NDArrayBuilder;
import io.bioimage.modelrunner.system.PlatformDetection;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.tensor.shm.SharedMemoryArray;
import io.bioimage.modelrunner.utils.CommonUtils;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.util.Cast;
import net.imglib2.util.Util;

import java.io.File;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.net.URISyntaxException;
import java.net.URL;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.security.ProtectionDomain;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import com.google.gson.Gson;

import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import ai.djl.util.Platform;

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
	
	private String modelSource;
	
	private boolean interprocessing = true;
    /**
     * Process where the model is being loaded and executed
     */
    Service runner;
	
	private List<SharedMemoryArray> shmaInputList = new ArrayList<SharedMemoryArray>();
	
	private List<SharedMemoryArray> shmaOutputList = new ArrayList<SharedMemoryArray>();
	
	private List<String> shmaNamesList = new ArrayList<String>();

	private static final String NAME_KEY = "name";
	private static final String SHAPE_KEY = "shape";
	private static final String DTYPE_KEY = "dtype";
	private static final String IS_INPUT_KEY = "isInput";
	private static final String MEM_NAME_KEY = "memoryName";
	/**
	 * Name without vesion of the JAR created for this library
	 */
	private static final String JAR_FILE_NAME = "dl-modelrunner-pytorch-";
	
    public PytorchInterface(boolean doInterprocessing) throws IOException, URISyntaxException
    {
		interprocessing = doInterprocessing;
		if (this.interprocessing) {
			runner = getRunner();
			runner.debug((text) -> System.err.println(text));
		}
    }
    
    public PytorchInterface() throws IOException, URISyntaxException
    {
		this(true);
    }
    
    private Service getRunner() throws IOException, URISyntaxException {
		List<String> args = getProcessCommandsWithoutArgs();
		String[] argArr = new String[args.size()];
		args.toArray(argArr);

		Service service = new Service(new File("."), argArr);
		service.setEnvVar("CUDA_HOME", null);
		service.setEnvVar("CUDA_PATH", null);
		service.setEnvVar("LD_LIBRARY_PATH", null);
		service.setEnvVar("DYLD_LIBRARY_PATH", null);
		service.setEnvVar("PATH", null);
		return service;
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
		System.out.println("[DEBUG 1]: " + ai.djl.util.Utils.getenv("PATH"));
		System.out.println("[DEBUG 2]: " + Platform.fromSystem("pytorch"));
		this.modelFolder = modelFolder;
		this.modelSource = modelSource;
		if (interprocessing) {
			try {
				launchModelLoadOnProcess();
			} catch (IOException | InterruptedException e) {
				throw new LoadModelException(Types.stackTrace(e));
			}
			return;
		}
		try {
			String modelName = getModelName(modelSource);
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
			Utils.managePytorchExceptions(e);
			throw new LoadModelException("Error loading a Pytorch model", Types.stackTrace(e));
		}
	}
	
	private void launchModelLoadOnProcess() throws IOException, InterruptedException {
		HashMap<String, Object> args = new HashMap<String, Object>();
		args.put("modelFolder", modelFolder);
		args.put("modelSource", modelSource);
		Task task = runner.task("loadModel", args);
		task.waitFor();
		if (task.status == TaskStatus.CANCELED)
			throw new RuntimeException();
		else if (task.status == TaskStatus.FAILED)
			throw new RuntimeException(task.error);
		else if (task.status == TaskStatus.CRASHED) {
			this.runner.close();
			runner = null;
			throw new RuntimeException(task.error);
		}
	}
	
	/**
	 * Method that returns the file name and handles it in case the termination is not correct.
	 * File names should always end with the extension .pt, thus if hte file has any other
	 * extension, what the method does is to copy it into a new file with the new extension.
	 * If the copiped file already exists, the method simply changes the extension
	 * @param modelSource
	 * 	path to the model weights
	 * @return the model weights file name in the correct format, that is with .pt extension
	 * @throws IOException if there is any error copying the file, if it needs to be copied
	 */
	private static String getModelName(String modelSource) throws IOException {
		String modelName = new File(modelSource).getName();
		int ind = modelName.indexOf(".pt");
		if (ind == -1) {
            Path sourcePath = Paths.get(modelSource);
            Path targetPath = Paths.get(modelSource + ".pt");
            if (!targetPath.toFile().isFile())
            	Files.copy(sourcePath, targetPath, StandardCopyOption.REPLACE_EXISTING);
            return getModelName(modelSource + ".pt");
		}
		return modelName.substring(0, ind);
	}

	/**
	 * {@inheritDoc}
	 * 
	 * Run a Pytorch model on the data provided by the {@link Tensor} input list
	 * and modifies the output list with the results obtained
	 * 
	 */
	@Override
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
	void run(List<Tensor<T>> inputTensors, List<Tensor<R>> outputTensors)
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
			for (Tensor<T> tt : inputTensors) {
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
			throw new RunModelException(Types.stackTrace(e));
		}
	}
	
	protected void runFromShmas(List<String> inputs, List<String> outputs) throws IOException, RunModelException {
		try (NDManager manager = NDManager.newBaseManager()) {
			// Create the input lists of engine tensors (NDArrays) and their
			// corresponding names
			NDList inputList = new NDList();
			for (String ee : inputs) {
				Map<String, Object> decoded = Types.decode(ee);
				SharedMemoryArray shma = SharedMemoryArray.read((String) decoded.get(MEM_NAME_KEY));
				NDArray inT = TensorBuilder.build(shma, manager);
				if (PlatformDetection.isWindows()) shma.close();
				inputList.add(inT);
			}
			// Run model
			Predictor<NDList, NDList> predictor = model.newPredictor();
			NDList outputNDArrays = predictor.predict(inputList);

			int c = 0;
			for (String ee : outputs) {
				Map<String, Object> decoded = Types.decode(ee);
				ShmBuilder.build(outputNDArrays.get(c ++), (String) decoded.get(MEM_NAME_KEY));
			}
		}
		catch (TranslateException e) {
			throw new RunModelException(Types.stackTrace(e));
		}
	}
	
	/**
	 * MEthod only used in MacOS Intel and Windows systems that makes all the arrangements
	 * to create another process, communicate the model info and tensors to the other 
	 * process and then retrieve the results of the other process
	 * @param <T>
	 * 	ImgLib2 data type of the input tensors
	 * @param <R>
	 * 	ImgLib2 data type of the output tensors, it can be the same as the input tensors' data type
	 * @param inputTensors
	 * 	tensors that are going to be run on the model
	 * @param outputTensors
	 * 	expected results of the model
	 * @throws RunModelException if there is any issue running the model
	 */
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
	void runInterprocessing(List<Tensor<T>> inputTensors, List<Tensor<R>> outputTensors) throws RunModelException {
		shmaInputList = new ArrayList<SharedMemoryArray>();
		shmaOutputList = new ArrayList<SharedMemoryArray>();
		List<String> encIns = encodeInputs(inputTensors);
		List<String> encOuts = encodeOutputs(outputTensors);
		LinkedHashMap<String, Object> args = new LinkedHashMap<String, Object>();
		args.put("inputs", encIns);
		args.put("outputs", encOuts);

		try {
			Task task = runner.task("inference", args);
			task.waitFor();
			if (task.status == TaskStatus.CANCELED)
				throw new RuntimeException();
			else if (task.status == TaskStatus.FAILED)
				throw new RuntimeException(task.error);
			else if (task.status == TaskStatus.CRASHED) {
				this.runner.close();
				runner = null;
				throw new RuntimeException(task.error);
			}
			for (int i = 0; i < outputTensors.size(); i ++) {
	        	String name = (String) Types.decode(encOuts.get(i)).get(MEM_NAME_KEY);
	        	SharedMemoryArray shm = shmaOutputList.stream()
	        			.filter(ss -> ss.getName().equals(name)).findFirst().orElse(null);
	        	if (shm == null) {
	        		shm = SharedMemoryArray.read(name);
	        		shmaOutputList.add(shm);
	        	}
	        	RandomAccessibleInterval<?> rai = shm.getSharedRAI();
	        	outputTensors.get(i).setData(Tensor.createCopyOfRaiInWantedDataType(Cast.unchecked(rai), Util.getTypeFromInterval(Cast.unchecked(rai))));
	        }
		} catch (Exception e) {
			closeShmas();
			if (e instanceof RunModelException)
				throw (RunModelException) e;
			throw new RunModelException(Types.stackTrace(e));
		}
		closeShmas();
	}

	/**
	 * Create the list a list of output tensors agnostic to the Deep Learning
	 * engine that can be readable by the model-runner
	 * 
	 * @param <T>
	 * 	ImgLib2 data type of the output tensors
	 * @param outputNDArrays 
	 * 	an NDList containing NDArrays (tensors)
	 * @param outputTensors 
	 * 	the list of output tensors where the output data is going to be written to send back
	 * 	to the model runner
	 * @throws RunModelException If the number of tensors expected is not the same
	 *           as the number of Tensors outputed by the model
	 */
	public static <T extends RealType<T> & NativeType<T>>
	void fillOutputTensors(NDList outputNDArrays,
		List<Tensor<T>> outputTensors) throws RunModelException
	{
		if (outputNDArrays.size() != outputTensors.size())
			throw new RunModelException(outputNDArrays.size(), outputTensors.size());
		for (int i = 0; i < outputNDArrays.size(); i++) {
			outputTensors.get(i).setData(ImgLib2Builder.build(outputNDArrays.get(i)));
		}
	}
	
	private void closeShmas() {
		shmaInputList.forEach(shm -> {
			try { shm.close(); } catch (IOException e1) { e1.printStackTrace();}
		});
		shmaInputList = null;
		shmaOutputList.forEach(shm -> {
			try { shm.close(); } catch (IOException e1) { e1.printStackTrace();}
		});
		shmaOutputList = null;
	}
	
	
	private <T extends RealType<T> & NativeType<T>> List<String> encodeInputs(List<Tensor<T>> inputTensors) {
		List<String> encodedInputTensors = new ArrayList<String>();
		Gson gson = new Gson();
		for (Tensor<?> tt : inputTensors) {
			SharedMemoryArray shma = SharedMemoryArray.createSHMAFromRAI(tt.getData(), false, true);
			shmaInputList.add(shma);
			HashMap<String, Object> map = new HashMap<String, Object>();
			map.put(NAME_KEY, tt.getName());
			map.put(SHAPE_KEY, tt.getShape());
			map.put(DTYPE_KEY, CommonUtils.getDataTypeFromRAI(tt.getData()));
			map.put(IS_INPUT_KEY, true);
			map.put(MEM_NAME_KEY, shma.getName());
			encodedInputTensors.add(gson.toJson(map));
		}
		return encodedInputTensors;
	}
	
	
	private <T extends RealType<T> & NativeType<T>> 
	List<String> encodeOutputs(List<Tensor<T>> outputTensors) {
		Gson gson = new Gson();
		List<String> encodedOutputTensors = new ArrayList<String>();
		for (Tensor<?> tt : outputTensors) {
			HashMap<String, Object> map = new HashMap<String, Object>();
			map.put(NAME_KEY, tt.getName());
			map.put(IS_INPUT_KEY, false);
			if (!tt.isEmpty()) {
				map.put(SHAPE_KEY, tt.getShape());
				map.put(DTYPE_KEY, CommonUtils.getDataTypeFromRAI(tt.getData()));
				SharedMemoryArray shma = SharedMemoryArray.createSHMAFromRAI(tt.getData(), false, true);
				shmaOutputList.add(shma);
				map.put(MEM_NAME_KEY, shma.getName());
			} else if (PlatformDetection.isWindows()){
				SharedMemoryArray shma = SharedMemoryArray.create(0);
				shmaOutputList.add(shma);
				map.put(MEM_NAME_KEY, shma.getName());
			} else {
				String memName = SharedMemoryArray.createShmName();
				map.put(MEM_NAME_KEY, memName);
				shmaNamesList.add(memName);
			}
			encodedOutputTensors.add(gson.toJson(map));
		}
		return encodedOutputTensors;
	}

	/**
	 * {@inheritDoc}
	 * 
	 * Closes the Pytorch model and sets it to null once the model is not needed anymore.
	 * 
	 */
	@Override
	public void closeModel() {
		if (this.interprocessing && runner != null) {
			Task task;
			try {
				task = runner.task("close");
				task.waitFor();
			} catch (IOException | InterruptedException e) {
				throw new RuntimeException(Types.stackTrace(e));
			}
			if (task.status == TaskStatus.CANCELED)
				throw new RuntimeException();
			else if (task.status == TaskStatus.FAILED)
				throw new RuntimeException(task.error);
			else if (task.status == TaskStatus.CRASHED) {
				this.runner.close();
				runner = null;
				throw new RuntimeException(task.error);
			}
			this.runner.close();
			this.runner = null;
			return;
		} else if (this.interprocessing) {
			return;
		}
		if (model != null) 
			model.close();
		model = null;
	}
	
	/**
	 * Create the arguments needed to execute tensorflow 2 in another 
	 * process with the corresponding tensors
	 * @return the command used to call the separate process
	 * @throws IOException if the command needed to execute interprocessing is too long
	 * @throws URISyntaxException if there is any error with the URIs retrieved from the classes
	 */
	private List<String> getProcessCommandsWithoutArgs() throws IOException, URISyntaxException {
		String javaHome = System.getProperty("java.home");
        String javaBin = javaHome +  File.separator + "bin" + File.separator + "java";

        String classpath = getCurrentClasspath();
        ProtectionDomain protectionDomain = PytorchInterface.class.getProtectionDomain();
        String codeSource = protectionDomain.getCodeSource().getLocation().getPath();
        String f_name = URLDecoder.decode(codeSource, StandardCharsets.UTF_8.toString());
        f_name = new File(f_name).getAbsolutePath();
        for (File ff : new File(f_name).getParentFile().listFiles()) {
        	if (ff.getName().startsWith(JAR_FILE_NAME) && !ff.getAbsolutePath().equals(f_name))
        		continue;
        	classpath += ff.getAbsolutePath() + File.pathSeparator;
        }
        String className = JavaWorker.class.getName();
        List<String> command = new LinkedList<String>();
        command.add(padSpecialJavaBin(javaBin));
        command.add("-cp");
        command.add(classpath);
        command.add(className);
        return command;
	}
	
    private static String getCurrentClasspath() throws UnsupportedEncodingException {

        String modelrunnerPath = getPathFromClass(DeepLearningEngineInterface.class);
        String imglib2Path = getPathFromClass(NativeType.class);
        String gsonPath = getPathFromClass(Gson.class);
        String jnaPath = getPathFromClass(com.sun.jna.Library.class);
        String jnaPlatformPath = getPathFromClass(com.sun.jna.platform.FileUtils.class);
        if (modelrunnerPath == null || (modelrunnerPath.endsWith("DeepLearningEngineInterface.class") 
        		&& !modelrunnerPath.contains(File.pathSeparator)))
        	modelrunnerPath = System.getProperty("java.class.path");
        String classpath =  modelrunnerPath + File.pathSeparator + imglib2Path + File.pathSeparator;
        classpath =  classpath + gsonPath + File.pathSeparator;
        classpath =  classpath + jnaPath + File.pathSeparator;
        classpath =  classpath + jnaPlatformPath + File.pathSeparator;
        return classpath;
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
	public static <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> void
	main(String[] args) throws IOException, URISyntaxException, LoadModelException, RunModelException {
		PytorchInterface pi = new PytorchInterface(false);
		String folder = "/home/carlos/git/deepimagej-plugin/models/DeepBacs Segmentation Boundary Model_29012025_162730";
		String wt = folder + "/44a0b00b-f171-4fa2-9c39-160e610e9496.pt";
		String npy = folder + "/cfdcff5e-c4e4-4baf-826f-b453751a139d_raw_test_tensor_.npy";
		pi.loadModel(folder, wt);
		
		RandomAccessibleInterval<T> in = DecodeNumpy.loadNpy(npy);

		ArrayList<Tensor<T>> ins = new ArrayList<Tensor<T>>();
		ArrayList<Tensor<R>> ous = new ArrayList<Tensor<R>>();
		Tensor<T> inT = Tensor.build("ff", "bcyx", (RandomAccessibleInterval<T>) in);
		Tensor<R> ouT = (Tensor<R>) Tensor.build("gg", "bcyx", ArrayImgs.floats(new long[] {1, 2, 256, 256}));

		ins.add(inT);
		ous.add(ouT);
		
		pi.run(Cast.unchecked(ins), Cast.unchecked(ous));
		
		DecodeNumpy.saveNpy(folder + "/out_yes_pre.npy", ous.get(0).getData());
		
	}
	*/
}
