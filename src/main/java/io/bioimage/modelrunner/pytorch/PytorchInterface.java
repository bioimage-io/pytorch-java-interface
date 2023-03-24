/*-
 * #%L
 * This project complements the DL-model runner acting as the engine that works loading models 
 * 	and making inference with Java API for Pytorch.
 * %%
 * Copyright (C) 2022 - 2023 Institut Pasteur and BioImage.IO developers.
 * %%
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the BioImage.io nor the names of its contributors
 *    may be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 * #L%
 */

package io.bioimage.modelrunner.pytorch;

import io.bioimage.modelrunner.engine.DeepLearningEngineInterface;
import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.pytorch.tensor.ImgLib2Builder;
import io.bioimage.modelrunner.pytorch.tensor.NDArrayBuilder;
import io.bioimage.modelrunner.tensor.Tensor;

import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;

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
 * @see ImgLib2Builder tensors from images and sequences.
 * @author Carlos Garcia Lopez de Haro
 */
public class PytorchInterface implements DeepLearningEngineInterface {

	/**
	 * The Pytorch model loaded with the DJL API
	 */
	private ZooModel<NDList, NDList> model;

	/**
	 * Constructor for the interface. It is going to be called from the 
	 * dlmodel-runner
	 */
	public PytorchInterface() {}

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
		String modelName = new File(modelSource).getName();
		modelName = modelName.substring(0, modelName.indexOf(".pt"));
		try {
			// Find the URL that corresponds to the file
			URL url = new File(modelFolder).toURI().toURL();
			// Define the location and type of the model
			Criteria<NDList, NDList> criteria = Criteria.builder().setTypes(
				NDList.class, NDList.class).optModelUrls(url.toString()) // search
																																	// models in
																																	// specified
																																	// path
				.optModelName(modelName).optProgress(new ProgressBar()).build();
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
		if (model != null) model.close();
		model = null;
	}

	/**
	 * Create the list a list of output tensors agnostic to the Deep Learning
	 * engine that can be readable by Deep Icy
	 * 
	 * @param outputNDArrays an NDList containing NDArrays (tensors)
	 * @param outputTensors the names given to the tensors by the model
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
				msg = "DeepIcy could not load the model.\n" +
					"Please install the Visual Studio 2019 redistributables and reboot" +
					"your machine to be able to use Pytorch with DeepIcy.\n" +
					"For more information:\n" +
					" -https://github.com/awslabs/djl/blob/master/docs/development/troubleshooting.md\n" +
					" -https://github.com/awslabs/djl/issues/126\n" +
					"If you already have installed VS2019 redistributables, the error" +
					"might be caused by a missing dependency or an incompatible Pytorch version.\n" +
					"Furthermore, the DJL Pytorch dependencies (pytorch-egine, pytorch-api and pytorch-native-auto).\n" +
					"should be compatible with each other." +
					"Please check the DeepIcy Wiki.";
			}
			else if ((os.contains("linux") || os.contains("unix")) && err.contains(
				"https://github.com/awslabs/djl/blob/master/docs/development/troubleshooting.md"))
			{
				msg = "DeepIcy could not load the model.\n" +
					"Check that there are no repeated dependencies on the jars folder.\n" +
					"The problem might be caused by a missing or repeated dependency or an incompatible Pytorch version.\n" +
					"Furthermore, the DJL Pytorch dependencies (pytorch-egine, pytorch-api and pytorch-native-auto) " +
					"should be compatible with each other.\n" +
					"If the problem persists, please check the DeepIcy Wiki.";
			}
			else {
				msg = "DeepIcy could not load the model.\n" +
					"Either the DJL Pytorch version is incompatible with the Torchscript model's " +
					"Pytorch version or the DJL Pytorch dependencies (pytorch-egine, pytorch-api and pytorch-native-auto) " +
					"are not compatible with each other.\n" +
					"Please check the DeepIcy Wiki.";
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
