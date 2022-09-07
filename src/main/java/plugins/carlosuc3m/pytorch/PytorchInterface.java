package plugins.carlosuc3m.pytorch;

import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;

import org.bioimageanalysis.icy.deeplearning.exceptions.LoadModelException;
import org.bioimageanalysis.icy.deeplearning.exceptions.RunModelException;
import org.bioimageanalysis.icy.deeplearning.tensor.Tensor;
import org.bioimageanalysis.icy.deeplearning.utils.DeepLearningInterface;
import org.bioimageanalysis.icy.pytorch.tensor.ImgLib2Builder;
import org.bioimageanalysis.icy.pytorch.tensor.NDarrayBuilder;

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
 * This class implements an interface that allows the main plugin
 * to interact in an agnostic way with the Deep Learning model and
 * tensors to make inference.
 * This implementation add the Pytorch support to the main program.
 * 
 * @see SequenceBuilder SequenceBuilder: Create sequences from tensors.
 * @see TensorBuilder TensorBuilder: Create tensors from images and sequences.
 * @author Carlos Garcia Lopez de Haro
 */
public class PytorchInterface implements DeepLearningInterface
{
	private ZooModel<NDList, NDList> model;
    
    public PytorchInterface()
    {    
    }

	@Override
	public List<Tensor> run(List<Tensor> inputTensors, List<Tensor> outputTensors) throws RunModelException {
		try (NDManager manager = NDManager.newBaseManager())
        {
			// Create the input lists of engine tensors (NDArrays) and their corresponding names
            NDList inputList = new NDList();
            List<String> inputListNames = new ArrayList<String>();
            for (Tensor tt : inputTensors) {
            	inputListNames.add(tt.getName());
            	inputList.add(NDarrayBuilder.build(tt, manager));
            }
            // Run model
			Predictor<NDList, NDList> predictor = model.newPredictor();
			NDList outputNDArrays = predictor.predict(inputList);
			// Fill the agnostic output tensors list with data from the inference result
			outputTensors = fillOutputTensors(outputNDArrays, outputTensors);
        } catch (TranslateException e) {
        	e.printStackTrace();
			throw new RunModelException(e.getMessage());
		}
		return outputTensors;
	}

	@Override
	public void loadModel(String modelFolder, String modelSource) throws LoadModelException {
		String modelName = new File(modelSource).getName();
		modelName = modelName.substring(0, modelName.indexOf(".pt"));
		try {
			// Find the URL that corresponds to the file
			URL url = new File(modelFolder).toURI().toURL();
			// Define the location and type of the model
			Criteria<NDList, NDList> criteria = Criteria.builder()
			        .setTypes(NDList.class, NDList.class)
			        .optModelUrls(url.toString()) // search models in specified path
			        .optModelName(modelName)
			        .optEngine("PyTorch")
			        .optProgress(new ProgressBar()).build();
			criteria.getTranslatorFactory();
			// Load the model using the criteria defined previously
			this.model = ModelZoo.loadModel(criteria);
		} catch (Exception e) {
			e.printStackTrace();
			managePytorchExceptions(e);
			throw new LoadModelException("Error loading a Pytorch model", e.getMessage());
		}
	}

	@Override
	public void closeModel() {
		if (model != null)
			model.close();
		model = null;		
	}
	
	/**
	 * Create the list a list of output tensors agnostic to the Deep Learning engine
	 * that can be readable by Deep Icy
	 * @param outputTensors
	 * 	an NDList containing NDArrays (tensors)
	 * @param outputTensors2
	 * 	the names given to the tensors by the model
	 * @return a list with Deep Learning framework agnostic tensors
	 * @throws RunModelException If the number of tensors expected is not the same as the number of
	 * 	Tensors outputed by the model
	 */
	public static List<Tensor> fillOutputTensors(NDList outputNDArrays, List<Tensor> outputTensors) throws RunModelException{
		if (outputNDArrays.size() != outputTensors.size())
			throw new RunModelException(outputNDArrays.size(), outputTensors.size());
		for (int i = 0; i < outputNDArrays.size(); i ++) {
			outputTensors.get(i).setData(ImgLib2Builder.build(outputNDArrays.get(i)));
		}
		return outputTensors;
	}
	
	/**
	 * Print the correct message depending on the exception produced when
	 * trying to load the model
	 * 
	 * @param ex
	 * 	the exception that occurred
	 */
	public static void managePytorchExceptions(Exception e) {
		if (e instanceof ModelNotFoundException || e instanceof MalformedURLException) {
			System.out.println("No model was found in the folder provided.");
		} else if (e instanceof EngineException) {
			String err = e.getMessage();
			String os = System.getProperty("os.name").toLowerCase();
			String msg;
			if (os.contains("win") && err.contains("https://github.com/awslabs/djl/blob/master/docs/development/troubleshooting.md")) {
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
			} else if((os.contains("linux") || os.contains("unix")) && err.contains("https://github.com/awslabs/djl/blob/master/docs/development/troubleshooting.md")){
				msg  = "DeepIcy could not load the model.\n" +
					"Check that there are no repeated dependencies on the jars folder.\n" +
					"The problem might be caused by a missing or repeated dependency or an incompatible Pytorch version.\n" +
					"Furthermore, the DJL Pytorch dependencies (pytorch-egine, pytorch-api and pytorch-native-auto) " +
					"should be compatible with each other.\n" +
					"If the problem persists, please check the DeepIcy Wiki.";
			} else {
				msg  = "DeepIcy could not load the model.\n" +
					"Either the DJL Pytorch version is incompatible with the Torchscript model's " +
					"Pytorch version or the DJL Pytorch dependencies (pytorch-egine, pytorch-api and pytorch-native-auto) " + 
					"are not compatible with each other.\n" +
					"Please check the DeepIcy Wiki.";
			}
			System.out.println(msg);
		} else if (e instanceof MalformedModelException) {
			String msg = "DeepImageJ could not load the model.\n" + 
				"The model provided is not a correct Torchscript model.";
			System.out.println(msg);
		} else if (e instanceof IOException) {
			System.out.println("An error occurred accessing the model file.");
		}
	}
	
	public void finalize() {
		System.out.println("Collected Garbage");
	}
}
