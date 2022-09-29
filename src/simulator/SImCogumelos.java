package simulator;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Debug.Random;
import weka.core.converters.ConverterUtils.DataSource;

public class SImCogumelos {
	public static J48 classifier;
	public static Instances dataset;
	public DataSource source;
	public double predict;
	public String pred;

	public void SImulaCogumelos() throws Exception {
		source = new DataSource("mushroom.arff");
		dataset = source.getDataSet();
		// definitr o atributo target
		dataset.setClassIndex(dataset.numAttributes() - 1);
		// Generated model
		classifier = new J48();
		classifier.buildClassifier(dataset);
		// Visualize decision tree
		Visualizer v = new Visualizer();
		v.start(classifier);
		// cross validation test
		Evaluation eval = new Evaluation(dataset);
		eval.crossValidateModel(classifier, dataset, 10, new Random(1));
		System.out.println(eval.toSummaryString("Results\n ", false));
		System.out.println(eval.toMatrixString());
		System.out.println(classifier.toString());
		// Test a new instance

	}

	
}
