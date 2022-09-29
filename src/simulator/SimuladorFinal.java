package simulator;

import net.sourceforge.jFuzzyLogic.FIS;
import net.sourceforge.jFuzzyLogic.FunctionBlock;
import weka.core.Instance;
import weka.core.Instances;

public class SimuladorFinal {
	public static double distanceR;
	public static double distanceC;
	public static double distanceL;
	public static double predict;
	public static String pred;

	private static double getClassification(String test) {
		if (test == null)
			return 2.0;
		if (test.equals("poisonous"))
			return 1.0;
		if (test.equals("edible"))
			return 3.0;

		return 0.0;
	}

	public static void main(String[] args) throws Exception {

		SImCogumelos sImCogumelos = new SImCogumelos();
		sImCogumelos.SImulaCogumelos();
		Simulator s = new Simulator();

		s.setSimulationSpeed(50);
		s.setRobotSpeed(5);
		s.step();
		while (true) {
			distanceR = s.getDistanceR();
			distanceL = s.getDistanceL();
			distanceC = s.getDistanceC();
			int speed = s.getRobotSpeed();

			String[] banana = s.getMushroomAttributes();

			if (banana != null) {
				NewInstances ni = new NewInstances(sImCogumelos.dataset);

				ni.addInstance(banana);

				Instances test_dt = ni.getDataset();
				System.out.println("ActualClass \t PredictedClass");
				for (int i = 0; i < test_dt.numInstances(); i++) {
					Instance inst = test_dt.instance(i);
					String actual = inst.stringValue(inst.numAttributes() - 1);
					predict = sImCogumelos.classifier.classifyInstance(inst);
					pred = test_dt.classAttribute().value((int) (predict));
					System.out.println(actual + " \t " + pred);
				}

			}
			String filename = "shroom.fcl";
			FIS fis = FIS.load(filename, true);

			if (fis == null) {
				System.err.println("Can't load file: '" + filename + "'");
				System.exit(1);
			}
			FunctionBlock fb = fis.getFunctionBlock(null);

			fb.setVariable("sensor_esquerdo", distanceL);
			fb.setVariable("sensor_centro", distanceC);
			fb.setVariable("sensor_direito", distanceR);
			fb.setVariable("speed", speed);
			s.setRobotSpeed(speed);
			fb.setVariable("classification", getClassification(pred));
			fb.evaluate();

			System.out.println(getClassification(pred));
			double c = fb.getVariable("acao_realizada").defuzzify();
			if (c == 1.0) {
				System.out.println("no action");
				s.setAction(Action.NO_ACTION);

			}
			if (c == 3.0) {
				System.out.println("pick_up");
				s.setAction(Action.PICK_UP);

			}
			if (c == 2.0) {
				System.out.println("destroy");
				s.setAction(Action.DESTROY);

			}

			s.setRobotAngle(fb.getVariable("angle").defuzzify());

			s.step();

		}

	}

// TODO Auto-generated constructor stub
}
