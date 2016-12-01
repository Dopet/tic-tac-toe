package deepqnetwork;

import org.jblas.DoubleMatrix;
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;

public class NeuralNetworkTest {
    
    NeuralNetwork network;
    DoubleMatrix weights[];
    DoubleMatrix biasWeights[];
    
    @Before
    public void setUp() {
        network = new NeuralNetwork(2, 1);
        network.addHiddenLayer(3);
        initWeights();
    }
    
    private void initWeights() {
        weights = new DoubleMatrix[2];
        biasWeights = new DoubleMatrix[2];
        weights[0] = new DoubleMatrix(3,2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6);
        weights[1] = new DoubleMatrix(1,3, -0.1, -0.2, -0.3);
        biasWeights[0] = new DoubleMatrix(3,1, 1, 2, 3);
        biasWeights[1] = new DoubleMatrix(1,1, 4);
        network.setWeights(weights);
        network.setBiasWeights(biasWeights);
    }
    
    
    
    @Test
    public void gradientCheck() {
        double e = 0.00001;
        double target = 0.5;
        biasWeights[0] = new DoubleMatrix(3,1, 0, 0, 0);
        biasWeights[1] = new DoubleMatrix(1,1, 0);
        network.setBiasWeights(biasWeights);
        weights[0] = new DoubleMatrix(3,2, 0.1, 0.2, -0.3, 0.4, -0.5, 0.6);
        weights[1] = new DoubleMatrix(1,3, -0.1, 0.2, 0.3);
        network.setWeights(weights);
        network.backPropagate(new DoubleMatrix(1,1,target), network.forwardPropagate(new DoubleMatrix(2,1,1,2)));
        double analytic = network.getWeightGradients()[0].get(2);
        System.out.println(analytic);
        // [0,032000, 0,064000; 0,000000, 0,000000; -0,096000, -0,192000]
        biasWeights[0] = new DoubleMatrix(3,1, 0, 0, 0);
        biasWeights[1] = new DoubleMatrix(1,1, 0);
        network.setBiasWeights(biasWeights);
        weights[0] = new DoubleMatrix(3,2, 0.1, 0.2, -0.3+e, 0.4, -0.5, 0.6);
        weights[1] = new DoubleMatrix(1,3, -0.1, 0.2, 0.3);
        network.setWeights(weights);
        DoubleMatrix result1 = network.forwardPropagate(new DoubleMatrix(2,1,1,2));
        
        biasWeights[0] = new DoubleMatrix(3,1, 0, 0, 0);
        biasWeights[1] = new DoubleMatrix(1,1, 0);
        network.setBiasWeights(biasWeights);
        weights[0] = new DoubleMatrix(3,2, 0.1, 0.2, -0.3-e, 0.4, -0.5, 0.6);
        weights[1] = new DoubleMatrix(1,3, -0.1, 0.2, 0.3);
        network.setWeights(weights);
        DoubleMatrix result2 = network.forwardPropagate(new DoubleMatrix(2,1,1,2));
        
        double out1 = result1.get(0);
        double out2 = result2.get(0);
        double error1 = 0.5 * Math.pow( (target - out1) , 2);
        double error2 = 0.5 * Math.pow( (target - out2) , 2);
        
        double numerical = (error1 - error2 ) / (2 * e);
        
        double comparison = Math.abs(analytic - numerical) / Math.max(Math.abs(analytic), Math.abs(numerical));
        assertEquals(0 , comparison , 0.000000001);
    }
    
    
    @Test
    public void gradientCheck2() {
        double e = 0.00001;
        double target = 0.5;
        biasWeights[0] = new DoubleMatrix(3,1, 0, 0, 0);
        biasWeights[1] = new DoubleMatrix(1,1, 0);
        network.setBiasWeights(biasWeights);
        
        network.backPropagate(new DoubleMatrix(1,1,target), network.forwardPropagate(new DoubleMatrix(2,1,1,2)));
        double analytic = network.getWeightGradients()[0].get(5);
        
        biasWeights[0] = new DoubleMatrix(3,1, 0, 0, 0);
        biasWeights[1] = new DoubleMatrix(1,1, 0);
        network.setBiasWeights(biasWeights);
        weights[0] = new DoubleMatrix(3,2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6+e);
        weights[1] = new DoubleMatrix(1,3, -0.1, -0.2, -0.3);
        network.setWeights(weights);
        DoubleMatrix result1 = network.forwardPropagate(new DoubleMatrix(2,1,1,2));
        
        biasWeights[0] = new DoubleMatrix(3,1, 0, 0, 0);
        biasWeights[1] = new DoubleMatrix(1,1, 0);
        network.setBiasWeights(biasWeights);
        weights[0] = new DoubleMatrix(3,2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6-e);
        weights[1] = new DoubleMatrix(1,3, -0.1, -0.2, -0.3);
        network.setWeights(weights);
        DoubleMatrix result2 = network.forwardPropagate(new DoubleMatrix(2,1,1,2));
        
        double out1 = result1.get(0);
        double out2 = result2.get(0);
        double error1 = 0.5 * Math.pow( (target - out1) , 2);
        double error2 = 0.5 * Math.pow( (target - out2) , 2);
        
        double numerical = (error1 - error2 ) / (2 * e);
        double comparison = Math.abs(analytic - numerical) / Math.max(Math.abs(analytic), Math.abs(numerical));
        assertEquals(0 , comparison , 0.000000001);
    }
    
    @Test
    public void gradientCheck3() {
        double e = 0.00001;
        double target = 0.5;
        biasWeights[0] = new DoubleMatrix(3,1, 0, 0, 0);
        biasWeights[1] = new DoubleMatrix(1,1, 0);
        network.setBiasWeights(biasWeights);
        network.backPropagate(new DoubleMatrix(1,1,target), network.forwardPropagate(new DoubleMatrix(2,1,1,2)));
        double analytic = network.getBiasGradients()[0].get(0);
        
        biasWeights[0] = new DoubleMatrix(3,1, 0+e, 0, 0);
        biasWeights[1] = new DoubleMatrix(1,1, 0);
        network.setBiasWeights(biasWeights);
        weights[0] = new DoubleMatrix(3,2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6);
        weights[1] = new DoubleMatrix(1,3, -0.1, -0.2, -0.3);
        network.setWeights(weights);
        DoubleMatrix result1 = network.forwardPropagate(new DoubleMatrix(2,1,1,2));
        
        biasWeights[0] = new DoubleMatrix(3,1, 0-e, 0, 0);
        biasWeights[1] = new DoubleMatrix(1,1, 0);
        network.setBiasWeights(biasWeights);
        weights[0] = new DoubleMatrix(3,2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6);
        weights[1] = new DoubleMatrix(1,3, -0.1, -0.2, -0.3);
        network.setWeights(weights);
        DoubleMatrix result2 = network.forwardPropagate(new DoubleMatrix(2,1,1,2));
        
        double out1 = result1.get(0);
        double out2 = result2.get(0);
        double error1 = 0.5 * Math.pow( (target - out1) , 2);
        double error2 = 0.5 * Math.pow( (target - out2) , 2);
        
        double numerical = (error1 - error2 ) / (2 * e);
        double comparison = Math.abs(analytic - numerical) / Math.max(Math.abs(analytic), Math.abs(numerical));
        assertEquals(0 , comparison , 0.000000001);
    }
    
    
    
    @Test
    public void testForwardPropagate_correctResultWithVectorInput() {
        DoubleMatrix result = network.forwardPropagate(new DoubleMatrix(2, 1, 1, 2));
        double expResult = 0.9684505135724133;
        assertEquals(expResult, result.get(0), 0.000000000000001);
    }
    
    
    
    
    @Test
    public void testForwardPropagate_correctResultWithTwoColumnInput() {
        DoubleMatrix result = network.forwardPropagate(new DoubleMatrix(2, 2, 1, 2, 3, 4));
        assertEquals(0.9684505135724133, result.get(0, 0), 0.000000000000001);
        assertEquals(0.9679460396624073, result.get(0, 1), 0.000000000000001);
    }
    
    
    
    
    
    @Test
    public void testBackPropagate_deltaTestTwoColumnInput() {
        DoubleMatrix expected = new DoubleMatrix(1,2,0.5, 0.2);
        DoubleMatrix actual = network.forwardPropagate(new DoubleMatrix(2, 2, 1, 2, 3, 4));
        network.backPropagate(expected, actual);
        double outputDelta1 = 0.01431309149;
        assertEquals(outputDelta1, network.getDeltas()[1].get(0), 0.00000000001);
        double hidden11Delta = -0.0001619954383;
        double hidden12Delta = -0.0001077261218;
        double hidden13Delta = -0.00004665904298;
        assertEquals(hidden11Delta, network.getDeltas()[0].get(0, 0), 0.000000001);
        assertEquals(hidden12Delta, network.getDeltas()[0].get(1, 0), 0.000000001);
        assertEquals(hidden13Delta, network.getDeltas()[0].get(2, 0), 0.000000001);
        
        double outputDelta2 = 0.02382668084;
        assertEquals(outputDelta2, network.getDeltas()[1].get(1), 0.00000000001);
        double hidden21Delta = -0.0001177847194;
        double hidden22Delta = -0.0000469526557;
        double hidden23Delta = -0.0000130782753;
;
        assertEquals(hidden21Delta, network.getDeltas()[0].get(0, 1), 0.000000001);
        assertEquals(hidden22Delta, network.getDeltas()[0].get(1, 1), 0.000000001);
        assertEquals(hidden23Delta, network.getDeltas()[0].get(2, 1), 0.000000001);
    }
    
    @Test
    public void testBackPropagate_weightUpdateTestTwoColumnInput() {
        DoubleMatrix expected = new DoubleMatrix(1,2,0.5, 0.2);
        DoubleMatrix actual = network.forwardPropagate(new DoubleMatrix(2, 2, 1, 2, 3, 4));
        network.backPropagate(expected, actual);
        double weightUpdate1 = -0.00025767479825;
        double weightUpdate4 = -0.0003975648771;
        double weightUpdate7 = 0.017517433954945;
        assertEquals(weightUpdate1, network.getWeightGradients()[0].get(0), 0.00000001);
        assertEquals(weightUpdate4, network.getWeightGradients()[0].get(3), 0.00000001);
        assertEquals(weightUpdate7, network.getWeightGradients()[1].get(0), 0.00000001);
    }
    
    @Test
    public void testBackPropagate_weightTestTwoColumnInput() {
        DoubleMatrix expected = new DoubleMatrix(1,2,0.5, 0.2);
        DoubleMatrix actual = network.forwardPropagate(new DoubleMatrix(2, 2, 1, 2, 3, 4));
        network.backPropagate(expected, actual);
        double weight1 = 0.100025767479825;
        double weight4 = 0.40003975648771;
        double weight7 = -0.1017517433954945;
        assertEquals(weight1, network.getWeights()[0].get(0), 0.000000001);
        assertEquals(weight4, network.getWeights()[0].get(3), 0.000000001);
        assertEquals(weight7, network.getWeights()[1].get(0), 0.000000001);
    }
    
    @Test
    public void testBackPropagate_biasTestTwoColumnInput() {
        DoubleMatrix expected = new DoubleMatrix(1,2,0.5, 0.2);
        DoubleMatrix actual = network.forwardPropagate(new DoubleMatrix(2, 2, 1, 2, 3, 4));
        network.backPropagate(expected, actual);
        double biasWeight1 = 1.000013989007885;
        double biasWeight2 = 2.000007733938875;
        double biasWeight3 = 3.000002986865914;
        double biasWeight4 = 3.9980930113835;
        assertEquals(biasWeight1, network.getBiasWeights()[0].get(0), 0.0000000001);
        assertEquals(biasWeight2, network.getBiasWeights()[0].get(1), 0.0000000001);
        assertEquals(biasWeight3, network.getBiasWeights()[0].get(2), 0.0000000001);
        assertEquals(biasWeight4, network.getBiasWeights()[1].get(0), 0.000000000001);
    }


    @Test
    public void testBackPropagate_deltaTestVectorInput() {
        DoubleMatrix expected = new DoubleMatrix(1,1,0.5);
        DoubleMatrix actual = network.forwardPropagate(new DoubleMatrix(2, 1, 1, 2));
        network.backPropagate(expected, actual);
        double outputDelta = 0.01431309149;
        assertEquals(outputDelta, network.getDeltas()[1].get(0), 0.00000000001);
        double hidden1Delta = -0.0001619954383;
        double hidden2Delta = -0.0001077261218;
        double hidden3Delta = -0.00004665904298;
        assertEquals(hidden1Delta, network.getDeltas()[0].get(0), 0.000000001);
        assertEquals(hidden2Delta, network.getDeltas()[0].get(1), 0.000000001);
        assertEquals(hidden3Delta, network.getDeltas()[0].get(2), 0.000000001);
    }
    
    @Test
    public void testBackPropagate_weightUpdateTestVectorInput() {
        DoubleMatrix expected = new DoubleMatrix(1,1,0.5);
        DoubleMatrix actual = network.forwardPropagate(new DoubleMatrix(2, 1, 1, 2));
        network.backPropagate(expected, actual);
        double weightUpdate1 = -0.0001619954383;
        double weightUpdate4 = -0.0003239908766;
        double weightUpdate7 = 0.01245084378;
        assertEquals(weightUpdate1, network.getWeightGradients()[0].get(0), 0.000000001);
        assertEquals(weightUpdate4, network.getWeightGradients()[0].get(3), 0.00000001);
        assertEquals(weightUpdate7, network.getWeightGradients()[1].get(0), 0.00000001);
    }
    
    @Test
    public void testBackPropagate_weightTestVectorInput() {
        DoubleMatrix expected = new DoubleMatrix(1,1,0.5);
        DoubleMatrix actual = network.forwardPropagate(new DoubleMatrix(2, 1, 1, 2));
        network.backPropagate(expected, actual);
        double weight1 = 0.1000161995;
        double weight4 = 0.4000323991;
        double weight7 = -0.1012450844;
        assertEquals(weight1, network.getWeights()[0].get(0), 0.0000000001);
        assertEquals(weight4, network.getWeights()[0].get(3), 0.0000000001);
        assertEquals(weight7, network.getWeights()[1].get(0), 0.000000001);
    }
    
    @Test
    public void testBackPropagate_biasTestVectorInput() {
        DoubleMatrix expected = new DoubleMatrix(1,1,0.5);
        DoubleMatrix actual = network.forwardPropagate(new DoubleMatrix(2, 1, 1, 2));
        network.backPropagate(expected, actual);
        double biasWeight1 = 1.0000162;
        double biasWeight2 = 2.000010773;
        double biasWeight3 = 3.000004666;
        double biasWeight4 = 3.998568691;
        assertEquals(biasWeight1, network.getBiasWeights()[0].get(0), 0.000000001);
        assertEquals(biasWeight2, network.getBiasWeights()[0].get(1), 0.000000001);
        assertEquals(biasWeight3, network.getBiasWeights()[0].get(2), 0.000000001);
        assertEquals(biasWeight4, network.getBiasWeights()[1].get(0), 0.000000001);
    }


    @Test
    public void testAddHiddenLayer_addNoLayers() {
        NeuralNetwork instance = new NeuralNetwork(1,3);
        assertEquals(1, instance.getWeights().length);
        assertEquals(2, instance.getLayers().length);
        assertEquals(1, instance.getLayers()[0].rows);
        assertEquals(3, instance.getLayers()[1].rows);
        assertEquals(3, instance.getWeights()[0].rows);
    }
    
    @Test
    public void testAddHiddenLayer_addOneLayer() {
        NeuralNetwork instance = new NeuralNetwork(1,3);
        instance.addHiddenLayer(2);
        assertEquals(2, instance.getWeights().length);
        assertEquals(3, instance.getLayers().length);
        assertEquals(1, instance.getLayers()[0].rows);
        assertEquals(2, instance.getLayers()[1].rows);
        assertEquals(3, instance.getLayers()[2].rows);
        assertEquals(2, instance.getWeights()[0].rows);
    }
    
    @Test
    public void testAddHiddenLayer_addTwoLayers() {
        NeuralNetwork instance = new NeuralNetwork(1,3);
        instance.addHiddenLayer(2);
        instance.addHiddenLayer(4);
        assertEquals(3, instance.getWeights().length);
        assertEquals(4, instance.getLayers().length);
        assertEquals(1, instance.getLayers()[0].rows);
        assertEquals(2, instance.getLayers()[1].rows);
        assertEquals(4, instance.getLayers()[2].rows);
        assertEquals(3, instance.getLayers()[3].rows);
        assertEquals(2, instance.getWeights()[0].rows);
        assertEquals(4, instance.getWeights()[1].rows);
        assertEquals(2, instance.getWeights()[1].columns);
        assertEquals(4, instance.getWeights()[2].columns);
    }

    @Test
    public void testClone_cloneGetsSameValues() throws CloneNotSupportedException  {
        NeuralNetwork clone = (NeuralNetwork) network.clone();
        assertFalse(network == clone);
        assertTrue(network.getClass() == clone.getClass());
        assertTrue(clone.getWeights() == network.getWeights());
        DoubleMatrix[] x = new DoubleMatrix[2];
        x[0] = DoubleMatrix.ones(2);
        x[1] = DoubleMatrix.zeros(2);
        network.setWeights(x);
        assertFalse(clone.getWeights() == network.getWeights());
        clone.setWeights(x);
        assertTrue(clone.getWeights() == network.getWeights());
        assertTrue(clone.getLayers() == network.getLayers());
    }
    
    @Test
    public void testClone_changingOriginalValuesDoesntAffectClone() throws CloneNotSupportedException {
        NeuralNetwork clone = (NeuralNetwork) network.clone();
        DoubleMatrix[] testWeights = new DoubleMatrix[1];
        clone.setWeights(testWeights);
        assertFalse(clone.getWeights() == network.getWeights());
        clone.addHiddenLayer(5);
        assertFalse(clone.getLayers() == network.getLayers());
    }
}
