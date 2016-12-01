package deepqnetwork;

import org.jblas.DoubleMatrix;
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;


public class XorTest {
    
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
        weights[0] = new DoubleMatrix(3,2, 0.52, 0.22, 0.15, 0.84, 0.93, 0.22);
        weights[1] = new DoubleMatrix(1,3, 0.26, 0.52, 0.41);
        biasWeights[0] = new DoubleMatrix(3,1, 0.59, 0.76, 0.89);
        biasWeights[1] = new DoubleMatrix(1,1, 0.26);
        network.setWeights(weights);
        network.setBiasWeights(biasWeights);
    }
    
    //0.215sek vs 0.17sek
    @Test
    public void testXor_batchGradientDescentWithAdam() throws CloneNotSupportedException {
        DoubleMatrix actual;
        DoubleMatrix[] input = new DoubleMatrix[4];
        DoubleMatrix batchInput = new DoubleMatrix(2,4,0,0,0,1,1,0,1,1);
        input[0] = new DoubleMatrix(2,1,0,0);
        input[1] = new DoubleMatrix(2,1,0,1);
        input[2] = new DoubleMatrix(2,1,1,0);
        input[3] = new DoubleMatrix(2,1,1,1);
        DoubleMatrix[] expected = new DoubleMatrix[4];
        expected[0] = new DoubleMatrix(1,4,0,1,1,0);
        for(int i=0 ; i<20000; i++) {
            actual = network.forwardPropagate(batchInput);
            network.backPropagateWithAdam(expected[0], actual);
        }
        assertEquals(0, network.forwardPropagate(input[0]).get(0), 0.03);
        assertEquals(1, network.forwardPropagate(input[1]).get(0), 0.03);
        assertEquals(1, network.forwardPropagate(input[2]).get(0), 0.03);
        assertEquals(0, network.forwardPropagate(input[0]).get(0), 0.03);
    }

    @Test
    public void testXor_stochasticGradientDescent() {
        DoubleMatrix actual;
        DoubleMatrix[] input = new DoubleMatrix[4];
        input[0] = new DoubleMatrix(2,1,0,0);
        input[1] = new DoubleMatrix(2,1,0,1);
        input[2] = new DoubleMatrix(2,1,1,0);
        input[3] = new DoubleMatrix(2,1,1,1);
        DoubleMatrix[] expected = new DoubleMatrix[4];
        expected[0] = new DoubleMatrix(1,1,0);
        expected[1] = new DoubleMatrix(1,1,1);
        expected[2] = new DoubleMatrix(1,1,1);
        expected[3] = new DoubleMatrix(1,1,0);
        for(int i=0 ; i<81000; i++) {
            actual = network.forwardPropagate(input[i%4]);
            network.backPropagate(expected[i%4], actual);
        }
        assertEquals(0, network.forwardPropagate(input[0]).get(0), 0.03);
        assertEquals(1, network.forwardPropagate(input[1]).get(0), 0.03);
        assertEquals(1, network.forwardPropagate(input[2]).get(0), 0.03);
        assertEquals(0, network.forwardPropagate(input[0]).get(0), 0.03);
    }
    
    @Test
    public void testXor_batchGradientDescent() {
        DoubleMatrix actual;
        DoubleMatrix[] input = new DoubleMatrix[4];
        DoubleMatrix batchInput = new DoubleMatrix(2,4,0,0,0,1,1,0,1,1);
        input[0] = new DoubleMatrix(2,1,0,0);
        input[1] = new DoubleMatrix(2,1,0,1);
        input[2] = new DoubleMatrix(2,1,1,0);
        input[3] = new DoubleMatrix(2,1,1,1);
        DoubleMatrix[] expected = new DoubleMatrix[4];
        expected[0] = new DoubleMatrix(1,4,0,1,1,0);
        for(int i=0 ; i<79000; i++) {
            actual = network.forwardPropagate(batchInput);
            network.backPropagate(expected[0], actual);
        }
        assertEquals(0, network.forwardPropagate(input[0]).get(0), 0.03);
        assertEquals(1, network.forwardPropagate(input[1]).get(0), 0.03);
        assertEquals(1, network.forwardPropagate(input[2]).get(0), 0.03);
        assertEquals(0, network.forwardPropagate(input[0]).get(0), 0.03);
    }
    
    @Test
    public void testXor_minibatchGradientDescent() {
        DoubleMatrix actual;
        DoubleMatrix[] input = new DoubleMatrix[4];
        DoubleMatrix[] batchInputs = new DoubleMatrix[4];
        batchInputs[0] = new DoubleMatrix(2,2,0,0,1,1);
        batchInputs[1] = new DoubleMatrix(2,2,0,1,1,0);
        batchInputs[2] = new DoubleMatrix(2,2,0,0,1,0);
        batchInputs[3] = new DoubleMatrix(2,3,1,1,1,0,0,0);
        input[0] = new DoubleMatrix(2,1,0,0);
        input[1] = new DoubleMatrix(2,1,0,1);
        input[2] = new DoubleMatrix(2,1,1,0);
        input[3] = new DoubleMatrix(2,1,1,1);
        DoubleMatrix[] expected = new DoubleMatrix[4];
        expected[0] = new DoubleMatrix(1,2,0,0);
        expected[1] = new DoubleMatrix(1,2,1,1);
        expected[2] = new DoubleMatrix(1,2,0,1);
        expected[3] = new DoubleMatrix(1,3,0,1,0);
        for(int i=0 ; i<260000; i++) {
            actual = network.forwardPropagate(batchInputs[i%4]);
            network.backPropagate(expected[i%4], actual);
        }
        assertEquals(0, network.forwardPropagate(input[0]).get(0), 0.03);
        assertEquals(1, network.forwardPropagate(input[1]).get(0), 0.03);
        assertEquals(1, network.forwardPropagate(input[2]).get(0), 0.03);
        assertEquals(0, network.forwardPropagate(input[0]).get(0), 0.03);
    }
}
