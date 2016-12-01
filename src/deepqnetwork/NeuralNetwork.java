package deepqnetwork;

import java.util.logging.FileHandler;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;


public class NeuralNetwork implements Cloneable {
    private final double ALPHA = 0.001;
    private final double BETA1 = 0.9;
    private final double BETA2 = 0.999;
    private final double EPSILON = 0.00000001;
    private int timestep;
    double totalError;
    private DoubleMatrix[] firstMomentVector;
    private DoubleMatrix[] secondMomentVector;
    
    private double alpha;
    private DoubleMatrix[] layers;
    private DoubleMatrix[] weights;
    private DoubleMatrix[] biasWeights;
    private DoubleMatrix[] deltas;
    private DoubleMatrix[] weightGradients;
    private DoubleMatrix[] biasGradients;
    Logger logger = Logger.getLogger("MyLog");  
    FileHandler fh;  

    public NeuralNetwork(int inputLayerNeuronCount, int outputLayerNeuronCount) {
        try {  
            fh = new FileHandler("MyLogFile.log", true); 
            logger.addHandler(fh);
            SimpleFormatter formatter = new SimpleFormatter();  
            fh.setFormatter(formatter);  
            logger.setUseParentHandlers(false);
        } catch (Exception e) {} 
        
        timestep = 0;
        totalError = 0;
        alpha = 0.1;
        
        layers = new DoubleMatrix[2];
        layers[0] = new DoubleMatrix(inputLayerNeuronCount);
        layers[1] = new DoubleMatrix(outputLayerNeuronCount);
        initMatrices();
    }
    
    public boolean addHiddenLayer(int neuronCount) {
        if (neuronCount <= 0) return false;
        layers = Util.addToArray(layers, layers.length - 1, new DoubleMatrix(neuronCount));
        initMatrices();
        return true;
    }
    
    /**
     * Forward propagates input through network
     * @param input DoubleMatrix Input of neural network
     * @return DoubleMatrix Return output of the forward propagation
     */
    public DoubleMatrix forwardPropagate(DoubleMatrix input) {
        if(input.rows != layers[0].rows) return null;
        layers[0] = input;
        for (int i = 0; i < weights.length; i++) {
            layers[i + 1] = weights[i].mmul(layers[i]).addColumnVector(biasWeights[i]); //Z
            layers[i + 1] = Util.sigmoid(layers[i + 1]); //Activation function
        }
        
        if (timestep % 2001 == 0 && timestep != 0){
            logger.log(Level.INFO, "Input: {0}", input.getColumn(0));
            logger.log(Level.INFO, "Q-values: {0}", layers[layers.length-1].getColumn(0));
        }
        
        return layers[layers.length - 1];
    }
    
    /**
     * Adjust neural network weights with mini-batch gradient descent
     */
    public boolean backPropagate(DoubleMatrix expected, DoubleMatrix actual) {
        timestep++;
        if (expected.columns != actual.columns)
            return false;
        calculateDeltas(expected, actual);
        calculateGradients(expected.columns);
        updateWeights();
        return true;
    }
    
    public void backPropagateWithAdam(DoubleMatrix expected, DoubleMatrix actual) {
        timestep++;
        calculateDeltas(expected, actual);
        calculateGradients(expected.columns);
        updateBiasedMomentEstimates();
        updateWeightsWithAdam();
    }
    
    private void calculateDeltas(DoubleMatrix expected, DoubleMatrix actual) {
        
        DoubleMatrix sub = expected.getColumn(0).sub(actual.getColumn(0));
        double error = MatrixFunctions.pow(sub, 2).mul(0.5).sum();
        totalError += error;
        
        if (timestep % 2001 == 0 && timestep != 0) {
            totalError = totalError / 2001;
            logger.log(Level.INFO, "Error: {0}", totalError);
            totalError = 0;
        }
        
        deltas[deltas.length - 1] = actual.sub(expected).mul(Util.sigmoidDerivate(actual)); //Output layer delta
        for (int i = weights.length - 1; i >= 1; i--) { //Hidden layer deltas
            deltas[i-1] = weights[i].transpose().mmul(deltas[i]).mul(Util.sigmoidDerivate(layers[i]));
        }
    }
    
    private void calculateGradients(int batchSize) {
        for (int i = weights.length - 1; i >= 0; i--) {
            weightGradients[i] = deltas[i].mmul(layers[i].transpose()).div(batchSize);
            biasGradients[i] = deltas[i].rowSums().div(batchSize);
        }
    }
    
    private void updateBiasedMomentEstimates() {
        for (int i=0; i<firstMomentVector.length; i++) {
            firstMomentVector[i] = firstMomentVector[i].mul(BETA1)
                    .add(weightGradients[i].mul(1 - BETA1));
            secondMomentVector[i] = secondMomentVector[i].mul(BETA2)
                    .add(weightGradients[i].mul(weightGradients[i]).mul(1 - BETA2));
        }
    }
    
    private double getModifiedAlpha() {
        return ALPHA * Math.sqrt(1 - Math.pow(BETA2, timestep))
                / (1 - Math.pow(BETA1, timestep));
    }
    
    private void updateWeightsWithAdam() {
        for (int i = 0; i < weights.length; i++) { 
            DoubleMatrix weightUpdate = firstMomentVector[i].mul(getModifiedAlpha())
                    .div(MatrixFunctions.sqrt(secondMomentVector[i]).add(EPSILON));
            
            weights[i] = weights[i].sub(weightUpdate);
            biasWeights[i] = biasWeights[i].sub(biasGradients[i].mul(0.05));
        }
    }
    
    private void updateWeights() {
        for (int i = 0; i < weights.length; i++) { 
            weights[i] = weights[i].sub(weightGradients[i].mul(alpha));
            biasWeights[i] = biasWeights[i].sub(biasGradients[i].rowSums().mul(alpha));
        }
    }
    
    private void initMatrices() {
        initWeights();
        initMomentVectors();
        initOtherMatrices();
    }
    
    private void initWeights() {
        weights = new DoubleMatrix[layers.length - 1];
        biasWeights = new DoubleMatrix[layers.length - 1];
        for (int i = 0; i < layers.length - 1; i++) {
            weights[i] = DoubleMatrix.randn(layers[i +1].rows, layers[i].rows).div(4);
            biasWeights[i] = DoubleMatrix.randn(layers[i+1].rows).div(3);
        }
    }
    
    private void initMomentVectors() {
        firstMomentVector = new DoubleMatrix[weights.length];
        secondMomentVector = new DoubleMatrix[weights.length];
        for (int i=0; i<firstMomentVector.length; i++) {
            firstMomentVector[i] = DoubleMatrix.zeros(1);
            secondMomentVector[i] = DoubleMatrix.zeros(1);
        }
    }
    
    private void initOtherMatrices() {
        deltas = new DoubleMatrix[weights.length];
        weightGradients = new DoubleMatrix[weights.length];
        biasGradients = new DoubleMatrix[weights.length];
    }
    
    @Override
    protected Object clone() throws CloneNotSupportedException {
        return super.clone();
    }
    
    /**
    * Test helper methods below
    */
    public void setWeights(DoubleMatrix[] weights) {
        this.weights = weights;
    }
    
    public void setBiasWeights(DoubleMatrix[] weights) {
        this.biasWeights = weights;
    }
    
    public DoubleMatrix[] getDeltas() {
        return deltas;
    }
    
    public DoubleMatrix[] getWeights() {
        return weights;
    }
    
    public DoubleMatrix[] getBiasWeights() {
        return biasWeights;
    }
    
    public DoubleMatrix[] getWeightGradients() {
        return weightGradients;
    }
    
    public DoubleMatrix[] getBiasGradients() {
        return biasGradients;
    }
    
    public DoubleMatrix[] getLayers() {
        return layers;
    }
    
}
