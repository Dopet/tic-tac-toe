package deepqnetwork;

import java.util.Random;
import org.jblas.DoubleMatrix;


public class NeuralQLearner {
    
    private final int BATCH_SIZE = 32;
    private final int TARGET_NETWORK_UPDATE_FREQ = 10000;
    private final int REPLAY_START_SIZE = 50000;
    private final int REPLAY_CAPACITY = 1000000;
    private final int ACTION_REPEAT = 1;
    private final int FRAMES_TO_EPSILON_BOTTOM = 1000000;

    private double epsilon;
    private double gamma;
    private int legalActions;
    private int lastAction;
    private int actionRepeated;
    private int framesCompleted;
    private int inputNeurons;
    private double[] expectedValue;
    private Replay[] replays;
    private ReplayMemory replayMemory;
    private Replay replay;
    private NeuralNetwork targetNetwork;
    private NeuralNetwork neuralNetwork;
    private DoubleMatrix actual;
    private DoubleMatrix expected;
    
    public NeuralQLearner(int inputNeurons, int outputs, double gamma) {
        neuralNetwork = new NeuralNetwork(inputNeurons, outputs);
        neuralNetwork.addHiddenLayer(32);
        
        expectedValue = new double[BATCH_SIZE];
        replays = new Replay[BATCH_SIZE];
        this.inputNeurons = inputNeurons;
        this.gamma = gamma;
        this.legalActions = outputs;
        actionRepeated = 0;
        epsilon = 1.0;
        framesCompleted = 0;
        replayMemory = new ReplayMemory(REPLAY_CAPACITY);
    }

    /**
     * Returns action to agent using epsilon-greedy strategy
     * @param state DoubleMatrix input/state the agent is currently in
     * @return int Returns action to agent
     */
    public int getAction(DoubleMatrix state) {
        if(actionRepeated != 0 && actionRepeated < ACTION_REPEAT) {
            actionRepeated++;
            return lastAction;
        }
        actionRepeated = 0;
        if (Math.random() <= epsilon) {
            lastAction = new Random().nextInt(legalActions);
            actionRepeated++;
            return lastAction;
        } else {
            actionRepeated++;
            lastAction = neuralNetwork.forwardPropagate(state).argmax();
            return lastAction;
        }
        
    }
    
    private void updateRandomness() {
        if (framesCompleted == 10000000) {
            epsilon = 0;
        }
        //Epsilon to 0.1 in about 1 000 000 frames
        if (epsilon > 0.1 && framesCompleted >= REPLAY_START_SIZE) {
            epsilon -= (0.9 / FRAMES_TO_EPSILON_BOTTOM);
        }
    }
    
  /**
     * Update neural network weights
     * @param state DoubleMatrix State from which the agent left from
     * @param action byte Action the agent did from state
     * @param reward byte Reward the agent got for doing the action from state (-1, 0 or 1)
     * @param nextState DoubleMatrix State the agent ended after choosing action from state
     * @throws CloneNotSupportedException 
     */
    public void update(DoubleMatrix state, int action, double reward, DoubleMatrix nextState) {
        updateRandomness();
        if (framesCompleted < REPLAY_START_SIZE) {
            replayMemory.add(state, action, reward, nextState);
        } else {
            if(framesCompleted % TARGET_NETWORK_UPDATE_FREQ == 0) {
                try {
                    targetNetwork =  (NeuralNetwork) neuralNetwork.clone();
                } catch (CloneNotSupportedException e) {
                    e.printStackTrace();
                }
                
            }
            replayMemory.add(state, action, reward, nextState);
            createBatches();
            updateNetworkWeights();
        }
        framesCompleted++;
        
    }
    
    private void createBatches() {
        createActualResult();
        createExpectedResult();
    }
    
    private void updateNetworkWeights() {
        neuralNetwork.backPropagateWithAdam(expected, actual);
    }
    
    private void createActualResult() {
        actual = neuralNetwork.forwardPropagate(getInputMatrix());
    }
    
    private DoubleMatrix getInputMatrix() {
        DoubleMatrix result = new DoubleMatrix(inputNeurons, BATCH_SIZE);
        for (int i=0; i < BATCH_SIZE; i++) {
            replay = replayMemory.getRandom();
            result.putColumn(i, replay.getState());
            if (replay.getReward() == 0) {
                expectedValue[i] = replay.getReward() + gamma * targetNetwork.forwardPropagate(replay.getNextState()).max();
            } else {
                expectedValue[i] = replay.getReward();
            }
            
            replays[i] = replay;
        }
        return result;
    }
    
    private void createExpectedResult() {
        expected = actual.dup();
        for (int i=0; i<BATCH_SIZE; i++) {
            expected.put(replays[i].getAction(), i, expectedValue[i]);
        }
    }
    
    
    
    
    
    
}
