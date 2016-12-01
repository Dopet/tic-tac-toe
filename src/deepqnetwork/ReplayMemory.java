package deepqnetwork;

import java.util.ArrayList;
import java.util.Random;
import org.jblas.DoubleMatrix;

public class ReplayMemory {
    private ArrayList<Replay> replayMemory;
    private int replayCapacity;

    /**
     * 
     * @param replayCapacity int Maximum amount of items to store in ReplayMemory
     */
    public ReplayMemory(int replayCapacity) {
        this.replayMemory = new ArrayList();
        this.replayCapacity = replayCapacity;
    }
    
    /**
     * Add an item to ReplayMemory
     * @param state DoubleMatrix State from which the agent left from
     * @param action int Action the agent did from state
     * @param reward int Reward the agent got for doing the action from state (-1, 0 or 1)
     * @param nextState DoubleMatrix State the agent ended after choosing action from state
     */
    public void add(DoubleMatrix state, int action, double reward, DoubleMatrix nextState) {
        if(replayMemory.size() >= replayCapacity) {
            replayMemory.add(0, new Replay(state, action, reward, nextState));
            replayMemory.remove(replayCapacity);
        } else {
        replayMemory.add(new Replay(state, action, reward, nextState));
        }
    }
    
    /**
     * Get random Replay from memory
     * @return Replay Returns random Replay from memory
     */
    public Replay getRandom() {
        int rnd = new Random().nextInt(replayMemory.size());
        return replayMemory.get(rnd);
    }
    
    public int getSize() {
        return replayMemory.size();
    }
    
}
