package deepqnetwork;

import org.jblas.DoubleMatrix;

public class Replay {
    
    private DoubleMatrix state;
    private int action;
    private double reward;
    private DoubleMatrix nextState;

    /**
     * 
     * @param state DoubleMatrix State from which the agent left from
     * @param action int Action the agent did from state
     * @param reward int Reward the agent got for doing the action from state (-1, 0 or 1)
     * @param nextState DoubleMatrix State the agent ended after choosing action from state
     */
    public Replay(DoubleMatrix state, int action, double reward, DoubleMatrix nextState) {
        this.state = state;
        this.action = action;
        this.reward = reward;
        this.nextState = nextState;
    }
    
    /**
     * 
     * @return DoubleMatrix State
     */
    public DoubleMatrix getState() {
        return state;
    }

    /**
     * 
     * @return int Action
     */
    public int getAction() {
        return action;
    }

    /**
     * 
     * @return int Reward
     */
    public double getReward() {
        return reward;
    }

    /**
     * 
     * @return DoubleMatrix Next state
     */
    public DoubleMatrix getNextState() {
        return nextState;
    }
    
    
}
