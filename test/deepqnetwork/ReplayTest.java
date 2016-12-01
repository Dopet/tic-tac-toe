package deepqnetwork;

import org.jblas.DoubleMatrix;
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;


public class ReplayTest {
    Replay instance;
    
    @Before
    public void BeforeTestMethod() {
        instance = new Replay(DoubleMatrix.zeros(1), 2, 1, DoubleMatrix.ones(1));
    }

    /**
     * Test of getState method, of class Replay.
     */
    @Test
    public void testGetState() {
        System.out.println("getState");
        DoubleMatrix expResult = DoubleMatrix.zeros(1);
        DoubleMatrix result = instance.getState();
        assertEquals(expResult, result);
    }

    /**
     * Test of getAction method, of class Replay.
     */
    @Test
    public void testGetAction() {
        System.out.println("getAction");
        int expResult = 2;
        int result = instance.getAction();
        assertEquals(expResult, result);
    }

    /**
     * Test of getReward method, of class Replay.
     */
    @Test
    public void testGetReward() {
        System.out.println("getReward");
        double expResult = 1;
        double result = instance.getReward();
        assertEquals(expResult, result, 0);
    }

    /**
     * Test of getNextState method, of class Replay.
     */
    @Test
    public void testGetNextState() {
        System.out.println("getNextState");
        DoubleMatrix expResult = DoubleMatrix.ones(1);
        DoubleMatrix result = instance.getNextState();
        assertEquals(expResult, result);
    }
    
}
