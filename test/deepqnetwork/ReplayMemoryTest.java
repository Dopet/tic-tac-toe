package deepqnetwork;

import java.util.ArrayList;
import static junit.framework.TestSuite.warning;
import org.jblas.DoubleMatrix;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;


public class ReplayMemoryTest {
    
    /**
     * Test of add method, of class ReplayMemory.
     */
    @Test
    public void testAdd() {
        System.out.println("add method test");
        DoubleMatrix state = DoubleMatrix.zeros(1);
        int action = 0;
        int reward = 0;
        DoubleMatrix nextState = DoubleMatrix.ones(1);
        ReplayMemory instance = new ReplayMemory(2);
        assertEquals(instance.getSize(), 0);
        instance.add(state, action, reward, nextState);
        assertEquals(instance.getSize(), 1);
        instance.add(state, action, reward, nextState);
        assertEquals(instance.getSize(), 2);
        instance.add(state, action, reward, nextState);
        assertEquals(instance.getSize(), 2);
        
        DoubleMatrix state2 = DoubleMatrix.ones(1);
        int action2 = 1;
        int reward2 = 1;
        DoubleMatrix nextState2 = DoubleMatrix.zeros(1);
        
        
        instance.add(state2, action2, reward2, nextState2);
        assertEquals(instance.getSize(), 2);
        instance.add(state2, action2, reward2, nextState2);
        assertEquals(instance.getSize(), 2);
        assertEquals(1, instance.getRandom().getAction());
    }

    /**
     * Test of getRandom method, of class ReplayMemory.
     */
    @Test
    public void testGetRandom() {
        System.out.println("---------------------------");
        System.out.println("getRandom method test");
        ReplayMemory instance = new ReplayMemory(2);
        instance.add(DoubleMatrix.ones(1), 1, 1, DoubleMatrix.ones(1));
        instance.add(DoubleMatrix.zeros(1), 0, 0, DoubleMatrix.zeros(1));
        
        ArrayList<Replay> table1 = new ArrayList();
        ArrayList<Replay> table2 = new ArrayList();
        Replay replay;
        for (int i=0; i<100000; i++) {
            replay = instance.getRandom();
            if (replay.getAction() == 1) {
                table1.add(replay);
            } else {
                table2.add(replay);
            }
        }
        assertEquals(table1.size(), table2.size(), 1000);
        // TODO review the generated test code and remove the default call to fail.
        System.out.println("Check that results are near each other");
        System.out.println("Table 1 size: " + table1.size());
        System.out.println("Table 2 size: " + table2.size());
    }
    
}
