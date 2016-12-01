package deepqnetwork;

import org.jblas.DoubleMatrix;
import org.junit.Test;
import static org.junit.Assert.*;


public class UtilTest {
    
    /**
     * Test of sigmoid method, of class Util.
     */
    @Test
    public void testSigmoid() {
        System.out.println("sigmoid");
        assertEquals(new DoubleMatrix(1, 1, 0.5), Util.sigmoid(DoubleMatrix.zeros(1))); //testataan sigmoidin arvo f(0)
        assertEquals(new DoubleMatrix(2, 2, 0.5, 0.5, 0.5, 0.5), Util.sigmoid(new DoubleMatrix(2, 2, 0, 0, 0, 0))); //testataan sama jokaiselle matriisin elementille
        assertEquals(null, Util.sigmoid(null)); //testataan sama jokaiselle matriisin elementille
    }

    /**
     * Test of sigmoidDerivate method, of class Util.
     */
    @Test
    public void testSigmoidDerivate() {
        System.out.println("sigmoidDerivate");
        DoubleMatrix sigmoid=Util.sigmoid(new DoubleMatrix (1,1,0));
        
        assertEquals(new DoubleMatrix (1,1,0.25), Util.sigmoidDerivate(sigmoid));
        assertEquals(null, Util.sigmoidDerivate(null));
    }

    /**
     * Test of addToArray method, of class Util.
     */
    @Test
    public void testAddToArray() {
        System.out.println("addToArray");
        DoubleMatrix[] a = new DoubleMatrix[2];
        a[0] = new DoubleMatrix(1,1,0);
        a[1] = new DoubleMatrix(1,1,2);
        
        DoubleMatrix item = DoubleMatrix.ones(1);
        
        DoubleMatrix[] expResult = new DoubleMatrix[3];
        expResult[0] = new DoubleMatrix(1,1,0);
        expResult[1] = new DoubleMatrix(1,1,1);
        expResult[2] = new DoubleMatrix(1,1,2);
        
        DoubleMatrix[] result = Util.addToArray(a, 1, item);
        assertArrayEquals(expResult, result);
    }
    
}
