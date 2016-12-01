package deepqnetwork;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

public class Util {
    
    /**
     * Element-wise sigmoid
     * @param x DoubleMatrix x
     * @return DoubleMatrix 
     */
    protected static DoubleMatrix sigmoid(DoubleMatrix x) {
        if (x != null) {
            DoubleMatrix ones = DoubleMatrix.ones(x.rows, x.columns);
            return ones.div(ones.add(MatrixFunctions.exp(x.neg())));
        } 
        return null;
    }
    
    /**
     * Element-wise derivated sigmoid
     * @param x DoubleMatrix 
     * @return DoubleMatrix sigmoid'()
     */
    protected static DoubleMatrix sigmoidDerivate(DoubleMatrix x) {
        if (x != null) {
            DoubleMatrix ones = DoubleMatrix.ones(x.rows, x.columns);
            return x.mul(ones.sub(x));
        }
        return null;
    }
    
    protected static DoubleMatrix[] addToArray(DoubleMatrix[] a, int pos, DoubleMatrix item) {
    DoubleMatrix[] result = new DoubleMatrix[a.length + 1];
    for(int i = 0; i < pos; i++)
        result[i] = a[i];
    result[pos] = item;
    for(int i = pos + 1; i < result.length; i++)
        result[i] = a[i - 1];
    return result;
}
}
