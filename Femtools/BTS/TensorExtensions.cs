using static TorchSharp.torch;

namespace Femtools;

public static class TensorExtensions
{
    /// <summary>
    /// Calculates the geometric mean of the tensor for dimension <param name="dim"/>
    /// </summary>
    /// <param name="t">The tensor for which the mean should be calculated</param>
    /// <param name="dim">The dimension for which the geometric mean should be calculated</param>
    /// <returns>Geometric mean</returns>
    public static Tensor GeometricMean(this Tensor t, long dim = 0)
    {
        return exp(mean(log(t), new[] { dim }));
    }

    /// <summary>
    /// Calculates the average of the tensor for dimension <param name="dim"/>
    /// </summary>
    /// <param name="t"></param>
    /// <param name="dim"></param>
    /// <returns></returns>
    public static Tensor Average(this Tensor t, long dim = 0)
        => mean(t, new[] { dim });

    /// <summary>
    /// Normalizes input vectors of matrix,
    /// such that { e in input[i,j] | e in [0..1] }
    /// </summary>
    public static Tensor Normalize(this Tensor input)
    {
        // offset negative values
        var input_ = input - min(input);

        // sum rows
        var summed = einsum("ij->i", input_);

        // resize and transpose to match shape of input matrix
        var denominator = summed.expand(input.size()[1], input.size()[0]).t();

        // divide all elements by the sum of the vector they belong to
        return div(input_, denominator);
    }

    public static Tensor ToTensor(this double[,] values)
    {
        var numberOfVectors = (long)values.GetLength(0);
        var lengthOfVector = (long)values.GetLength(1);
        var array = values.AsEnumerable().ToArray();
        return tensor(array, dimensions: new[] { numberOfVectors, lengthOfVector }, dtype: ScalarType.Float32);
    }

    public static Tensor ToTensor(this double[] values)
    {
        var length = (long)values.GetLength(0);
        var array = values.AsEnumerable().ToArray();
        return tensor(array, dimensions: new[] { length }, dtype: ScalarType.Float32);
    }

    public static double[,] ToMatrix(this Tensor tensor)
    {
        var shape = tensor.shape;
        if (shape.Length != 2)
            throw new ArgumentException("Tensor is not of rank 2");

        var rows = shape[0];
        var cols = shape[1];

        var result = new double[rows, cols];
        for (var r = 0; r < rows; r++)
        {
            for (var c = 0; c < cols; c++)
            {
                result[r, c] = (double)tensor[r, c];
            }
        }
        return result;
    }

    public static double[] ToVector(this Tensor tensor)
    {
        var shape = tensor.shape;
        if (shape.Length != 1)
            throw new ArgumentException("Tensor is not of rank 1");

        var cols = shape[0];
        var result = new double[cols];
        for (var c = 0; c < cols; c++)
        {
            result[c] = (double)tensor[c];
        }
        return result;
    }

    public static IEnumerable<double> AsEnumerable(this double[,] values)
    {
        var numberOfVectors = values.GetLength(0);
        var lengthOfVector = values.GetLength(1);

        for (var i = 0; i < numberOfVectors; i++)
            for (var j = 0; j < lengthOfVector; j++)
            {
                yield return values[i, j];
            }
    }
}