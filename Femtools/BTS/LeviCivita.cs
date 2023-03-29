using static TorchSharp.torch;

namespace Femtools;

public static class LeviCivita
{
    public static Tensor OfRank(int dimension)
    {
        // calculate the sizes for the tensor
        var size = new long[dimension];
        for (var i = 0L; i < dimension; i++)
            size[i] = dimension;

        // create the tensor
        var t = zeros(size, dtype: ScalarType.Int64);

        // create list of numbers [0..n] that get permuted for the position
        var a = new long[dimension];
        for (var i = 0L; i < dimension; i++)
        {
            a[i] = i;
        }

        // permute over all positions of the tensor
        var permutations = CombinationsWithRepetition(a, a.Length);
        for(var i = 0; i < permutations.GetLength(0); i++)
        {
            var pos = GetRow(permutations, i);
            var idx = pos.Select(p => (TensorIndex)p).ToArray();

            // calculate the value at the given position
            var value = CalculateAt(pos);
            t[idx] = tensor(value);
        }
        return t;
    }

    private static long[] GetRow(long[,] matrix, int rowIndex)
    {
        if (rowIndex < 0 || rowIndex >= matrix.GetLength(0))
        {
            throw new ArgumentOutOfRangeException(nameof(rowIndex), "Row index is out of range.");
        }

        var columns = matrix.GetLength(1);
        var row = new long[columns];

        for (var i = 0; i < columns; i++)
        {
            row[i] = matrix[rowIndex, i];
        }

        return row;
    }

    /// <summary>
    /// Calculates the value of the Levi-Civita tensor at a given position.
    /// </summary>
    private static long CalculateAt(params long[] a)
    {
        var productSum = 1L;
        for (var i = 0; i < a.Length; i++)
        for (var j = i + 1; j < a.Length; j++)
        {
            productSum *= Math.Sign(a[j] - a[i]);
        }
        return productSum;
    }

    /// <summary>
    /// Calculates all permutations of the input including repetitions
    /// </summary>
    private static long[,] CombinationsWithRepetition(long[] input, int length)
    {
        if (length <= 0)
        {
            return new long[0, 0];
        }

        var inputSize = input.Length;
        var result = new long[(int)Math.Pow(inputSize, length), length];
        CombinationsWithRepetitionHelper(input, length, result, new long[length], 0, 0);
        return result;
    }

    private static int CombinationsWithRepetitionHelper(
        long[] input,
        int length,
        long[,] result,
        long[] current,
        int currentLength,
        int resultIndex)
    {
        if (currentLength == length)
        {
            for (var i = 0; i < length; i++)
            {
                result[resultIndex, i] = current[i];
            }
            return resultIndex + 1;
        }

        foreach (var t1 in input)
        {
            current[currentLength] = t1;
            resultIndex = CombinationsWithRepetitionHelper(input, length, result, current, currentLength + 1, resultIndex);
        }

        return resultIndex;
    }
}
