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
        foreach (var perm in permutations)
        {
            var pos = perm.ToArray();
            var idx = pos.Select(p => (TensorIndex)p).ToArray();

            // calculate the value at the given position
            var value = CalculateAt(pos);
            t[idx] = tensor(value);
        }
        return t;
    }

    /// <summary>
    /// Calculates the value of the Levi-Civita tensor at a given position.
    /// </summary>
    public static long CalculateAt(params long[] a)
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
    private static IEnumerable<IEnumerable<long>> CombinationsWithRepetition(long[] input, int length)
    {
        if (length <= 0)
        {
            yield break;
        }

        foreach (var i in input)
        {
            if (length - 1 <= 0)
            {
                yield return new long[] { i };
            }

            foreach (var c in CombinationsWithRepetition(input, length - 1))
                yield return c.Append(i);
        }
    }
}
