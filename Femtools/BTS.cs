using static System.Diagnostics.Debug;
using static TorchSharp.torch;

namespace Femtools;

public sealed class BTS
{
    /// <summary>
    /// Calculates the Bayesian Truth Serum (BTS) scores.
    /// </summary>
    /// <param name="answers">Answers</param>
    /// <param name="frequencies">Predicted frequencies</param>
    /// <param name="α">Correlation factor</param>
    /// <returns>BTS Scores</returns>
    public static double[] Calculate(double[,] answers, double[,] frequencies, double α = 1.0f)
    {
        if (answers.GetLength(1) != frequencies.GetLength(1))
        {
            throw new InvalidOperationException("The dimension [1] of answers and frequencies must be equal.");
        }

        if (α is < 0d or > 1d)
        {
            throw new ArgumentOutOfRangeException(nameof(α), α, "Must be between 0 and 1.");
        }

        // offset to prevent division by zero
        const double σ = 0.000001d;

        // normalize input tensors to e∈[0..1]
        var x = answers.ToTensor().Normalize() + σ;
        var f = frequencies.ToTensor().Normalize() + σ;

        var ξ = x.Average();
        var φ = f.GeometricMean();
        EnsureShapeOfTensors(x, f, ξ, φ);

        var score = InformationScore(x, ξ, φ) + PredictionScore(f, ξ) * α;
        return score.ToVector();
    }

    /// <summary>
    /// Measures how common a given answer in x is.
    /// </summary>
    /// <param name="x"></param>
    /// <param name="ξ"></param>
    /// <param name="φ"></param>
    /// <returns>BTS information score</returns>
    private static Tensor InformationScore(Tensor x, Tensor ξ, Tensor φ)
        => einsum("kr,r->k", x, log(div(ξ, φ)));

    /// <summary>
    /// Calculates the
    /// <a href="https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence">
    /// Kullback–Leibler divergence
    /// </a> between true and predicted frequencies of answers.
    /// </summary>
    /// <param name="f">Observed Frequencies matrix</param>
    /// <param name="ξ">Vector of averages of the predictions</param>
    /// <returns>BTS prediction score</returns>
    private static Tensor PredictionScore(Tensor f, Tensor ξ)
        => einsum("r,kr->k", ξ, log(div(f, ξ)));

    #region Debug
    [System.Diagnostics.Conditional("DEBUG")]
    private static void EnsureShapeOfTensors(Tensor x, Tensor f, Tensor ξ, Tensor φ)
    {
        Assert(x.Dimensions == 2);
        Assert(f.Dimensions == 2);
        Assert(x.shape.SequenceEqual(f.shape));
        Assert(ξ.Dimensions == 1);
        Assert(φ.Dimensions == 1);
        Assert(ξ.shape.SequenceEqual(φ.shape));
        Assert(ξ.shape[0] == x.shape[1]);

        EnsureAnswersAreValid(x);
        EnsureFrequenciesAreValid(f);
    }

    [System.Diagnostics.Conditional("DEBUG")]
    private static void EnsureAnswersAreValid(Tensor x)
    {
        var xm = x.ToMatrix();
        for (var row = 0; row < xm.GetLength(0); row++)
        for (var col = 0; col < xm.GetLength(1); col++)
        {
            if (xm[row, col] is < 0d or > 1.000001d)
            {
                throw new ArgumentOutOfRangeException($"answer at [{row},{col}] was not in [0,1]");
            }
        }

        const string message = "answers must statisfy {e_s = Σ_k x_ks = 1}";
        var xs = einsum("ks->k", x).ToVector();
        if (xs.Any(y => Math.Abs(y - 1) > 0.00001))
        {
            throw new ArgumentOutOfRangeException(message);
        }
    }

    [System.Diagnostics.Conditional("DEBUG")]
    private static void EnsureFrequenciesAreValid(Tensor f)
    {
        const string message = "frequencies must statisfy {e_s = Σ_k y_ks = 1}";

        var ys = einsum("ks->k", f).ToVector();
        if (ys.Any(y => Math.Abs(y - 1) > 0.00001))
        {
            throw new ArgumentOutOfRangeException(message);
        }
    }
    #endregion
}