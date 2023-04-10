using System.Diagnostics;
using MathNet.Numerics.LinearAlgebra;
using static Femtools.AssertionHelpers;
using static System.Math;

namespace Femtools;

/// <summary>
/// Dominantly Truthful Multi-task Peer Prediction with a Constant Number of Tasks (DT-MPP)
/// is a mechanism design framework for incentivizing workers in a multi-task setting to provide
/// truthful feedback. The mechanism is designed to be dominant strategy truthful,
/// meaning that it is always in the best interest of a worker to report their true feedback,
/// regardless of the strategies chosen by other workers.
/// </summary>
/// <remarks>
/// See the <a href="https://arxiv.org/abs/1911.00272">Dominantly Truthful Multi-task Peer Prediction with a Constant Number of Tasks</a> paper by Yuqing Kong.
/// </remarks>
public static class DMI
{
    /// <summary>
    /// Calculates the payments for the agents based on the DT-MPP algorithm.
    /// </summary>
    /// <param name="answers">
    /// A matrix of answers, where the rows represent the agents and the column represent the questions.
    /// The values in the fields represent the answers.</param>
    /// <param name="numberOfChoices">The number of possible choices</param>
    /// <param name="random">Random value generator</param>
    /// <returns>The payments for a given worker, where result[i] is the payment for worker i.</returns>
    /// <exception cref="ArgumentOutOfRangeException"></exception>
    public static double[] CalculatePayments(Matrix<double> answers, int numberOfChoices, Random? random = null)
    {
        random ??= new Random();
        var numberOfAgents = answers.RowCount;
        var numberOfTasks = answers.ColumnCount;

        if (numberOfTasks < 2 * numberOfChoices) throw new ArgumentOutOfRangeException(nameof(answers), "Insufficient number of tasks.");
        if (numberOfAgents <= 1) throw new ArgumentOutOfRangeException(nameof(answers), "Too few agents.");
 
        var randomizedAnswers = answers
            .Transpose()
            .RandomizeRows(random);

        var half = numberOfTasks / 2;

        var T1 = randomizedAnswers
            .SubMatrix(0, half, 0, randomizedAnswers.ColumnCount)
            .Transpose();

        var T2 = randomizedAnswers
            .SubMatrix(half, randomizedAnswers.RowCount-half, 0, randomizedAnswers.ColumnCount)
            .Transpose();

        var payments = new double[numberOfAgents];
        var norm_factor = CalculateNormFactor(numberOfAgents, numberOfTasks, numberOfChoices, half);

        for (var i = 0; i < numberOfAgents; i++)
        {
            var payment = 0d;
            for (var j = 0; j < numberOfAgents; j++)
            {
                if (i == j) continue;
                payment += CalculatePaymentForAgent(T1.Row(i), T1.Row(j), T2.Row(i), T2.Row(j), numberOfChoices);
            }

            payment /= norm_factor;
            payments[i] = payment;
        }

        return payments;
    }

    private static Matrix<double> GetM(IList<double> a, IList<double> b, int rangeOfAnswers)
    {
        AssertEqualCount(a, b);
        AssertAllElementsInRange(a, 0, rangeOfAnswers);
        AssertAllElementsInRange(b, 0, rangeOfAnswers);
        
        var matrix = Matrix<double>.Build.Dense(rangeOfAnswers, rangeOfAnswers);
        for (var i = 0; i < a.Count; i++)
        {
            var x = (int)a[i];
            var y = (int)b[i];
            matrix[x, y]++;
            matrix[y, x]++;
        }

        return matrix;
    }

    private static double CalculatePaymentForAgent(IList<double> a1, IList<double> b1, IList<double> a2, IList<double> b2, int rangeOfAnswers)
    {
        var m1 = GetM(a1, b1, rangeOfAnswers);
        var m2 = GetM(a2, b2, rangeOfAnswers);
        return m1.Determinant() * m2.Determinant();
    }

    private static double CalculateNormFactor(int numberOfAgents, int numberOfTasks, int numberOfChoices, int split)
    {
        Debug.Assert(split < numberOfTasks);

        return (numberOfAgents - 1)
               * Pow(numberOfChoices.Factorial(), 2)
               * split.Comb(numberOfChoices)
               * (numberOfTasks - split).Comb(numberOfChoices);
    }
}