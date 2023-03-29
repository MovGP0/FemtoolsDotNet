using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace Femtools;

public static class DMI
{
    private static double Comb(int n, int m)
        => n.Factorial() / (m.Factorial() * (double)(n - m).Factorial());

    private static Matrix<double> GetM(int[] A, int[] B, int C)
    {
        if (A.Length != B.Length)
        {
            throw new ArgumentException( "Lengths of A and B must be equal.", nameof(A));
        }

        var M = DenseMatrix.Create(C, C, 0);
        for (var i = 0; i < A.Length; i++)
        {
            var x = A[i];
            var y = B[i];
            if (0 <= x && x < C && 0 <= y && y < C)
            {
                M[x, y]++;
            }
            else
            {
                throw new ArgumentOutOfRangeException(nameof(A), "The values of answers must be integers in [0, C)");
            }
        }

        return M;
    }

    private static double Dmi2(int[] A1, int[] B1, int[] A2, int[] B2, int C)
    {
        var M1 = GetM(A1, B1, C);
        var M2 = GetM(A2, B2, C);
        return M1.Determinant() * M2.Determinant();
    }

    public static double[] CalculateDMI(int[][] answers, int choice_n)
    {
        var agent_n = answers.Length;
        var task_n = answers[0].Length;

        if (task_n < 2 * choice_n)
        {
            throw new ArgumentOutOfRangeException(nameof(answers), "Insufficient number of tasks.");
        }

        if (agent_n <= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(answers), "Too few agents.");
        }

        // Transpose the answers array
        var transposedAnswers =
            Enumerable
                .Range(0, task_n)
                .Select(x => answers.Select(y => y[x]).ToArray())
                .ToArray();

        // Shuffle the tasks
        var random = new Random();
        transposedAnswers = transposedAnswers
            .OrderBy(x => random.Next())
            .ToArray();

        var half = task_n / 2;
        var T1 = transposedAnswers.Take(half).ToArray();
        var T2 = transposedAnswers.Skip(half).ToArray();

        // Transpose T1 and T2
        T1 = Enumerable.Range(0, agent_n).Select(x => T1.Select(y => y[x]).ToArray()).ToArray();
        T2 = Enumerable.Range(0, agent_n).Select(x => T2.Select(y => y[x]).ToArray()).ToArray();

        var payments = new double[agent_n];
        var norm_factor = (agent_n - 1) * Math.Pow(choice_n.Factorial(), 2) * Comb(half, choice_n) *
                          Comb(task_n - half, choice_n);
        for (var i = 0; i < agent_n; i++)
        {
            double p = 0;
            for (var j = 0; j < agent_n; j++)
            {
                if (i == j) continue;
                p += Dmi2(T1[i], T1[j], T2[i], T2[j], choice_n);
            }

            p /= norm_factor;
            payments[i] = p;
        }

        return payments;
    }
}