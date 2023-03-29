using MathNet.Numerics.LinearAlgebra;

namespace Femtools;

public static class CA
{
    public static int[] CalculateScore(string[][] reports, bool agentFirst = true, Random? random = default)
    {
        random ??= new Random();
        reports = agentFirst ? Transpose(reports) : reports;
        var m = reports.Length;
        var n = reports[0].Length;

        string[] options = reports.SelectMany(x => x).Distinct().ToArray();
        Dictionary<string, int> optionIndex = new Dictionary<string, int>();
        for (var i = 0; i < options.Length; i++) {
            optionIndex[options[i]] = i;
        }

        var groupA = Enumerable.Range(0, m).OrderBy(_ => random.Next()).Take(m / 2).ToArray();
        var groupB = Enumerable.Range(0, m).Except(groupA).ToArray();

        double[,] DeltaA = learning(reports.GetRows(groupA), options, optionIndex, random).GreaterThan(0);
        double[,] DeltaB = learning(reports.GetRows(groupB), options, optionIndex, random).GreaterThan(0);

        var score = new int[n];

        Scorer(reports.GetRows(groupA), DeltaB, optionIndex, ref score);
        Scorer(reports.GetRows(groupB), DeltaA, optionIndex, ref score);

        return score;
    }

    private static void Scorer(string[][] reports, double[,] scoreMatrix, IReadOnlyDictionary<string, int> optionIndex, ref int[] score)
    {
        var n = reports[0].Length;
        foreach (string[] task in reports)
        {
            for (var agent1 = 0; agent1 < n; agent1++)
            {
                for (var agent2 = 0; agent2 < agent1; agent2++)
                {
                    var report1 = optionIndex[task[agent1]];
                    var report2 = optionIndex[task[agent2]];
                    if (scoreMatrix[report1, report2] != 0d)
                    {
                        score[agent1]++;
                        score[agent2]++;
                    }
                }
            }
        }
    }

    private static double[,] learning(string[][] reports, string[] options, Dictionary<string, int> optionIndex, Random random)
    {
        var k = options.Length;
        var joint = new double[k, k];
        var alpha = 1.0 / reports.Length;
        foreach (string[] task in reports) {
            string[] sample = task.OrderBy(_ => random.Next()).Take(2).ToArray();
            var i = optionIndex[sample[0]];
            var j = optionIndex[sample[1]];
            joint[i, j] += alpha;
        }
        var marginal = new double[k];
        for (var i = 0; i < k; i++) {
            marginal[i] = joint.GetRow(i).Sum();
        }

        var prod =
            Matrix<double>.Build.DenseOfRowVectors(Vector<double>.Build.Dense(marginal)).Transpose() *
            Matrix<double>.Build.DenseOfRowVectors(Vector<double>.Build.Dense(marginal));
        return joint.Subtract(prod.ToArray());
    }

    private static T[][] Transpose<T>(IReadOnlyList<T[]> matrix)
    {
        var rowCount = matrix.Count;
        var colCount = matrix[0].Length;
        var result = new T[colCount][];
        for (var j = 0; j < colCount; j++)
        {
            result[j] = new T[rowCount];
            for (var i = 0; i < rowCount; i++)
            {
                result[j][i] = matrix[i][j];
            }
        }
        return result;
    }
}

internal static class ArrayExtensions
{
    public static T[][] GetRows<T>(this T[][] matrix, int[] indices)
    {
        var result = new T[indices.Length][];
        for (var i = 0; i < indices.Length; i++) {
            result[i] = matrix[indices[i]];
        }
        return result;
    }
    
    public static double[,] Subtract(this double[,] matrix, double[,] other)
    {
        var rowCount = matrix.GetLength(0);
        var colCount = matrix.GetLength(1);
        var result = new double[rowCount, colCount];
        for (var i = 0; i < rowCount; i++) {
            for (var j = 0; j < colCount; j++) {
                result[i, j] = matrix[i, j] - other[i, j];
            }
        }
        return result;
    }

    public static double[,] GreaterThan(this double[,] matrix, double threshold)
    {
        var rowCount = matrix.GetLength(0);
        var colCount = matrix.GetLength(1);
        var result = new double[rowCount, colCount];
        for (var i = 0; i < rowCount; i++) {
            for (var j = 0; j < colCount; j++) {
                result[i, j] = matrix[i, j] > threshold ? matrix[i, j] : 0d;
            }
        }
        return result;
    }

    public static double[] GetRow(this double[,] matrix, int rowIndex)
    {
        var colCount = matrix.GetLength(1);
        var result = new double[colCount];
        for (var j = 0; j < colCount; j++) {
            result[j] = matrix[rowIndex, j];
        }
        return result;
    }
}
