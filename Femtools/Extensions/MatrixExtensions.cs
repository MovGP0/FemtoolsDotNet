using MathNet.Numerics.LinearAlgebra;

namespace Femtools;

public static class MatrixExtensions
{
    /// <summary>
    /// Shuffle the row indices using the Fisher-Yates algorithm
    /// </summary>
    /// <param name="matrix">The matrix to shuffle.</param>
    /// <param name="rng">Random number generator.</param>
    /// <returns>A new matrix where the rows are shuffled.</returns>
    public static Matrix<double> RandomizeRows(this Matrix<double> matrix, Random? rng = default)
    {
        rng ??= new Random();
        var rowIndices = Enumerable.Range(0, matrix.RowCount).ToArray();
        
        // shuffle indices using Fisher-Yates
        for (var i = rowIndices.Length - 1; i >= 1; i--)
        {
            var j = rng.Next(i + 1);
            if (j == i) continue;
            (rowIndices[i], rowIndices[j]) = (rowIndices[j], rowIndices[i]);
        }
        
        // create new matrix using the shuffled indices
        var shuffledMatrix = Matrix<double>.Build.Dense(matrix.RowCount, matrix.ColumnCount);
        foreach (var rowIndex in rowIndices)
        {
            shuffledMatrix.SetRow(rowIndex, matrix.Row(rowIndex));
        }

        return shuffledMatrix;
    }
}
