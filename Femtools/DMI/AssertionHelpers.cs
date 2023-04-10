namespace Femtools;

public static class AssertionHelpers
{
    /// <summary>
    /// Asserts that two collections have equal counts.
    /// </summary>
    /// <typeparam name="T">The type of elements in the collections.</typeparam>
    /// <param name="a">The first collection.</param>
    /// <param name="b">The second collection.</param>
    /// <exception cref="ArgumentException">Thrown when the lengths of the two collections are not equal.</exception>
    public static void AssertEqualCount<T>(ICollection<T> a, ICollection<T> b)
    {
        if (a.Count != b.Count)
            throw new ArgumentException( "The lengths of both collections must be equal.", nameof(a));
    }
    
    /// <summary>
    /// Asserts that all elements in the specified collection are within the specified range.
    /// </summary>
    /// <param name="values">The collection of integers to check.</param>
    /// <param name="lowerBound">The lower bound of the range (inclusive). The default value is int.MinValue.</param>
    /// <param name="upperBound">The upper bound of the range (exclusive). The default value is int.MaxValue.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any element in the collection is not within the specified range.</exception>
    public static void AssertAllElementsInRange(ICollection<double> values, double lowerBound = double.MinValue, double upperBound = double.MaxValue)
    {
        if (values.All(value => lowerBound <= value && value < upperBound)) return;
        var message = $"The values must be in range [{lowerBound}, {upperBound})";
        throw new ArgumentOutOfRangeException(nameof(values), message);
    }
}
