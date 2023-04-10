namespace Femtools;

internal static class MathExtensions
{
    private static readonly Dictionary<int, long> cache = new();

    public static long Factorial(this int n)
    {
        if (n <= 1) return 1;
        if (cache.TryGetValue(n, out var factorial)) return factorial;

        var result = n * Factorial(n - 1);
        cache[n] = result;
        return result;
    }

    /// <summary>
    /// Calculates the number of possible combinations of k items that can be selected from a set of
    /// n distinct items, where order does not matter.
    /// The formula for the comb function is given by:
    /// C(n, k) = n! / (k! * (n-k)!)
    /// </summary>
    /// <param name="n"></param>
    /// <param name="m"></param>
    /// <returns></returns>
    public static double Comb(this int n, int m)
        => n.Factorial() / (m.Factorial() * (double)(n - m).Factorial());
}