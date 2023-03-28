namespace Femtools.MIG;

public static class MathExtensions
{
    private static Dictionary<int, long> cache = new();

    public static long Factorial(this int n)
    {
        if (n <= 1)
        {
            return 1;
        }

        if (cache.ContainsKey(n))
        {
            return cache[n];
        }

        var result = n * Factorial(n - 1);
        cache[n] = result;
        return result;
    }
}