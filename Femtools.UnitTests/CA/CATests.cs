using System.Text;
using NUnit.Framework;

namespace Femtools.UnitTests.CA;

public sealed class CATests
{
    [Test]
    public void ShouldScoreRestaurants()
    {
        var random = new Random(1234);
        var restaurants = new[] { "subway", "burgerK", "McDonald", "KFC", "PizzaHot" };
        var probabilities = new[] { 0.1, 0.2, 0.2, 0, 0.5 };

        const int NumberOfReports = 8;
        const int SizeOfReport = 100;

        var reports = new string[NumberOfReports][];
        for (var x = 0; x < NumberOfReports; x++)
        {
            var report = new string[SizeOfReport];
            for (var i = 0; i < SizeOfReport; i++)
            {
                report[i] = ChooseRandomElementWithProbabilities(random, restaurants, probabilities);
            }

            reports[x] = report;
        }

        var score = Femtools.CA.CalculateScore(reports, agentFirst: false, random);
        
        Assert.That(score.Length, Is.EqualTo(SizeOfReport));
        
        var expected = new[] { 175, 279, 200, 214, 183, 307, 371, 220, 420, 209, 161, 361, 310, 117, 284, 214, 186, 207, 262, 260, 264, 390, 179, 160, 264, 227, 220, 318, 204, 282, 300, 248, 283, 191, 279, 300, 309, 272, 298, 249, 306, 315, 164, 319, 338, 201, 160, 341, 260, 225, 269, 252, 336, 263, 291, 329, 241, 216, 256, 184, 232, 250, 271, 278, 205, 380, 236, 293, 267, 140, 246, 290, 324, 255, 198, 270, 216, 264, 279, 260, 269, 255, 359, 244, 206, 306, 184, 269, 273, 288, 249, 238, 295, 265, 162, 384, 235, 286, 273, 307 };

        var result = score.Aggregate(
            new StringBuilder("new int[] { "),
            (sb, val) =>
            {
                sb.Append(val).Append(", ");
                return sb;
            },
            sb => sb.Append("};"));
        Console.WriteLine(result);
        
        Assert.That(score.SequenceEqual(expected));
    }

    private static string ChooseRandomElementWithProbabilities(
        Random random,
        IReadOnlyList<string> elements,
        IReadOnlyList<double> probabilities)
    {
        var value = random.NextDouble();
        double cumulativeSum = 0;

        for (var i = 0; i < elements.Count; i++)
        {
            cumulativeSum += probabilities[i];

            if (value < cumulativeSum)
            {
                return elements[i];
            }
        }

        return elements.Last();
    }
}
