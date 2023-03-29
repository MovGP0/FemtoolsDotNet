using System.Runtime.CompilerServices;
using NUnit.Framework;

namespace Femtools.UnitTests;

[TestOf(typeof(BTS))]
public sealed class BtsTests
{
    [TestOf(nameof(BTS.Calculate))]
    public sealed class CalculateTests
    {
        [Test]
        public void SimplestCase()
        {
            var answers = new[,]
            {
                {1d, 0d}
            };

            var frequencies = new[,]
            {
                {1d, 0d}
            };

            var scores = BTS.Calculate(answers, frequencies);

            const double d = 0.00001d;
            const double expected = 0d;
            Assert.That(scores[0], Is.InRange(expected - d, expected + d));
        }

        [Test]
        public void ShouldCalculateBtsScore()
        {
            var answers = new[,]
            {
                {1.000d, 0.000d, 0.000d},
                {0.000d, 1.000d, 0.000d},
                {0.000d, 0.000d, 1.000d},
                {1d/3d, 1d/3d, 1d/3d}
            };

            var frequencies = new[,]
            {
                { 1d/3d, 1d/3d, 1d/3d },
                { 1.000d, 0.000d, 0.000d},
                { 0.000d, 1.000d, 0.000d},
                { 1.000d, 1.000d, 1.000d}
            };

            var scores = BTS.Calculate(answers, frequencies, 1d);
            
            Console.WriteLine($"[{scores[0]}, {scores[1]}, {scores[2]}, {scores[3]}]");

            const double d = 0.01d;
            Assert.Multiple(() =>
            {
                Assert.That(scores[0], Is.InRange(2.90d - d, 2.90 + d));
                Assert.That(scores[1], Is.InRange(-5.20-d, -5.20+d));
                Assert.That(scores[2], Is.InRange(-1.75-d, 1.75d+d));
                Assert.That(scores[3], Is.InRange(4.05-d, 4.05+d));
            });
        }

        [Test]
        public void ShouldCalculateBtsScore_ValidateExpectedProperties_OfPredictionScore()
        {
            var answers = new[,]
            {
                {1.000d, 0.000d, 0.000d},
                {0.000d, 1.000d, 0.000d},
                {0.000d, 0.000d, 1.000d},
                {1d/3d, 1d/3d, 1d/3d}
            };

            var frequencies = new[,]
            {
                { 3d, 3d, 1d },
                { 1.000d, 0.000d, 0.000d},
                { 0.000d, 1.000d, 0.000d},
                { 1.000d, 1.000d, 1.000d}
            };

            var scores = BTS.Calculate(answers, frequencies, 1d);
            var iScores = BTS.Calculate(answers, frequencies, 0d);
            var pScores = (scores.ToTensor() - iScores.ToTensor()).ToVector();
            
            Console.WriteLine($"[{pScores[0]}, {pScores[1]}, {pScores[2]}, {pScores[3]}]");

            Assert.Multiple(() =>
            {
                Assert.That(pScores[0], Is.InRange(-0.12d, 0.0d));

                Assert.That(pScores[1], Is.LessThan(pScores[0]));
                Assert.That(pScores[1], Is.LessThan(pScores[3]));

                Assert.That(pScores[2], Is.LessThan(pScores[0]));
                Assert.That(pScores[2], Is.EqualTo(pScores[1]));
                Assert.That(pScores[2], Is.LessThan(pScores[3]));

                Assert.That(pScores[3], Is.InRange(-0.0001d, 0.0001d));
            });
        }

        [Test]
        public void ShouldCalculateBtsScore_ValidateExpectedProperties_OfInformationScore()
        {
            var answers = new[,]
            {
                {1.000d, 0.000d, 0.000d},
                {1.000d, 0.000d, 0.000d},
                {1.000d, 0.000d, 0.000d},
                {0.000d, 1.000d, 0.000d}
            };

            var frequencies = new[,]
            {
                { 0.750d, 0.250d, 0.000d },
                { 1.000d, 0.000d, 0.000d},
                { 0.000d, 1.000d, 0.000d},
                { 0.000d, 0.000d, 1.000d}
            };

            var iScores = BTS.Calculate(answers, frequencies, 0d);

            Console.WriteLine($"[{iScores[0]}, {iScores[1]}, {iScores[2]}, {iScores[3]}]");

            Assert.Multiple(() =>
            {
                Assert.That(iScores[0], Is.GreaterThan(iScores[3]));
                Assert.That(iScores[1], Is.GreaterThan(iScores[3]));
                Assert.That(iScores[2], Is.GreaterThan(iScores[3]));
                Assert.That(iScores[1], Is.EqualTo(iScores[2]));
            });
        }
    }
}