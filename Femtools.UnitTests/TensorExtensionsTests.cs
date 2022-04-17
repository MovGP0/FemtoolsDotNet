using NUnit.Framework;

namespace Femtools.UnitTests;

public static class TensorExtensionsTests
{
    [TestOf(nameof(TensorExtensions.Average))]
    public sealed class AverageTests
    {
        [Test]
        public void ShouldCalculateAverages()
        {
            var input = new[,]
            {
                {0d, 1d, 2d, 3d},
                {1d, 1d, 1d, 1d},
                {2d, 2d, 2d, 2d},
                {1d, 2d, 3d, 4d}
            };

            var average = input
                .ToTensor()
                .Average(1)
                .ToVector();

            Console.WriteLine($"[{average[0]},{average[1]},{average[2]},{average[3]}]");

            Assert.Multiple(() =>
            {
                Assert.That(average[0], Is.EqualTo(1.5d));
                Assert.That(average[1], Is.EqualTo(1d));
                Assert.That(average[2], Is.EqualTo(2d));
                Assert.That(average[3], Is.EqualTo(2.5d));
            });
        }

        [Test]
        public void ShouldCalculateAverages2()
        {
            var input = new[,]
            {
                {0d, 1d, 2d },
                {1d, 1d, 1d },
                {2d, 2d, 2d },
                {1d, 2d, 3d }
            };

            var average = input
                .ToTensor()
                .Average()
                .ToVector();
            Console.WriteLine($"[{average[0]},{average[1]},{average[2]}]");

            Assert.Multiple(() =>
            {
                Assert.That(average[0], Is.EqualTo(1d));
                Assert.That(average[1], Is.EqualTo(1.5d));
                Assert.That(average[2], Is.EqualTo(2d));
            });
        }
    }

    [TestOf(nameof(TensorExtensions.GeometricMean))]
    public sealed class GeometricMeanTests
    {
        private static bool AreEqual(
            double left,
            double right,
            double maxDelta = 0.000001d)
        {
            return Math.Abs(left - right) <= maxDelta;
        }

        [Test]
        public void ShouldCalculateGeometricMeanValues()
        {
            var input = new[,]
            {
                {0d, 1d, 2d, 3d},
                {1d, 1d, 1d, 1d},
                {2d, 2d, 2d, 2d},
                {1d, 2d, 3d, 4d}
            };

            var mean = input
                .ToTensor()
                .GeometricMean(1)
                .ToVector();

            Assert.Multiple(() =>
            {
                Assert.That(mean[0], Is.EqualTo(0d), "geom. mean with 0 value should be zero");
                Assert.That(mean[1], Is.EqualTo(1d), "geometric mean of ones should be one");
                Assert.That(mean[2], Is.EqualTo(2d), "geometric mean of twos should be two");
                Assert.That(AreEqual(mean[3], 2.213364d));
            });
        }
    }

    [TestOf(nameof(TensorExtensions.ToMatrix))]
    public sealed class ToMatrixTests
    {
        [Test]
        public void ShouldRevertTensorToMatrix()
        {
            var input = new[,]
            {
                {0d, 1d, 2d, 3d},
                {4d, 5d, 6d, 7d}
            };

            var result = input
                .ToTensor() 
                .ToMatrix();

            Assert.Multiple(() =>
            { // Assert shape
                Assert.That(input.Rank, Is.EqualTo(result.Rank));
                Assert.That(input.GetLength(0), Is.EqualTo(result.GetLength(0)));
                Assert.That(input.GetLength(1), Is.EqualTo(result.GetLength(1)));
            });

            Assert.Multiple(() =>
            { // Assert values
                Assert.That(input[0, 0], Is.EqualTo(result[0, 0]));
                Assert.That(input[0, 1], Is.EqualTo(result[0, 1]));
                Assert.That(input[0, 2], Is.EqualTo(result[0, 2]));
                Assert.That(input[0, 3], Is.EqualTo(result[0, 3]));
                Assert.That(input[1, 0], Is.EqualTo(result[1, 0]));
                Assert.That(input[1, 1], Is.EqualTo(result[1, 1]));
                Assert.That(input[1, 2], Is.EqualTo(result[1, 2]));
                Assert.That(input[1, 3], Is.EqualTo(result[1, 3]));
            });
        }
    }

    [TestOf(nameof(TensorExtensions.Normalize))]
    public sealed class NormalizeTests
    {
        [Test]
        public void ShouldNormalizeTensor()
        {
            var result = new[,]
                {
                    {1d, 1d, 1d, 1d},
                    {2d, 0d, 0d, 0d}
                }
                .ToTensor()
                .Normalize();

            Assert.Multiple(() =>
            {
                Assert.That((float) result[0, 0], Is.EqualTo(0.25d));
                Assert.That((float) result[0, 1], Is.EqualTo(0.25d));
                Assert.That((float) result[0, 2], Is.EqualTo(0.25d));
                Assert.That((float) result[0, 3], Is.EqualTo(0.25d));
                Assert.That((float) result[1, 0], Is.EqualTo(1.00d));
                Assert.That((float) result[1, 1], Is.EqualTo(0.00d));
                Assert.That((float) result[1, 2], Is.EqualTo(0.00d));
                Assert.That((float) result[1, 3], Is.EqualTo(0.00d));
            });
        }

        [Test]
        public void ShouldNormalizeTensor_WithNegativeValues()
        {
            var result = new[,]
                {
                    {0d, -2d, -2d, -2d}
                }
                .ToTensor()
                .Normalize();

            Assert.Multiple(() =>
            {
                Assert.That((float) result[0, 0], Is.EqualTo(1.00d));
                Assert.That((float) result[0, 1], Is.EqualTo(0.00d));
                Assert.That((float) result[0, 2], Is.EqualTo(0.00d));
                Assert.That((float) result[0, 3], Is.EqualTo(0.00d));
            });
        }
    }
}
