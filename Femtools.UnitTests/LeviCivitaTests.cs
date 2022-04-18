using FluentAssertions;
using FluentAssertions.Execution;
using NUnit.Framework;
using static TorchSharp.torch;

namespace Femtools.UnitTests;

[TestOf(typeof(LeviCivita))]
public sealed class LeviCivitaTests
{
    [TestOf(nameof(LeviCivita.OfRank))]
    public sealed class OfRankTests
    {
        [Test]
        public void Rank2()
        {
            var ε2 = LeviCivita.OfRank(2);

            using var scope = new AssertionScope();
            ((long)ε2[0, 0]).Should().Be(+0L, "[0,0] has repeating index");
            ((long)ε2[0, 1]).Should().Be(+1L, "[0,1] is odd permutation");
            ((long)ε2[1, 0]).Should().Be(-1L, "[1,0] is even permutation");
            ((long)ε2[1, 1]).Should().Be(+0L, "[1,1] has repeating index");
        }

        [Test]
        public void Rank3()
        {
            var ε3 = LeviCivita.OfRank(3);

            using var scope = new AssertionScope();
            ((long)ε3[0, 0, 0]).Should().Be(+0L, "[0,0,0] has repeating index");
            ((long)ε3[0, 0, 1]).Should().Be(+0L, "[0,0,1] has repeating index");
            ((long)ε3[0, 0, 2]).Should().Be(+0L, "[0,0,2] has repeating index");
            ((long)ε3[0, 1, 0]).Should().Be(+0L, "[0,1,0] has repeating index");
            ((long)ε3[0, 1, 1]).Should().Be(+0L, "[0,1,1] has repeating index");
            ((long)ε3[0, 1, 2]).Should().Be(+1L, "[0,1,2] is even permutation");
            ((long)ε3[0, 2, 0]).Should().Be(+0L, "[0,2,0] has repeating index");
            ((long)ε3[0, 2, 1]).Should().Be(-1L, "[0,2,1] is odd permutation");
            ((long)ε3[0, 2, 2]).Should().Be(+0L, "[0,2,2] has repeating index");

            ((long)ε3[1, 0, 0]).Should().Be(+0L, "[1,0,0] has repeating index");
            ((long)ε3[1, 0, 1]).Should().Be(+0L, "[1,0,1] has repeating index");
            ((long)ε3[1, 0, 2]).Should().Be(-1L, "[1,0,2] is odd permutation");
            ((long)ε3[1, 1, 0]).Should().Be(+0L, "[1,1,0] has repeating index");
            ((long)ε3[1, 1, 1]).Should().Be(+0L, "[1,1,1] has repeating index");
            ((long)ε3[1, 1, 2]).Should().Be(+0L, "[1,1,2] has repeating index");
            ((long)ε3[1, 2, 0]).Should().Be(+1L, "[1,2,0] is even permutation");
            ((long)ε3[1, 2, 1]).Should().Be(+0L, "[1,2,1] has repeating index");
            ((long)ε3[1, 2, 2]).Should().Be(+0L, "[1,2,2] has repeating index");

            ((long)ε3[2, 0, 0]).Should().Be(+0L, "[2,0,0] has repeating index");
            ((long)ε3[2, 0, 1]).Should().Be(+1L, "[2,0,1] is even permutation");
            ((long)ε3[2, 0, 2]).Should().Be(+0L, "[2,0,2] has repeating index");
            ((long)ε3[2, 1, 0]).Should().Be(-1L, "[2,1,0] is odd permutation");
            ((long)ε3[2, 1, 1]).Should().Be(+0L, "[2,1,1] has repeating index");
            ((long)ε3[2, 1, 2]).Should().Be(+0L, "[2,1,2] has repeating index");
            ((long)ε3[2, 2, 0]).Should().Be(+0L, "[2,2,0] has repeating index");
            ((long)ε3[2, 2, 1]).Should().Be(+0L, "[2,2,1] has repeating index");
            ((long)ε3[2, 2, 2]).Should().Be(+0L, "[2,2,2] has repeating index");
        }

        [Test]
        public void Rank4()
        {
            var ε4 = LeviCivita.OfRank(4);

            using var scope = new AssertionScope();
            ((long)ε4[0, 3, 2, 1]).Should().Be(-1L, "[0,3,2,1] is odd permutation");
            ((long)ε4[1, 0, 2, 3]).Should().Be(-1L, "[1,0,2,3] is odd permutation");
            ((long)ε4[3, 2, 1, 0]).Should().Be(+1L, "[3,2,1,0] is even permutation");
            ((long)ε4[2, 1, 3, 2]).Should().Be(+0L, "[2,1,3,2] has repeating index");
        }
    }

    public sealed class DeterminantTests
    {
        [Test]
        public void ShouldCalculateDeterminantOfMatrix()
        {
            var ε = LeviCivita.OfRank(3);
            var matrix = tensor(new[] { 3L, 2L, 1L, 5L, 7L, 4L, 9L, 6L, 8L }, new[] { 3L, 3L }, ScalarType.Int64);
            var det = einsum("ijk,i,j,k->", ε, matrix[0], matrix[1], matrix[2]);
            var result = (int)det;
            result.Should().Be(55);
        }
    }

    public sealed class CrossProductTests
    {
        [Test]
        public void ShouldCalculateCrossProductOfTwoVectors()
        {
            var ε = LeviCivita.OfRank(3);
            var a = tensor(new[] { 3L, -3L, 1L }, new[] { 3L }, ScalarType.Int64);
            var b = tensor(new[] { 4L, 9L, 2L }, new[] { 3L }, ScalarType.Int64);
            var expected = tensor(new[] { -15L, -2L, 39L }, new[] { 3L }, ScalarType.Int64);

            var result = einsum("ijk,j,k->i", ε, a, b);

            using var scope = new AssertionScope();
            ((long)result[0]).Should().Be((long)expected[0]);
            ((long)result[1]).Should().Be((long)expected[1]);
            ((long)result[2]).Should().Be((long)expected[2]);
        }
    }
}