using NUnit.Framework;

namespace Femtools.UnitTests.MIG;

[TestOf(typeof(MIGAlgorithm))]
public class MIGAlgorithmTests
{
    [Test]
    public void ShouldCalculateMIG()
    {
        var data = new[,]
        {
            { 0.2, 0.3, 0.2 },
            { 0.3, 0.5, 0.3 }
        };
        
        var result = MIGAlgorithm.Calculate(data, MIGFunction.TVD, 0.3);

        Assert.Multiple(() =>
        {
            Assert.That(result[0, 0], Is.EqualTo(1d/3d));
            Assert.That(result[0, 1], Is.EqualTo(1d/3d));
        });
    }
}
