using System.Text;
using NUnit.Framework;
using MathNet.Numerics.LinearAlgebra;

namespace Femtools.UnitTests.DMI;

[TestFixture]
public sealed class DMITests
{
    [Test]
    public void ShouldCalculatePayments()
    {
        var random = new Random(1234);
       
        const int NumberOfAgents = 30;
        const int NumberOfAnswers = 40;
        const int NumberOfChoices = 10;

        var answers = Matrix<double>.Build
            .Dense(NumberOfAgents, NumberOfAnswers, (row, col) => random.Next(0, NumberOfChoices));

        var payments = Femtools.DMI.CalculatePayments(answers, NumberOfChoices, random);
        PrintArray(payments);

        var expected = new []
        { 
            -7.755860510840756E-23, 
            -4.7279296071524795E-22, 
            8.322782461138619E-22, 
            2.1867537473938282E-21, 
            -1.7731078534823193E-21, 
            -2.975764878491723E-22, 
            -1.2082572012437375E-22, 
            -3.152975934674138E-22, 
            -8.015156341964818E-22, 
            -4.702613741983569E-22, 
            2.918995968719001E-22, 
            -1.950088765435926E-22, 
            -8.222286147892314E-22, 
            4.476305250322041E-22, 
            1.0663349268119342E-22, 
            -5.748542547174496E-21, 
            1.109525327084965E-21, 
            -6.987178786620936E-22, 
            -5.279815467834863E-21, 
            -9.155137421995417E-22, 
            -1.1161995097204064E-22, 
            1.9792403677516296E-23, 
            -2.0421464569592592E-22, 
            6.3993137115015224E-21, 
            -5.114188337835801E-21, 
            -2.4347725113062725E-21, 
            1.6301882873923462E-22, 
            -1.4378644268515174E-21, 
            3.2458007736268314E-22, 
            1.6525889923296905E-21, 
        };

        Assert.That(payments.SequenceEqual(expected));
    }

    private static void PrintArray(double[] payments)
    {
        var result = payments.Aggregate(
            new StringBuilder("new double[] { " + Environment.NewLine),
            (sb, val) =>
            {
                sb
                    .Append(val)
                    .Append(", ")
                    .Append(Environment.NewLine);
                return sb;
            },
            sb => sb.Append("};"));
        Console.WriteLine(result);
    }
}
