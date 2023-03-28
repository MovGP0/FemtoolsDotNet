namespace Femtools.MIG
{
    public sealed class MIGAlgorithm
    {
        private struct Functions
        {
            internal Func<double, double> F { get; }
            internal Func<double, double> FPrime { get; }
            internal Func<double, double> FStar { get; }

            public Functions(Func<double, double> f, Func<double, double> fPrime, Func<double, double> fStar)
            {
                F = f;
                FPrime = fPrime;
                FStar = fStar;
            }
        }

        public static double[,] Calculate(double[,] answers, MIGFunction function, double prior)
        {
            var agent_n = answers.GetLength(0);
            var task_n = answers.GetLength(1);

            if (agent_n != 2) throw new ArgumentException("Invalid number of agents.");

            var fs = function == MIGFunction.KLD ? KLDs() : TVDs();

            double reward = 0;
            double penalty = 0;

            for (var i = 0; i < task_n; i++)
            for (var j = 0; j < task_n; j++)
            {
                if (i == j)
                    reward += R(fs.FPrime, answers[0, i], answers[1, i], prior);
                else
                    penalty += R(fs.FStar, answers[0, i], answers[1, j], prior);
            }

            var payment = reward / task_n - penalty / (task_n * (task_n - 1));
            return new [,] { { payment, payment } };
        }

        private static double R(Func<double, double> f, double a, double b, double p)
            => f(a * b / p + (1 - a) * (1 - b) / (1 - p));

        private static Functions TVDs() => new(TVD, TVDPrime, TVDStar);
        private static double TVD(double t) => Math.Abs(t - 1);
        private static double TVDPrime(double x) => x > 1d ? 1d : Math.Abs(x - 1d) < 0.0001 ? 0d : -1d;
        private static double TVDStar(double x) => TVDPrime(x);

        private static Functions KLDs() => new(KLD, KLDPrime, KLDStar);
        private static double KLD(double t) => t * Math.Log(t);
        private static double KLDPrime(double t) => 1 + Math.Log(t);
        private static double KLDStar(double t) => t;
    }
}