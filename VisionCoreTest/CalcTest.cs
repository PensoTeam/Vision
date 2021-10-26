using VisionCore;
using Xunit;

namespace VisionCoreTest
{
    public class CalcTest
    {
        [Fact]
        public void TestAdd()
        {
            Assert.Equal(5, Calc.Add(2, 3));
        }
    }
}