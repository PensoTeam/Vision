using System;
using TorchSharp;
using VisionCore;
using Xunit;
using Xunit.Abstractions;

namespace VisionCoreTest
{
    public class SimpleNetTest
    {
        private readonly ITestOutputHelper _testOutputHelper;

        public SimpleNetTest(ITestOutputHelper testOutputHelper)
        {
            _testOutputHelper = testOutputHelper;
        }
        
        [Fact]
        public void TestDevice()
        {
            var cudaAvail = torch.cuda.is_available();
            Assert.IsType<Boolean>(cudaAvail);
        }

        [Fact]
        public void TestPredictUsingCpu()
        {
            var input = torch.tensor(new float[] {1, 2, 3, 4, 5});
            Assert.Equal(15, SimpleNet.PredictUsingCpu(input)[0].item<float>());
        }
        
        [Fact]
        public void TestPredict()
        {   
            var device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;

            var input = torch.tensor(new float[] {1, 2, 2, 3, 3, 4, 4, 5, 5, 6}, 5, 2, device: device);
            var res = SimpleNet.Predict(input);
            
            Assert.Equal(15, res[0].to(torch.CPU).item<float>());
            Assert.Equal(20, res[1].to(torch.CPU).item<float>());
        }
    }
}