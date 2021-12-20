using TorchSharp;

namespace VisionCore
{
    public class SimpleNet
    {
        public static torch.Tensor PredictUsingCpu(torch.Tensor input)
        {
            var w = torch.ones(1, 5);
            return w.matmul(input);
        }
        public static torch.Tensor Predict(torch.Tensor input)
        {
            var device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;

            var w = torch.ones(1,5, device: device);

            return w.matmul(input).reshape(2);
        }
    }
}