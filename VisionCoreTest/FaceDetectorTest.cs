using OpenCvSharp;
using VisionCore;
using Xunit;

namespace VisionCoreTest
{
    public class FaceDetectorTest
    {
        private readonly FaceDetector _faceDetector;
        
        public FaceDetectorTest()
        {
            _faceDetector = new FaceDetector();
        }

        ~FaceDetectorTest()
        {
            _faceDetector.Dispose();
        }
        
        [Fact]
        public void TestFaceNum()
        {
            using (var img = Cv2.ImRead("../../../../VisionCoreTest/data/Lenna.png"))
            {
                var faces = _faceDetector.GetFacePositions(img);
                Assert.Single(faces);
            }
        }
    }
}