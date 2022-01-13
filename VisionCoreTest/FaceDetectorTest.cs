using OpenCvSharp;
using VisionCore;
using Xunit;
using Xunit.Abstractions;

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
            var images = new string[] {"Denis_Mukwege.jpg", "dog_backpack.jpg", "gorilla.jpg"};
            
            foreach (var image in images)
            {
                using (var img = Cv2.ImRead("../../../../VisionCoreTest/data/" + image))
                {
                    var faces = _faceDetector.GetFacePositions(img);

                    if (image == "Denis_Mukwege.jpg")
                    {
                        Assert.Single(faces);
                    }

                    else
                    {
                        Assert.Empty(faces);
                    }
                }
            }
        }
    }
}