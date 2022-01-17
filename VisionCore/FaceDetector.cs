using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using DlibDotNet;
using OpenCvSharp;

namespace VisionCore
{
    /// <summary>
    /// Detector of position and landmarks of faces in image.
    /// </summary>
    public class FaceDetector : IDisposable
    {
        private readonly FrontalFaceDetector _frontalFaceDetector;
        private readonly ShapePredictor _shapePredictor;
        private bool _disposed;

        public FaceDetector()
        {
            _frontalFaceDetector = Dlib.GetFrontalFaceDetector();
            var path = Path.GetFullPath("./shape_predictor_68_face_landmarks.dat");
            _shapePredictor = ShapePredictor.Deserialize(path);
        }

        /// <summary>
        /// Get position of faces in an image.
        /// </summary>
        /// <param name="img">Dlib 2D array to find face positions.</param>
        /// <returns>Positions of detected faces.</returns>
        private Rectangle[] GetFacePositions(Array2D<BgrPixel> img)
        {
            var faces = _frontalFaceDetector.Operator(img);
            return faces;
        }

        public List<Tuple<Rectangle, FullObjectDetection>> GetFaceInfos(Mat img)
        {
            var faces = new List<Tuple<Rectangle, FullObjectDetection>>();

            var array = new byte[img.Width * img.Height * img.ElemSize()];
            Marshal.Copy(img.Data, array, 0, array.Length);

            using (var cimg = Dlib.LoadImageData<BgrPixel>(array, (uint)img.Height, (uint)img.Width, (uint)(img.Width * img.ElemSize())))
            {
                var facePositions = GetFacePositions(cimg);

                foreach (var position in facePositions)
                {
                    var predicted = _shapePredictor.Detect(cimg, position);
                    faces.Add(new Tuple<Rectangle, FullObjectDetection>(position, predicted));
                }

                return faces;
            }
        }

        public void Dispose()
        {
            if (_disposed == false)
            {
                _frontalFaceDetector.Dispose();
                _shapePredictor.Dispose();
                _disposed = true;
            }
        }

        ~FaceDetector()
        {
            Dispose();
        }
    }
}