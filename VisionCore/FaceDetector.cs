using System;
using System.IO;
using System.Linq;
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
        /// Get information of faces in an image.
        /// </summary>
        /// <param name="img">Image to find faces in OpenCV Mat type.</param>
        /// <returns>Position and shape of faces in image.</returns>
        public (Rectangle position, FullObjectDetection shape)[] GetFaceInfos(Mat img)
        {
            var array = new byte[img.Width * img.Height * img.ElemSize()];
            Marshal.Copy(img.Data, array, 0, array.Length);

            using (var cimg = Dlib.LoadImageData<BgrPixel>(array, (uint)img.Height, (uint)img.Width, (uint)(img.Width * img.ElemSize())))
            {
                var positions = GetFacePositions(cimg);
                var shapes = PredictFacesShape(cimg, positions);

                return ZipInfos(positions, shapes);
            }
        }

        /// <summary>
        /// Get position of faces in an image.
        /// </summary>
        /// <param name="img">Image to find faces in Dlib Array2D type.</param>
        /// <returns>Position of faces in image.</returns>
        private Rectangle[] GetFacePositions(Array2D<BgrPixel> img)
        {
            return _frontalFaceDetector.Operator(img);
        }

        /// <summary>
        /// Predict shape of detected faces.
        /// </summary>
        /// <param name="img">Image to predict shape of faces in Dlib Array2D type.</param>
        /// <param name="positions">Position of faces to predict shape.</param>
        /// <returns>Shape of faces in image.</returns>
        private FullObjectDetection[] PredictFacesShape(Array2D<BgrPixel> img, Rectangle[] positions)
        {
            var shapes = new FullObjectDetection[positions.Length];

            foreach (var item in positions.Select((position, index) => (position, index)))
            {
                var shape = _shapePredictor.Detect(img, item.position);
                shapes[item.index] = shape;
            }

            return shapes;
        }

        /// <summary>
        /// Zip position and shape of faces into tuple.
        /// </summary>
        /// <param name="positions">Position of faces.</param>
        /// <param name="shapes">Shapes of faces.</param>
        /// <returns>Tuple array that is zipped with position and shape of face.</returns>
        private static (Rectangle, FullObjectDetection)[] ZipInfos(Rectangle[] positions, FullObjectDetection[] shapes)
        {
            var faces = new (Rectangle, FullObjectDetection)[positions.Length];

            for (var i = 0; i < positions.Length; i++)
            {
                faces[i] = (positions[i], shapes[i]);
            }

            return faces;
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