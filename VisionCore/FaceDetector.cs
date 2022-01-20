using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using DlibDotNet;
using OpenCvSharp;
using Point = OpenCvSharp.Point;

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

            Cv2.NamedWindow("mask");
            Cv2.CreateTrackbar("threshold", "mask", 255);
        }

        /// <summary>
        /// Get information of faces in an image.
        /// </summary>
        /// <param name="img">Image to find faces in OpenCV Mat type.</param>
        /// <returns>Position and shape of faces in image and position of eyeballs in faces.</returns>
        public (Rectangle position, FullObjectDetection shape, (Point left, Point right) eyeballs)[] GetFaceInfos(Mat img)
        {
            var array = new byte[img.Width * img.Height * img.ElemSize()];
            Marshal.Copy(img.Data, array, 0, array.Length);

            using (var cimg = Dlib.LoadImageData<BgrPixel>(array, (uint)img.Height, (uint)img.Width, (uint)(img.Width * img.ElemSize())))
            {
                var positions = GetFacePositions(cimg);
                var shapes = PredictFacesShape(cimg, positions);
                var eyeballs = GetEyeballsPosition(img, shapes);

                return ZipInfos(positions, shapes, eyeballs);
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

        private (Point left, Point right)[] GetEyeballsPosition(Mat img, FullObjectDetection[] shapes)
        {
            var eyeballs = new (Point, Point)[shapes.Length];

            foreach (var item in shapes.Select((shape, index) => (shape, index)))
            {
                var shape = item.shape;

                var eyes = MaskOnEyes(img, shape);

                var middleOfEyes = (shape.GetPart(41).X + shape.GetPart(44).X) / 2;
                var roi = new Rect(0, 0, middleOfEyes, img.Height);

                var left = ContourEyeball(eyes.SubMat(roi), roi.TopLeft);

                roi.X = roi.Width;
                roi.Width = img.Width - roi.Width;

                var right = ContourEyeball(eyes.SubMat(roi), roi.TopLeft);

                eyeballs[item.index] = (left, right);

                Cv2.ImShow("mask", eyes);
            }

            return eyeballs;
        }

        private Mat MaskOnEyes(Mat img, FullObjectDetection shape)
        {
            var eyes = Mat.Zeros(img.Size(), img.Type()).ToMat();
            var mask = Mat.Zeros(img.Size(), MatType.CV_8U).ToMat();
            var keypointIndices = new[] { new uint[] { 36, 37, 38, 39, 40, 41 }, new uint[] { 42, 43, 44, 45, 46, 47 } };

            foreach (var side in keypointIndices)
            {
                var points = new Point[side.Length];

                foreach (var item in side.Select((value, index) => (value, index)))
                {
                    var dlibPoint = shape.GetPart(item.value);
                    points[item.index] = new Point(dlibPoint.X, dlibPoint.Y);
                }

                Cv2.FillConvexPoly(mask, points, Scalar.White);
            }

            Cv2.BitwiseAnd(img, img, eyes, mask);
            Cv2.BitwiseNot(eyes, eyes, ~mask);
            Cv2.CvtColor(eyes, eyes, ColorConversionCodes.BGR2GRAY);

            var threshold = Cv2.GetTrackbarPos("threshold", "mask");
            Cv2.Threshold(eyes, eyes, threshold, 255, ThresholdTypes.Binary);

            Cv2.Erode(eyes, eyes, new Mat(), iterations: 2);
            Cv2.Dilate(eyes, eyes, new Mat(), iterations: 4);
            Cv2.MedianBlur(eyes, eyes, 3);
            eyes = ~eyes;

            return eyes;
        }

        private Point ContourEyeball(Mat eyes, Point roiOffset)
        {
            Cv2.FindContours(eyes, out var contours, out var _, RetrievalModes.External, ContourApproximationModes.ApproxNone);

            if (contours.Length > 0)
            {
                var contour = contours.Aggregate((max, candidate) => Cv2.ContourArea(candidate) > Cv2.ContourArea(max) ? candidate : max);

                var m = Cv2.Moments(contour);
                var cx = (int)(m.M10 / m.M00);
                var cy = (int)(m.M01 / m.M00);

                return new Point(cx, cy) + roiOffset;
            }

            return new Point();
        }

        /// <summary>
        /// Zip position and shape of faces into tuple.
        /// </summary>
        /// <param name="positions">Position of faces.</param>
        /// <param name="shapes">Shapes of faces.</param>
        /// <param name="eyeballs">Position of eyeballs.</param>
        /// <returns>Tuple array that is zipped with position and shape of face.</returns>
        private (Rectangle, FullObjectDetection, (Point, Point))[] ZipInfos(Rectangle[] positions, FullObjectDetection[] shapes, (Point left, Point right)[] eyeballs)
        {
            var faces = new (Rectangle, FullObjectDetection, (Point, Point))[positions.Length];

            for (var i = 0; i < positions.Length; i++)
            {
                faces[i] = (positions[i], shapes[i], eyeballs[i]);
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
                Cv2.DestroyAllWindows();
            }
        }

        ~FaceDetector()
        {
            Dispose();
        }
    }
}