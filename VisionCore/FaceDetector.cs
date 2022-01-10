﻿using System;
using System.Runtime.InteropServices;
using DlibDotNet;
using OpenCvSharp;

namespace VisionCore
{
    /// <summary>
    /// Detector of position and landmarks of faces in image.
    /// </summary>
    public class FaceDetector: IDisposable
    {
        private readonly FrontalFaceDetector _mDetector;
        private bool _disposed;

        public FaceDetector()
        {
            _mDetector = Dlib.GetFrontalFaceDetector();
        }

        /// <summary>
        /// Get position of faces in an image.
        /// </summary>
        /// <param name="img">OpenCV Mat array to find face positions.</param>
        /// <returns>Positions of detected faces.</returns>
        public Rectangle[] GetFacePositions(Mat img)
        {
            var array = new byte[img.Width * img.Height * img.ElemSize()];
            Marshal.Copy(img.Data, array, 0, array.Length);

            using (var cimg = Dlib.LoadImageData<BgrPixel>(array, (uint) img.Height, (uint) img.Width,
                       (uint) (img.Width * img.ElemSize())))
            {
                var faces = _mDetector.Operator(cimg);
                return faces;
            }
        }

        public void Dispose()
        {
            if (_disposed == false)
            {
                _mDetector.Dispose();
                _disposed = true;
            }
        }
        

        ~FaceDetector()
        {
            Dispose();
        }
    }
}