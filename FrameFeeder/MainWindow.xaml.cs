﻿using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using OpenCvSharp;
using OpenCvSharp.WpfExtensions;
using VisionCore;

namespace FrameFeeder
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : System.Windows.Window
    {
        private VideoCapture _cap;
        private WriteableBitmap _wb;
        private const int FrameWidth = 1920;
        private const int FrameHeight = 1080;
        private bool _loop;

        public MainWindow()
        {
            InitializeComponent();
        }

        private bool InitWebCam()
        {
            try
            {
                _cap = VideoCapture.FromCamera(0, 0);
                _cap.FrameWidth = FrameWidth;
                _cap.FrameHeight = FrameHeight;
                _cap.Open(0);

                _wb = new WriteableBitmap(_cap.FrameWidth, _cap.FrameHeight, 96, 96, PixelFormats.Bgr24, null);
                Image.Source = _wb;

                return true;
            }
            catch
            {
                return false;
            }
        }

        private void WindowLoaded(object sender, RoutedEventArgs e)
        {
            if (InitWebCam())
            {
                MessageBox.Show("Camera on");
            }
            else
            {
                MessageBox.Show("Error opening camera");
            }
        }

        private void WindowClosing(object sender, System.ComponentModel.CancelEventArgs e)
        {
            _loop = false;

            if (_cap.IsOpened())
            {
                _cap.Dispose();
            }
        }

        private void StartBtnClick(object sender, RoutedEventArgs e)
        {
            _loop = true;

            using (var frame = new Mat())
            using (var detector = new FaceDetector())
            {
                while (_loop)
                {
                    if (_cap.Read(frame))
                    {
                        Cv2.Flip(frame, frame, FlipMode.Y);
                        var infos = detector.GetFaceInfos(frame);

                        foreach (var info in infos)
                        {
                            var position = info.position;
                            Cv2.Rectangle(frame, new OpenCvSharp.Point(position.Left, position.Top), new OpenCvSharp.Point(position.Right, position.Bottom), Scalar.Aqua, 2, LineTypes.AntiAlias);

                            var shape = info.shape;
                            for (uint i = 0; i < shape.Parts; i++)
                            {
                                var part = shape.GetPart(i);
                                Cv2.Circle(frame, part.X, part.Y, 3, Scalar.LightGreen);
                            }

                            var eyeballs = info.eyeballs;

                            Cv2.Circle(frame, eyeballs.left, 3, Scalar.Red, 2);
                            Cv2.Circle(frame, eyeballs.right, 3, Scalar.Red, 2);
                        }

                        WriteableBitmapConverter.ToWriteableBitmap(frame, _wb);
                        Image.Source = _wb;
                    }

                    int c = Cv2.WaitKey(1000 / 30);

                    if (c != -1)
                    {
                        break;
                    }
                }
            }
        }

        private void StopBtnClick(object sender, RoutedEventArgs e)
        {
            _loop = false;
        }
    }
}