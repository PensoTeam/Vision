using System.Windows;
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
                        var faces = detector.GetFacePositions(frame);

                        foreach (var face in faces)
                        {
                            Cv2.Rectangle(frame, new OpenCvSharp.Point(face.Left, face.Top),
                                new OpenCvSharp.Point(face.Right, face.Bottom), Scalar.Aqua, 2, LineTypes.AntiAlias);
                        }

                        WriteableBitmapConverter.ToWriteableBitmap(frame, _wb);
                        Image.Source = _wb;
                    }

                    int c = Cv2.WaitKey(1000 / 33);

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