using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

using OpenCvSharp;
using OpenCvSharp.Extensions;
using OpenCvSharp.WpfExtensions;

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

        private void ButtonClick(object sender, RoutedEventArgs e)
        {
            Mat frame = new Mat();
            _loop = true;

            while (_loop)
            {
                if (_cap.Read(frame))
                {
                    Cv2.Flip(frame, frame, FlipMode.Y);
                    WriteableBitmapConverter.ToWriteableBitmap(frame, _wb);
                    Image.Source = _wb;
                }

                int c = Cv2.WaitKey(33);
                
                if (c != -1)
                    break;
            }
        }
    }
}