using DlibDotNet;
using Point = OpenCvSharp.Point;

namespace VisionCore
{
    public struct FaceInfo
    {
        public Rectangle Position { get; }
        public FullObjectDetection Shape { get; }
        public (Point left, Point right) Eyeballs { get; }

        public FaceInfo(Rectangle position, FullObjectDetection shape, (Point, Point) eyeballs)
        {
            Position = position;
            Shape = shape;
            Eyeballs = eyeballs;
        }
    }
}