using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Windows;

namespace FrameFeeder
{
    /// <summary>
    /// Interaction logic for App.xaml
    /// </summary>
    public partial class App : Application
    {
        public App()
        {
#if DEBUG
            if (Debugger.IsAttached == false)
            {
                NativeMethods.AllocConsole();
            }
#endif
        }
    }

    static class NativeMethods
    {
        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        public static extern bool AllocConsole();
    }
}