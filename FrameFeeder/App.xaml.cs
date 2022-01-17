using System.Diagnostics;
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
}