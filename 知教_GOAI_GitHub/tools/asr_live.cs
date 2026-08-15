// asr_live: Windows WinRT DNN speech recognizer (zh-CN), realtime mic -> text file
// Usage: asr_live.exe <out_txt_path> <stop_flag_path>
// Appends each finalized recognition segment as one UTF-8 line to out file.
// Exits when stop flag file appears, session completes, or 5 min timeout.
using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
using Windows.Globalization;
using Windows.Media.SpeechRecognition;

static class AsrLive
{
    static int Main(string[] args)
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;
        string outPath = args.Length > 0 ? args[0] : Path.Combine(Path.GetTempPath(), "zj_asr_live.txt");
        string stopPath = args.Length > 1 ? args[1] : null;
        var utf8 = new System.Text.UTF8Encoding(false);
        try { File.WriteAllText(outPath, "", utf8); } catch { }
        Console.WriteLine("BOOT");
        try
        {
            var rec = new SpeechRecognizer(new Language("zh-CN"));
            Console.WriteLine("CREATED");
            rec.Timeouts.InitialSilenceTimeout = TimeSpan.FromSeconds(30);
            rec.Timeouts.EndSilenceTimeout = TimeSpan.FromSeconds(1);
            Console.WriteLine("TIMEOUTS_OK");
            var done = new ManualResetEventSlim(false);
            rec.Constraints.Add(new SpeechRecognitionTopicConstraint(SpeechRecognitionScenario.Dictation, "zh-CN"));
            Console.WriteLine("T_CONSTRAINT_OK");
            var cop = rec.CompileConstraintsAsync();
            var csw = System.Diagnostics.Stopwatch.StartNew();
            while (cop.Status == Windows.Foundation.AsyncStatus.Started && csw.Elapsed < TimeSpan.FromSeconds(10)) Thread.Sleep(100);
            if (cop.Status == Windows.Foundation.AsyncStatus.Error)
            {
                Console.WriteLine("FATAL_COMPILE " + (cop.ErrorCode != null ? cop.ErrorCode.Message : "unknown"));
                return 1;
            }
            Console.WriteLine("T_COMPILE_OK");
            Console.WriteLine("T_EVENTS_PRE");
            rec.ContinuousRecognitionSession.ResultGenerated += (s, e) =>
            {
                string t = e.Result != null ? e.Result.Text : null;
                if (!string.IsNullOrWhiteSpace(t))
                {
                    try { File.AppendAllText(outPath, t + "\n", utf8); } catch { }
                    Console.WriteLine("SEG " + t);
                }
            };
            rec.ContinuousRecognitionSession.Completed += (s, e) =>
            {
                Console.WriteLine("SESSION_COMPLETED " + e.Status);
                done.Set();
            };
            Console.WriteLine("T_EVENTS_OK");
            var op = rec.ContinuousRecognitionSession.StartAsync();
            Console.WriteLine("T_START_OK");
            Console.WriteLine("SESSION_STARTED");
            var sw = System.Diagnostics.Stopwatch.StartNew();
            while (!done.IsSet && sw.Elapsed < TimeSpan.FromMinutes(5))
            {
                try
                {
                    if (op.Status == Windows.Foundation.AsyncStatus.Error)
                    {
                        Console.WriteLine("FATAL_START " + op.ErrorCode.Message);
                        return 1;
                    }
                }
                catch { }
                if (stopPath != null && File.Exists(stopPath)) break;
                Thread.Sleep(200);
            }
            try { rec.ContinuousRecognitionSession.StopAsync(); } catch { }
            Thread.Sleep(800);
            Console.WriteLine("BYE");
            return 0;
        }
        catch (Exception ex)
        {
            int hr = 0;
            try { hr = Marshal.GetHRForException(ex); } catch { }
            Console.WriteLine("FATAL hr=0x" + hr.ToString("X8") + " " + ex.GetType().Name + " " + ex.Message);
            return 1;
        }
    }
}
