// ocr: Windows WinRT offline OCR (zh-Hans-CN preferred), image file -> text file
// Usage: ocr.exe <image_path> [out_txt_path]   (out default: <image_path>.txt)
// Writes recognized text (UTF-8, lines joined by \n). Exit 0 = ok, 3 = no OCR engine.
using System;
using System.IO;
using System.Linq;
using System.Threading;
using Windows.Foundation;
using Windows.Globalization;
using Windows.Graphics.Imaging;
using Windows.Media.Ocr;
using Windows.Storage;

static class OcrTool
{
    static T WaitFor<T>(IAsyncOperation<T> op, int timeoutMs)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        while (op.Status == AsyncStatus.Started && sw.ElapsedMilliseconds < timeoutMs) Thread.Sleep(50);
        if (op.Status == AsyncStatus.Error)
            throw new Exception(op.ErrorCode != null ? op.ErrorCode.Message : "async error");
        return op.GetResults();
    }

    static int Main(string[] args)
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;
        if (args.Length < 1) { Console.WriteLine("ERR usage: ocr.exe <image> [out]"); return 2; }
        string img = args[0];
        string outp = args.Length > 1 ? args[1] : img + ".txt";
        var utf8 = new System.Text.UTF8Encoding(false);
        try
        {
            var file = WaitFor(StorageFile.GetFileFromPathAsync(img), 15000);
            var stream = WaitFor(file.OpenAsync(FileAccessMode.Read), 15000);
            var decoder = WaitFor(BitmapDecoder.CreateAsync(stream), 20000);
            var bmp = WaitFor(decoder.GetSoftwareBitmapAsync(), 20000);

            OcrEngine eng = null;
            try
            {
                if (OcrEngine.IsLanguageSupported(new Language("zh-Hans-CN")))
                    eng = OcrEngine.TryCreateFromLanguage(new Language("zh-Hans-CN"));
            }
            catch { }
            if (eng == null) eng = OcrEngine.TryCreateFromUserProfileLanguages();
            if (eng == null) { Console.WriteLine("ERR no_ocr_engine"); return 3; }
            Console.WriteLine("LANG " + (eng.RecognizerLanguage != null ? eng.RecognizerLanguage.LanguageTag : "?"));

            var res = WaitFor(eng.RecognizeAsync(bmp), 30000);
            string text = string.Join("\n", res.Lines.Select(l => l.Text));
            File.WriteAllText(outp, text, utf8);
            Console.WriteLine("LINES " + res.Lines.Count + " CHARS " + text.Length);
            return 0;
        }
        catch (Exception ex)
        {
            Console.WriteLine("ERR " + ex.Message);
            return 1;
        }
    }
}
