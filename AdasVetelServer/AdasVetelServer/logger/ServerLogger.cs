using System;
using System.Collections.Generic;
using System.Windows.Forms;

namespace AdasVetelServer.logger
{
    class ServerLogger
    {
        public static ServerLogger Instance { get; } = new ServerLogger();
        public TextBox output { get; set; }
        public bool Verbose { get; set; } = true;
        public void writeLog(string text, bool debugVerbose = true)
        {
            if (!(Verbose && debugVerbose && output != null))
                return;

            output.Invoke((MethodInvoker)delegate
            {
                output.AppendText(text + Environment.NewLine);
            });
            Console.WriteLine(text);
        }
        public void writeLog(bool text, bool debugVerbose = true)
        {
            if (!(Verbose && debugVerbose && output != null))
                return;

            output.Invoke((MethodInvoker)delegate
            {
                output.AppendText(text + Environment.NewLine);
            });
            Console.WriteLine(text);
        }
        public void writeLog(int text, bool debugVerbose = true)
        {
            if (!(Verbose && debugVerbose && output != null))
                return;

            output.Invoke((MethodInvoker)delegate
            {
                output.AppendText(text + Environment.NewLine);
            });
            Console.WriteLine(text);
        }
        public void writeLog(double text, bool debugVerbose = true)
        {
            if (!(Verbose && debugVerbose && output != null))
                return;

            output.Invoke((MethodInvoker)delegate
            {
                output.AppendText(text + Environment.NewLine);
            });

            Console.WriteLine(text);
        }
        public void writeLog(bool debugVerbose = true)
        {
            if (!(Verbose && debugVerbose && output != null))
                return;

            output.Invoke((MethodInvoker)delegate
            {
                output.AppendText(Environment.NewLine);
            });
            Console.WriteLine();
        }
        public void writeLog<T>(List<T> list, bool debugVerbose = true)
        {
            if (!(Verbose && debugVerbose && output != null))
                return;

            foreach (var element in list)
            {
                output.Invoke((MethodInvoker)delegate
                {
                    output.AppendText(element.ToString() + Environment.NewLine);
                });
                Console.WriteLine(element.ToString());
            }
        }

        public void writeLog<T>(Dictionary<string, List<T>> listDir, bool debugVerbose = true)
        {
            if (!(Verbose && debugVerbose && output != null))
                return;
            foreach (var list in listDir) {
                output.Invoke((MethodInvoker)delegate
                {
                    output.AppendText($"{list.Key}:{{{Environment.NewLine}");
                    Console.WriteLine($"{list.Key}:{{{Environment.NewLine}");
                    foreach (var element in list.Value)
                    {
                            output.AppendText($"\t{element}{Environment.NewLine}");
                        Console.WriteLine($"\t{element}{Environment.NewLine}");
                    }
                    output.AppendText($"}}{Environment.NewLine}");
                    Console.WriteLine($"}}{Environment.NewLine}");
                });
            }
        }
    }
}
