using AdasVetelServer.logger;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Windows.Forms;

namespace AdasVetelServer.bert
{
    class NerExecuter
    {
        readonly System.Diagnostics.ProcessStartInfo processStartInfo = null;
        public List<NerResult> Results = new List<NerResult>();
        private string text;
        private Thread thread;
        public NerExecuter(string text)
        {
            thread = new Thread(execute);

            this.text = text;
            string pathFile = Application.StartupPath + "\\config\\pythonPath.loc";

            if (!File.Exists(pathFile))
            {
                ServerLogger.Instance.writeLog("Unknown python path");
                return;
            }
            string pythonPath = File.ReadAllText(pathFile);

            processStartInfo = new System.Diagnostics.ProcessStartInfo();
            processStartInfo.FileName = pythonPath;
            processStartInfo.UseShellExecute = false;
            processStartInfo.CreateNoWindow = true;
            processStartInfo.RedirectStandardOutput = true;
            processStartInfo.RedirectStandardError = true;
            processStartInfo.LoadUserProfile = true;
        }

        private void execute() {
            lock (Results) {
                ServerLogger.Instance.writeLog($"{DateTime.Now} NER prediction started.");
                Results = predict();
                formatResults();
                ServerLogger.Instance.writeLog($"{DateTime.Now} NER prediction ended.");
            }
        }
        public void start() {
            thread.Start();
        }

        public void waitForExecutionEnd() {
            ServerLogger.Instance.writeLog($"{DateTime.Now} Waiting for NER execution to end...");
            if (thread.ThreadState == ThreadState.Running) {
                thread.Join();            
            }
        }

        private List<NerResult> predict()
        {
            if (processStartInfo == null)
            {
                ServerLogger.Instance.writeLog("Unknown python path");
                return new List<NerResult>();
            }
            string startup = Application.StartupPath;
            ServerLogger.Instance.writeLog($"{startup}\\pyScripts\\ner_controller.py");
            processStartInfo.Arguments = string.Format("\"{0}\" {1} \"{2}\" \"{3}\" \"{4}\"",               
                $"{startup}\\pyScripts\\ner_controller.py",
                "predict",
                $"{startup}\\pyScripts\\model_save",
                text,
                $"{startup}\\pyScripts\\script_output.json");

            using (System.Diagnostics.Process process = System.Diagnostics.Process.Start(processStartInfo))
            {
                string stderr = process.StandardError.ReadToEnd();
                if (!stderr.Equals(""))
                {
                    ServerLogger.Instance.writeLog(stderr);
                    return new List<NerResult>();
                }
                string result = File.ReadAllText($"{startup}\\pyScripts\\script_output.json", Encoding.UTF8);                
                return JsonSerializer.Deserialize<List<NerResult>>(result);

            }
        }
        public static List<NerResult> getResultsByTag(List<NerResult> result, string tags)
        {
            List<NerResult> filtered = new List<NerResult>();
            foreach (string tag in tags.Split(' '))
            {
                filtered.AddRange(result.FindAll(r => r.entity.Equals(tag)));
            }
            return filtered;
        }
        private void formatResults()
        {            
            for (int i = 0; i < Results.Count - 1; i++)
            {
                string firstEntity = Results[i].entity.Split('-')[0];
                string secondEntity = Results[i + 1].entity.Split('-')[0];
                if (firstEntity == secondEntity && Results[i + 1].start == Results[i].end)
                {
                    NerResult newResult = new NerResult();
                    newResult.start = Results[i].start;
                    newResult.end = Results[i + 1].end;
                    newResult.score = (Results[i].score + Results[i + 1].score) / 2;
                    newResult.word = Results[i].word + Results[i + 1].word.Replace("#", "");
                    newResult.entity = Results[i].entity;
                    Results[i] = newResult;
                    Results.RemoveAt(i + 1);
                    i--;
                }               
            }
            foreach (NerResult res in Results)
            {
                res.end--;
                string[] splitEntity = res.entity.Split('-');
                if (splitEntity.Length > 1) {
                    res.entity = splitEntity[1];
                }
                if (res.score <= 0.4) {
                    res.certancy = NerResult.Certancy.Uncertain;
                }
                if (res.score > 0.4) {
                    res.certancy = NerResult.Certancy.Medium;
                }
                if (res.score >= 0.8) {
                    res.certancy = NerResult.Certancy.Certain;
                }
            }
        }
    }
}
