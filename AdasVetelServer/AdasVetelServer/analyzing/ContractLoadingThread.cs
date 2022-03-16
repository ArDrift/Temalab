using AdasVetelServer.logger;
using AdasVetelServer.messages;
using AdasVetelServer.tcp;
using System;
using System.Collections.Generic;
using System.Threading;


namespace AdasVetelServer.logic.analyzing
{
    class ContractLoadingThread
    {
        private readonly ManualResetEvent waitForTask = new ManualResetEvent(false);
        private readonly List<LoadingTask> tasks = new List<LoadingTask>();
        private bool endThread = false;
        private Thread thread;
        public ContractLoadingThread() {
            thread = new Thread(startListening);
            thread.IsBackground = true;
        }
        private void startListening()
        {
            while (true)
            {
                waitForTask.WaitOne();
                while (tasks.Count > 0)
                {
                    ProgressInfoMessage progress = new ProgressInfoMessage("OK", $"{tasks[0].Contract.FileName}.", countTasksForClient(tasks[0].Client), "");
                    TcpServerWrapper.Instance.send<ProgressInfoMessage>(tasks[0].Client, progress);
                    ServerLogger.Instance.writeLog(progress.Info);
                    try
                    {
                        tasks[0].executeTask();
                    }
                    catch (Exception e)
                    {
                        ServerLogger.Instance.writeLog(e.StackTrace);
                        ProgressInfoMessage error = new ProgressInfoMessage("ERROR", "A feldolgozás megszakadt.", countTasksForClient(tasks[0].Client), "");
                        TcpServerWrapper.Instance.send<ProgressInfoMessage>(tasks[0].Client, error);
                        tasks.RemoveAt(0);
                        continue;
                    }
                    LoadMessage resp = new LoadMessage("", tasks[0].Contract.FileName, "");
                    TcpServerWrapper.Instance.send<LoadMessage>(tasks[0].Client, resp);
                    TcpServerWrapper.Instance.sendToAll<DatabaseChangedMessage>(new DatabaseChangedMessage(""));
                    ServerLogger.Instance.writeLog($"{resp.FileName} analyzed succesfully.");
                    tasks.RemoveAt(0);
                }
                waitForTask.Reset();
                if (endThread)
                    break;

            }

        }
        private int countTasksForClient(ClientModel client)
        {
            return tasks.FindAll(t => t.Client == client).Count;
        }
        public void addTask(LoadMessage contract, ClientModel client)
        {
            tasks.Add(new LoadingTask(contract, client));
            waitForTask.Set();
        }
        public void stopThread()
        {
            endThread = true;
            waitForTask.Set();
        }
        public void start() {
            thread.Start();
        }
    }
}
