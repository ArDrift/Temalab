using AdasVetelServer.messages;
using AdasVetelServer.tcp;

namespace AdasVetelServer.logic.analyzing
{
    class LoadingTask
    {
        public LoadMessage Contract { get; set; }
        public ClientModel Client { get; set; }
        public LoadingTask(LoadMessage contract, ClientModel client)
        {
            Contract = contract;
            Client = client;
        }
        public void executeTask()
        {
            TextAnalyzer.Instance.analyzeText(Contract.Data, Contract.FileName);
        }
    }
}
