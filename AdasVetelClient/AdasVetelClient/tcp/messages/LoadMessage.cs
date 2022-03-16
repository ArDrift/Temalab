namespace AdasVetelServer.messages
{
    class LoadMessage : MessageBase
    {
        public string Data { get; set; } = "";
        public string FileName { get; set; } = "";

        public LoadMessage(string data, string fileName, string sessionId) : base("LoadContract", sessionId)
        {
            Data = data;
            FileName = fileName;
        }

    }
}
