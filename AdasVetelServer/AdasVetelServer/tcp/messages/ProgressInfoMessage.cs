namespace AdasVetelServer.messages
{
    class ProgressInfoMessage : MessageBase
    {
        public string Info { get; set; }
        public int NumberOfTasks { get; set; }
        public string ProgressType { get; set; }
        public ProgressInfoMessage(string progressType, string info, int numberOfTasks, string sessionId)
            : base("ProgressInfo", sessionId)
        {
            ProgressType = progressType;
            Info = info;
            NumberOfTasks = numberOfTasks;
        }
    }
}
