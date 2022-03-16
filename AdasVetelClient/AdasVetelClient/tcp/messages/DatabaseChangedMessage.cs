namespace AdasVetelServer.messages
{
    class DatabaseChangedMessage : MessageBase
    {

        public DatabaseChangedMessage(string sessionId) : base("DataChanged", sessionId)
        { }
    }
}
