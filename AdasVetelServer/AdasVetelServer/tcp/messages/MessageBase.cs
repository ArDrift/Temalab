namespace AdasVetelServer.messages
{
    public class MessageBase
    {
        public string Type { get; set; }
        public string SessionId { get; set; }
        public MessageBase(string type, string sessionId)
        {
            Type = type;
            SessionId = sessionId;
        }
    }
}
