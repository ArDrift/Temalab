namespace AdasVetelServer.messages
{
    class ErrorMessage : MessageBase
    {
        public string Message { get; set; }
        public ErrorMessage(string message, string sessionId) : base("ErrorMessage", sessionId)
        {
            Message = message;
        }
    }
}
