namespace AdasVetelServer.messages
{
    public class LoginMessage : MessageBase
    {
        public string Username { get; set; } 
        public string Password { get; set; } 
        public int Authority { get; set; }
        public string Info { get; set; } 

        public LoginMessage(string username, string password, int authority, string info, string sessionId) 
            : base("Login", sessionId)
        {
            Username = username;
            Password = password;
            Authority = authority;
            Info = info;
        }

    }
}
