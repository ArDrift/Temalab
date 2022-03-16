using SimpleTcp;

namespace AdasVetelClient.tcp
{
    public class TcpClient : SimpleTcpClient
    {
        public enum Authority
        {
            NONE, LOW, HIGH
        };
        public TcpClient(string port) : base(port) { }
        private bool stopped = false;
        public bool Stopped
        {
            get
            {
                bool temp = stopped;
                stopped = false;
                return temp;
            }
            set
            {
                stopped = value;
            }
        }
        public string Username { get; set; } = "";
        public Authority Auth { get; set; } = Authority.NONE;
        public string SessionId { get; set; } = "";
        public void reset()
        {
            Username = "";
            Auth = Authority.NONE;
            SessionId = "";
        }
    }
}
