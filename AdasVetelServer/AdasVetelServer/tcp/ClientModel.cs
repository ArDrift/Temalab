using AdasVetelServer.model;

namespace AdasVetelServer.tcp
{
    public class ClientModel
    {
        public string IpPort { get; set; }
        public UserData userData { get; set; }
        public string SessionId { get; set; } = "";
        public bool UpToDate { get; set; } = false;

        public ClientModel(string ipPort)
        {
            IpPort = ipPort;
            userData = new UserData();
        }

        public void resetUser()
        {
            userData = new UserData();
            SessionId = "";
        }
    }
}
