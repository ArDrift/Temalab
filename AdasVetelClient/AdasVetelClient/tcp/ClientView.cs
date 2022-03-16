using AdasVetelServer.messages;

namespace AdasVetelClient.tcp
{
    public interface ClientView
    {
        void connected();
        void disconnected();
        void handleMessage(MessageBase message, string type);
        bool isActive();
        void clientLoggedIn();
    }
}
