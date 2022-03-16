using System.Collections.Generic;

namespace AdasVetelServer.tcp
{
    public class ClientHolder
    {
        public List<ClientModel> Clients = new List<ClientModel>();

        public void clear()
        {
            Clients.Clear();
        }
        public void addClient(ClientModel client)
        {
            Clients.Add(client);
        }
        public void addClient(string ipPort)
        {
            Clients.Add(new ClientModel(ipPort));
        }
        public void removeClient(ClientModel client)
        {
            Clients.Remove(client);
        }
        public void removeClient(string ipPort)
        {
            foreach (ClientModel item in Clients)
            {
                if (ipPort == item.IpPort)
                {
                    Clients.Remove(item);
                    return;
                }

            }
        }
        public ClientModel getClient(string ipPort)
        {
            foreach (ClientModel item in Clients)
            {
                if (ipPort == item.IpPort)
                    return item;
            }
            return null;
        }
        public bool getClientUpToDate(string ipPort)
        {
            foreach (ClientModel item in Clients)
            {
                if (ipPort == item.IpPort)
                    return item.UpToDate;
            }
            return false;
        }
        public void setClientUpToDate(string ipPort)
        {
            foreach (ClientModel item in Clients)
            {
                if (ipPort == item.IpPort)
                {
                    item.UpToDate = true;
                    return;
                }
            }
        }
        public void somethingChanged()
        {
            foreach (ClientModel item in Clients)
            {
                item.UpToDate = false;
            }
        }

    }
}
