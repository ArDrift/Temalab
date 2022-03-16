using AdasVetelServer.db;
using AdasVetelServer.dbHandler;
using AdasVetelServer.logger;
using AdasVetelServer.logic.analyzing;
using AdasVetelServer.messages;
using AdasVetelServer.model;
using SimpleTcp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Threading;
namespace AdasVetelServer.tcp
{
    public class TcpServerWrapper
    {
        public static TcpServerWrapper Instance { get; } = new TcpServerWrapper();
        private readonly JsonSerializerOptions sOption = new JsonSerializerOptions { WriteIndented = false };
        private readonly ClientHolder clients = new ClientHolder();
        private SimpleTcpServer server;
        private ContractLoadingThread contractLoadingThread;
        private int currentMessageSize = 0;
        private MemoryStream currentMessageStream = new MemoryStream();
        private BinaryReader br;
        long readPos = 0;
        readonly object lockObject = new object();

        private TcpServerWrapper()
        {
            br = new BinaryReader(currentMessageStream);
        }

        public void stopServer()
        {
            contractLoadingThread.stopThread();
            foreach (ClientModel item in clients.Clients)
                server.DisconnectClient(item.IpPort);
            clients.clear();
            server.Stop();
            server.Dispose();
            while (server.IsListening) ;
            ServerLogger.Instance.writeLog("Server stopped...");
            server.Events.ClientConnected -= EventClientConnected;
            server.Events.ClientDisconnected -= EventClientDisconnected;
            server.Events.DataReceived -= EventDataReceived;
        }
        public void startServer(string port, bool localhost)
        {
            string ip;
            if (localhost)
                ip = "127.0.0.1";
            else
            {
                ip = Dns.GetHostByName(Dns.GetHostName()).AddressList[0].ToString();
            }

            server = new SimpleTcpServer($"{ip}:{port}");

            ServerLogger.Instance.writeLog("Starting...");
            server.Start();
            while (!server.IsListening) ;
            ServerLogger.Instance.writeLog("Done.");
            ServerLogger.Instance.writeLog($"Server ip: {ip}");
            ServerLogger.Instance.writeLog($"Listening on port {port}...");
            server.Events.ClientConnected += EventClientConnected;
            server.Events.ClientDisconnected += EventClientDisconnected;
            server.Events.DataReceived += EventDataReceived;

            contractLoadingThread = new ContractLoadingThread();
            contractLoadingThread.start();
        }
        public void send<T>(ClientModel client, MessageBase message) where T : MessageBase
        {
            using (var outputStream = new MemoryStream())
            {
                using (var bw = new BinaryWriter(outputStream))
                {
                    byte[] data = JsonSerializer.SerializeToUtf8Bytes((T)message, sOption);
                    outputStream.Capacity = data.Length + 4;
                    bw.Write(data.Length);
                    bw.Write(data);
                    outputStream.Position = 0;
                    server.Send(client.IpPort, outputStream.Length, outputStream);
                    ServerLogger.Instance.writeLog($"{outputStream.Length} bytes sent to {client.IpPort}.");
                }
            }
        }
        public void sendToAll<T>(MessageBase message) where T : MessageBase
        {
            foreach (ClientModel client in clients.Clients)
                send<T>(client, message);
        }
        private void EventDataReceived(object sender, DataReceivedEventArgs e)
        {
            ClientModel client = clients.getClient(e.IpPort);
            ServerLogger.Instance.writeLog($"{e.Data.Length} bytes received from {e.IpPort}");
            lock (lockObject)
            {
                receiveData(e.Data, client);
            }
        }
        private void receiveData(byte[] data, ClientModel client)
        {
            currentMessageStream.Capacity = currentMessageStream.Capacity + data.Length;
            currentMessageStream.Write(data, 0, data.Length);
            currentMessageStream.Position = readPos;
            if (currentMessageSize == 0)
            {
                currentMessageSize = br.ReadInt32();
                if (currentMessageStream.Length - 4 < currentMessageSize)
                {
                    readPos = currentMessageStream.Position;
                    currentMessageStream.Seek(0, SeekOrigin.End);
                    return;
                }
            }
            handleData(br.ReadBytes(currentMessageSize), client);
            if (currentMessageStream.Position < currentMessageStream.Length)
            {
                byte[] remaining = br.ReadBytes((int)(currentMessageStream.Length - currentMessageStream.Position));
                resetStream();
                receiveData(remaining, client);
            }
            else if (currentMessageStream.Position == currentMessageStream.Length)
            {
                resetStream();
            }
        }
        private void resetStream()
        {
            currentMessageStream.Close();
            br.Close();
            currentMessageStream = new MemoryStream();
            br = new BinaryReader(currentMessageStream);
            readPos = 0;
            currentMessageSize = 0;
        }
        private void handleData(byte[] data, ClientModel client)
        {
            try
            {         
                MessageBase msg = JsonSerializer.Deserialize<MessageBase>(new ReadOnlySpan<byte>(data));

                if (msg.Type == "Login")
                {
                    LoginMessage loginMessage = JsonSerializer.Deserialize<LoginMessage>(new ReadOnlySpan<byte>(data));
                    ServerLogger.Instance.writeLog($"Server: {loginMessage.Type}," +
                        $" username: {loginMessage.Username}," +
                        $" auth: {loginMessage.Authority}");
                    if (authenticateUser(client, loginMessage)) { 
                        send<DatabaseChangedMessage>(client, new DatabaseChangedMessage(""));
                    }
                }
                else {
                    if (msg.SessionId == null || msg.SessionId.Equals("") || !msg.SessionId.Equals(client.SessionId)) {
                        ServerLogger.Instance.writeLog($"Unknown client at port {client.IpPort}. Ending session.");
                        client.resetUser();
                        return;
                    }
                    ServerLogger.Instance.writeLog($"Session id: {client.SessionId}");
                    if (msg.Type == "LoadContract")
                    {
                        if (client.userData.Auth > 1)
                        {
                            LoadMessage lmsg = JsonSerializer.Deserialize<LoadMessage>(new ReadOnlySpan<byte>(data));
                            if (lmsg.Data != "")
                                contractLoadingThread.addTask(lmsg, client);
                        }
                    }
                    if (msg.Type == "RecordRequest")
                    {
                        if (client.userData.Auth > 0)
                        {
                            GetRecordMessage grmsg = JsonSerializer.Deserialize<GetRecordMessage>(new ReadOnlySpan<byte>(data));
                            ServerLogger.Instance.writeLog($"Server: {grmsg.Type}, route: {grmsg.Route}");
                            List<byte[]> res = RouteContainer.Instance.getDataByRoute(grmsg.Route);
                            send<GetRecordMessage>(client, new GetRecordMessage(grmsg.Route, new byte[0], 0, client.SessionId));
                            for (int i = 0; i < res.Count; i++)
                                send<GetRecordMessage>(client, new GetRecordMessage(grmsg.Route, res[i], i + 1, client.SessionId));
                        }
                    }
                }

            }
            catch (Exception ex)
            {
                ServerLogger.Instance.writeLog($"Server error: {ex.StackTrace}\n {ex.Message}");
                send<ErrorMessage>(client, new ErrorMessage(ex.Message, ""));
            }
        }

        private void EventClientDisconnected(object sender, ClientDisconnectedEventArgs e)
        {
            clients.removeClient(e.IpPort);
            ServerLogger.Instance.writeLog($"{e.IpPort }: Disconnected.");

        }

        private void EventClientConnected(object sender, ClientConnectedEventArgs e)
        {
            clients.addClient(e.IpPort);
            ServerLogger.Instance.writeLog($"{e.IpPort }: Connected.");
        }
        private bool authenticateUser(ClientModel client, LoginMessage loginMessage)
        {
            using (DbHandler dbh = new DbHandler())
            {
                List<UserData> users = dbh.getUsers(loginMessage.Username);
                if (users.Count == 0)
                {
                    loginMessage.Password = "";
                    loginMessage.Authority = 0;
                    loginMessage.Info = "Nincs ilyen nevű felhasználó!";
                }
                else
                {
                    if (!users[0].Password.Equals(loginMessage.Password))
                    {
                        loginMessage.Password = "";
                        loginMessage.Authority = 0;
                        loginMessage.Info = "Hibás jelszó!";
                    }
                    else
                    {
                        users[0].Password = "";
                        client.userData = users[0];
                        client.SessionId = generateSessionId();
                        loginMessage.Password = "";
                        loginMessage.Authority = users[0].Auth;
                        loginMessage.SessionId = client.SessionId;
                        loginMessage.Info = "OK";
                        ServerLogger.Instance.writeLog($"User logged in. username: {users[0].Username}, authority: {users[0].Auth}, port: {client.IpPort}, session id: {client.SessionId}");
                    }
                }
                send<LoginMessage>(client, loginMessage);
            }
            return loginMessage.Info == "OK";
        }
        private string generateSessionId() {
            int length = 10;
            Random random = new Random();
            const string chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
            string sessionId =  new string(Enumerable.Repeat(chars, length).Select(s => s[random.Next(s.Length)]).ToArray());
            using (var sha256 = SHA256.Create())
            {
                var hashedBytes = sha256.ComputeHash(Encoding.UTF8.GetBytes(sessionId));
                sessionId = BitConverter.ToString(hashedBytes).Replace("-", "").ToLower();
            }
            return sessionId;
        }

    }
}
