using AdasVetelClient.tcp;
using AdasVetelServer.messages;
using System;
using System.Windows.Forms;


namespace AdasVetelClient
{
    public partial class ConnectForm : MaterialSkin.Controls.MaterialForm, ClientView
    {
        public ConnectForm()
        {
            InitializeComponent();
        }
        public void connected()
        {
            DialogResult = DialogResult.OK;
        }

        public void disconnected()
        {

        }
        public void clientLoggedIn() { }

        public void handleMessage(MessageBase message, string type)
        {
        }

        public bool isActive()
        {
            return !IsDisposed;
        }

        private void closeBtn_Click(object sender, EventArgs e)
        {
            DialogResult = DialogResult.Cancel;
        }

        private void ConnectForm_Load(object sender, EventArgs e)
        {
            TcpClientWrapper.Instance.views.Add(this);
        }
    }
}
