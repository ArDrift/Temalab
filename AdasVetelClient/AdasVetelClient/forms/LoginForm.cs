using AdasVetelClient.tcp;
using AdasVetelServer.messages;
using System;
using System.Security.Cryptography;
using System.Text;
using System.Windows.Forms;

namespace AdasVetelClient.forms
{
    public partial class LoginForm : MaterialSkin.Controls.MaterialForm, ClientView
    {
        private string username = "";
        private string password = "";
        public string Username { get { return username; } }
        public string Password { get { return password; } }
        public LoginForm()
        {
            InitializeComponent();
        }
        public LoginForm(string username, string message)
        {
            InitializeComponent();
            userNameBox.Text = username;
            warningLabel.Text = message;
        }
        private void LoginForm_Load(object sender, EventArgs e)
        {
            TcpClientWrapper.Instance.views.Add(this);
        }

        private void materialRaisedButton1_Click(object sender, EventArgs e)
        {
            password = passWordBox.Text;
            if (password.Length < 5)
            {
                warningLabel.Text = "A jelszó nem lehet 5 karakternél kisebb!";
                return;
            }
            username = userNameBox.Text;
            if (username.Length < 5)
            {
                warningLabel.Text = "A felhasználónév nem lehet 5 karakternél kisebb!";
                return;
            }
            using (var sha256 = SHA256.Create())
            {
                var hashedBytes = sha256.ComputeHash(Encoding.UTF8.GetBytes(password));
                password = BitConverter.ToString(hashedBytes).Replace("-", "").ToLower();
            }

            DialogResult = DialogResult.OK;
        }

        public void connected()
        {
        }
        public void clientLoggedIn() { }
        public bool isActive()
        {
            return !IsDisposed;
        }
        public void disconnected()
        {
            DialogResult = DialogResult.Cancel;
        }

        public void handleMessage(MessageBase message, string type)
        {
        }
    }
}
