using AdasVetelServer.dbHandler;
using AdasVetelServer.logger;
using AdasVetelServer.messages;
using AdasVetelServer.model;
using AdasVetelServer.tcp;
using System;
using System.Security.Cryptography;
using System.Text;
using System.Windows.Forms;

namespace AdasVetelServer
{
    public partial class AdasVetelServer : Form
    {
        public AdasVetelServer()
        {
            InitializeComponent();
            stopBox.Enabled = false;
            serverInfo.Visible = false;
            ServerLogger.Instance.output = serverLog;

        }

        private void stopServer_Click(object sender, EventArgs e)
        {
            TcpServerWrapper.Instance.stopServer();
            startBox.Enabled = true;
            stopBox.Enabled = false;
            serverInfo.Visible = false;
        }
        private void startServer()
        {
            serverLog.Text = "";
            TcpServerWrapper.Instance.startServer(portNumber.Value.ToString(), domainCheckBox.Checked);
            startBox.Enabled = false;
            stopBox.Enabled = true;
            serverInfo.Visible = true;
        }
        private void startServer_Click(object sender, EventArgs e)
        {
            startServer();
        }

        private void resetDataBaseBtn_Click(object sender, EventArgs e)
        {
            resetDatabase();
        }      
        private void resetDatabase() {
            DialogResult dr = MessageBox.Show("Biztos törli az adatbázis tartalmát?",
                "Figyelem!", MessageBoxButtons.YesNo, MessageBoxIcon.Question);
            if (dr == DialogResult.Yes)
            {
                using (DbHandler dbh = new DbHandler())
                {
                    dbh.deleteAllContent();
                    ServerLogger.Instance.writeLog("Database records deleted.");
                    TcpServerWrapper.Instance.sendToAll<DatabaseChangedMessage>(new DatabaseChangedMessage(""));
                }
            }
        }
        private void clearLogBtn_Click(object sender, EventArgs e)
        {
            serverLog.Text = "";
        }

        private void AdasVetelServer_Load(object sender, EventArgs e)
        {
            startServer();
        }

        private void logSwitchButton_Click(object sender, EventArgs e)
        {
            if (ServerLogger.Instance.Verbose)
            {
                ServerLogger.Instance.Verbose = false;
                logSwitchButton.Text = "Log bekapcs";
            }
            else
            {
                ServerLogger.Instance.Verbose = true;
                logSwitchButton.Text = "Log kikapcs";
            }
        }

        private void registerUserButton_Click(object sender, EventArgs e)
        {
            if (passwordBox.Text.Length < 5)
            {
                warningLabel.Text = "A jelszónak legalább 5 karakter hosszúnak kell lennie!";
                return;
            }
            if (userNameBox.Text.Length < 5)
            {
                warningLabel.Text = "A felhasználónévnek legalább 5 karakter hosszúnak kell lennie!";
                return;
            }
            if (!passwordBox.Text.Equals(Password2Box.Text))
            {
                warningLabel.Text = "A két jelszó nem egyezik!";
                return;
            }
            using (DbHandler dbh = new DbHandler())
            {
                UserData user = new UserData(userNameBox.Text, passwordBox.Text, (int)authBox.Value);
                using (var sha256 = SHA256.Create())
                {
                    var hashedBytes = sha256.ComputeHash(Encoding.UTF8.GetBytes(user.Password));
                    user.Password = BitConverter.ToString(hashedBytes).Replace("-", "").ToLower();
                }
                if (dbh.getUsers(userNameBox.Text).Count != 0)
                {
                    warningLabel.Text = "A felhasználónév már foglalt!";
                }
                else
                {
                    dbh.insertUserDataToDatabase(user);
                    dbh.submintChanges();
                    warningLabel.Text = "A regisztráció sikeres volt!";
                }
            }
        }

    }
}
