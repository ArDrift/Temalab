using AdasVetelClient.forms;
using AdasVetelClient.tcp;
using AdasVetelServer.messages;
using System;
using System.Drawing;
using System.IO;
using System.Threading;
using System.Windows.Forms;

namespace AdasVetelClient
{
    public partial class AdasVetel : MaterialSkin.Controls.MaterialForm, ClientView
    {
        private readonly int MAX_SENDABLE = 10;

        public AdasVetel()
        {
            InitializeComponent();


            docLoadButton.BackgroundImage = new Bitmap(docLoadButton.BackgroundImage, new Size(docLoadButton.Width, docLoadButton.Height));
            txtLoadButton.BackgroundImage = new Bitmap(txtLoadButton.BackgroundImage, new Size(txtLoadButton.Width, txtLoadButton.Height));
            pdfLoadButton.BackgroundImage = new Bitmap(pdfLoadButton.BackgroundImage, new Size(pdfLoadButton.Width, pdfLoadButton.Height));
            imageLoadButton.BackgroundImage = new Bitmap(imageLoadButton.BackgroundImage, new Size(imageLoadButton.Width, imageLoadButton.Height));
            connectButton.BackgroundImage = new Bitmap(connectButton.BackgroundImage, new Size(connectButton.Width, connectButton.Height));
        }

        private void readContract_Click(object sender, EventArgs e)
        {
            if (TcpClientWrapper.Instance.getClientAuth() < TcpClient.Authority.HIGH)
            {
                MessageBox.Show("Ehhez a művelethez nincs jogköröd!", "Message", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }
            if (TcpClientWrapper.Instance.isConnected())
            {
                if (fileFlowLayout.Controls.Count == 0)
                {
                    MessageBox.Show("Nem történt beolvasás", " Eredmény", MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
                else
                {
                    int fileCount = fileFlowLayout.Controls.Count > MAX_SENDABLE ? MAX_SENDABLE : fileFlowLayout.Controls.Count;
                    progressBar.Value = 5;
                    progressBar.Step = progressBar.Maximum / fileCount;                    
                    Thread contractSender = new Thread(contractSendingThread);
                    contractSender.IsBackground = true;
                    contractSender.Start();
                }
            }
        }

        private void contractSendingThread() {

            int fileCount = fileFlowLayout.Controls.Count > MAX_SENDABLE ? MAX_SENDABLE : fileFlowLayout.Controls.Count;

            foreach (FileIconControl item in fileFlowLayout.Controls)
            {
                if (fileCount == 0)
                {
                    timer1.Enabled = true;
                    limitLabel.Visible = true;
                    break;
                }
                LoadMessage msg = new LoadMessage(item.getText(), item.FileName, "");
                if (msg.Data != "")
                {
                    TcpClientWrapper.Instance.send<LoadMessage>(msg);
                }
            }
        }

        private void docLoadButton_Click(object sender, EventArgs e)
        {
            loadFileNames("Word Document|*.docx|" +
                "Word 97-2003|*.doc");
        }

        private void txtLoadButton_Click(object sender, EventArgs e)
        {
            loadFileNames("Text File|*.txt");
        }

        private void pdfLoadButton_Click(object sender, EventArgs e)
        {
            loadFileNames("PDF|*.pdf");
        }

        private void imageLoadButton_Click(object sender, EventArgs e)
        {
            loadFileNames("JPG|*.jpg|" +
                 "JPEG|*.jpeg|" +
                 "PNG|*.png|" +
                 "PDF|*.pdf");
        }
        private void loadFileNames(string filter)
        {
            string pathFile = Application.StartupPath + "\\config\\docPrevPath.loc";
            string dir = File.ReadAllText(pathFile);
            if (!Directory.Exists(dir))
                dir = "c:\\";
            using (OpenFileDialog ofd = new OpenFileDialog()
            {
                InitialDirectory = dir,
                ValidateNames = true,
                Multiselect = true,
                Filter = filter
            })
            {
                if (ofd.ShowDialog() == DialogResult.OK)
                {
                    foreach (var filename in ofd.FileNames)
                    {
                        fileFlowLayout.Controls.Add(new FileIconControl(filename,
                            Path.GetFileName(filename), fileFlowLayout.Controls));
                    }
                    File.WriteAllText(pathFile, Path.GetDirectoryName(ofd.FileName));
                }
            }
        }
        private void viewContracts_Click(object sender, EventArgs e)
        {
            if (TcpClientWrapper.Instance.getClientAuth() == TcpClient.Authority.NONE)
            {
                MessageBox.Show("Ehhez a művelethez be kell jelentkezned!", "Message", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }
            DisplayForm displayForm = new DisplayForm(null);
            displayForm.Show();
        }


        private void AdasVetel_Load(object sender, EventArgs e)
        {
            TcpClientWrapper.Instance.views.Add(this);
            connectToServer();

        }
        private void connectToServer()
        {
            using (ConnectForm connectForm = new ConnectForm())
            {
                TcpClientWrapper.Instance.connectToServer();
                connectForm.ShowDialog();
                if (connectForm.DialogResult == DialogResult.Cancel)
                {
                    TcpClientWrapper.Instance.stopConnecting();
                }
                else if (connectForm.DialogResult == DialogResult.OK)
                {
                    logInToServer();
                }
            }
        }


        private void clearListBtn_Click(object sender, EventArgs e)
        {
            fileFlowLayout.Controls.Clear();
        }

        private void connectButton_Click(object sender, EventArgs e)
        {
            if (!TcpClientWrapper.Instance.isConnected())
            {
                connectToServer();
            }
            else if (!TcpClientWrapper.Instance.isLoggedIn())
            {
                logInToServer();
            }
            else
            {

            }
        }
        public bool isActive()
        {
            return !IsDisposed;
        }
        public void clientLoggedIn()
        {
            userNameLabel.Text = TcpClientWrapper.Instance.getClientUsername();
            connectionSignal.BackColor = Color.Green;
            progressLabel.Text = "";
            progressBar.Value = 0;
            readContract.Enabled = true;
            fileFlowLayout.Controls.Clear();
        }
        public void connected()
        {
            Console.WriteLine("Server connected.");
        }


        public void disconnected()
        {
            Invoke((MethodInvoker)delegate
            {
                connectionSignal.BackColor = Color.Red;
                userNameLabel.Text = "";
                Console.WriteLine("Server disconnected.");
                connectToServer();
            });

        }

        public void handleMessage(MessageBase message, string type)
        {
            Invoke((MethodInvoker)delegate
            {
                if (type == "LoadContract")
                {
                    LoadMessage lmsg = (LoadMessage)message;
                    for (int i = 0; i < fileFlowLayout.Controls.Count; i++)
                    {
                        FileIconControl fic = (FileIconControl)fileFlowLayout.Controls[i];
                        if (fic.FileName == lmsg.FileName)
                        {
                            fileFlowLayout.Controls.RemoveAt(i);
                            i--;
                        }
                    }
                    progressLabel.Text = Path.GetFileName(lmsg.FileName) + " beolvasva.";
                    progressBar.PerformStep();
                    if (progressBar.Value == progressBar.Maximum)
                    {
                        progressLabel.Text = "Beolvasás kész!";
                        readContract.Enabled = true;
                    }
                }
                else if (type == "ProgressInfo")
                {
                    ProgressInfoMessage pimsg = (ProgressInfoMessage)message;
                    if (pimsg.ProgressType == "OK")
                    {
                        progressLabel.Text = Path.GetFileName(pimsg.Info) + " analizálása...";
                    }
                    else if (pimsg.ProgressType == "ERROR")
                    {
                        MessageBox.Show(pimsg.Info, "Message", MessageBoxButtons.OK, MessageBoxIcon.Error);
                        progressBar.PerformStep();
                        progressLabel.Text = pimsg.Info;
                        readContract.Enabled = true;
                    }

                }
                else if (type == "Login")
                {
                    LoginMessage loginMessage = (LoginMessage)message;
                    if (loginMessage.Authority == 0)
                    {
                        logInToServer(loginMessage.Username, loginMessage.Info);
                    }
                    else if (loginMessage.Authority == 1 || loginMessage.Authority == 2)
                    {
                        TcpClientWrapper.Instance.logInClient(loginMessage);
                    }
                }
                else if (type == "ErrorMessage")
                {
                    ErrorMessage ermsg = (ErrorMessage)message;
                    MessageBox.Show(ermsg.Message, "Message", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
            });
        }
        public void logInToServer(string username = "admin", string message = "")
        {
            using (LoginForm loginForm = new LoginForm(username, message))
            {
                loginForm.ShowDialog();
                if (loginForm.DialogResult == DialogResult.OK)
                {
                    username = loginForm.Username;
                    string password = loginForm.Password;
                    LoginMessage loginMessage = new LoginMessage(username, password, 0, "", "");
                    TcpClientWrapper.Instance.send<LoginMessage>(loginMessage);
                }
            }
        }

        private void timer1_Tick(object sender, EventArgs e)
        {
            timer1.Enabled = false;
            limitLabel.Visible = false;
        }
    }
}
