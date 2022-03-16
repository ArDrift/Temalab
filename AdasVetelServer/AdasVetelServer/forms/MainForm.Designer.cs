
namespace AdasVetelServer
{
    partial class AdasVetelServer
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(AdasVetelServer));
            this.tabPage3 = new System.Windows.Forms.TabPage();
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.warningLabel = new System.Windows.Forms.Label();
            this.userNameBox = new System.Windows.Forms.TextBox();
            this.registerUserButton = new System.Windows.Forms.Button();
            this.authBox = new System.Windows.Forms.NumericUpDown();
            this.label2 = new System.Windows.Forms.Label();
            this.label6 = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.passwordBox = new System.Windows.Forms.TextBox();
            this.label4 = new System.Windows.Forms.Label();
            this.Password2Box = new System.Windows.Forms.TextBox();
            this.label5 = new System.Windows.Forms.Label();
            this.resetDataBaseBtn = new System.Windows.Forms.Button();
            this.tabPage1 = new System.Windows.Forms.TabPage();
            this.splitContainer2 = new System.Windows.Forms.SplitContainer();
            this.logSwitchButton = new System.Windows.Forms.Button();
            this.clearLogBtn = new System.Windows.Forms.Button();
            this.startBox = new System.Windows.Forms.GroupBox();
            this.domainCheckBox = new System.Windows.Forms.CheckBox();
            this.portNumber = new System.Windows.Forms.NumericUpDown();
            this.label1 = new System.Windows.Forms.Label();
            this.startServerBtn = new System.Windows.Forms.Button();
            this.stopBox = new System.Windows.Forms.GroupBox();
            this.stopServerBtn = new System.Windows.Forms.Button();
            this.serverInfo = new System.Windows.Forms.Label();
            this.serverLog = new System.Windows.Forms.TextBox();
            this.tabControl1 = new System.Windows.Forms.TabControl();
            this.tabPage3.SuspendLayout();
            this.groupBox1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.authBox)).BeginInit();
            this.tabPage1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer2)).BeginInit();
            this.splitContainer2.Panel1.SuspendLayout();
            this.splitContainer2.Panel2.SuspendLayout();
            this.splitContainer2.SuspendLayout();
            this.startBox.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.portNumber)).BeginInit();
            this.stopBox.SuspendLayout();
            this.tabControl1.SuspendLayout();
            this.SuspendLayout();
            // 
            // tabPage3
            // 
            this.tabPage3.Controls.Add(this.groupBox1);
            this.tabPage3.Controls.Add(this.resetDataBaseBtn);
            this.tabPage3.Location = new System.Drawing.Point(4, 29);
            this.tabPage3.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.tabPage3.Name = "tabPage3";
            this.tabPage3.Padding = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.tabPage3.Size = new System.Drawing.Size(816, 520);
            this.tabPage3.TabIndex = 2;
            this.tabPage3.Text = "Adatbázis";
            this.tabPage3.UseVisualStyleBackColor = true;
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.userNameBox);
            this.groupBox1.Controls.Add(this.passwordBox);
            this.groupBox1.Controls.Add(this.Password2Box);
            this.groupBox1.Controls.Add(this.authBox);
            this.groupBox1.Controls.Add(this.registerUserButton);
            this.groupBox1.Controls.Add(this.warningLabel);
            this.groupBox1.Controls.Add(this.label2);
            this.groupBox1.Controls.Add(this.label6);
            this.groupBox1.Controls.Add(this.label3);
            this.groupBox1.Controls.Add(this.label4);
            this.groupBox1.Controls.Add(this.label5);
            this.groupBox1.Location = new System.Drawing.Point(8, 5);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(802, 335);
            this.groupBox1.TabIndex = 11;
            this.groupBox1.TabStop = false;
            // 
            // warningLabel
            // 
            this.warningLabel.AutoSize = true;
            this.warningLabel.Font = new System.Drawing.Font("Century Gothic", 9F, System.Drawing.FontStyle.Bold);
            this.warningLabel.ForeColor = System.Drawing.Color.Red;
            this.warningLabel.Location = new System.Drawing.Point(6, 302);
            this.warningLabel.Name = "warningLabel";
            this.warningLabel.Size = new System.Drawing.Size(0, 18);
            this.warningLabel.TabIndex = 11;
            // 
            // userNameBox
            // 
            this.userNameBox.Location = new System.Drawing.Point(195, 95);
            this.userNameBox.Name = "userNameBox";
            this.userNameBox.Size = new System.Drawing.Size(288, 26);
            this.userNameBox.TabIndex = 0;
            // 
            // registerUserButton
            // 
            this.registerUserButton.Location = new System.Drawing.Point(617, 274);
            this.registerUserButton.Name = "registerUserButton";
            this.registerUserButton.Size = new System.Drawing.Size(179, 55);
            this.registerUserButton.TabIndex = 4;
            this.registerUserButton.Text = "Felhasználó regisztrálása";
            this.registerUserButton.UseVisualStyleBackColor = true;
            this.registerUserButton.Click += new System.EventHandler(this.registerUserButton_Click);
            // 
            // authBox
            // 
            this.authBox.Location = new System.Drawing.Point(195, 235);
            this.authBox.Maximum = new decimal(new int[] {
            2,
            0,
            0,
            0});
            this.authBox.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.authBox.Name = "authBox";
            this.authBox.Size = new System.Drawing.Size(120, 26);
            this.authBox.TabIndex = 3;
            this.authBox.Value = new decimal(new int[] {
            1,
            0,
            0,
            0});
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(42, 95);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(125, 20);
            this.label2.TabIndex = 1;
            this.label2.Text = "Felhasználónév:";
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Font = new System.Drawing.Font("Century Gothic", 12F, System.Drawing.FontStyle.Bold);
            this.label6.Location = new System.Drawing.Point(23, 27);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(144, 23);
            this.label6.TabIndex = 9;
            this.label6.Text = "Új felhasználó";
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(112, 146);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(55, 20);
            this.label3.TabIndex = 4;
            this.label3.Text = "Jelszó:";
            // 
            // passwordBox
            // 
            this.passwordBox.Location = new System.Drawing.Point(195, 140);
            this.passwordBox.Name = "passwordBox";
            this.passwordBox.Size = new System.Drawing.Size(288, 26);
            this.passwordBox.TabIndex = 1;
            this.passwordBox.UseSystemPasswordChar = true;
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(104, 237);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(63, 20);
            this.label4.TabIndex = 5;
            this.label4.Text = "Jogkör:";
            // 
            // Password2Box
            // 
            this.Password2Box.Location = new System.Drawing.Point(195, 187);
            this.Password2Box.Name = "Password2Box";
            this.Password2Box.Size = new System.Drawing.Size(288, 26);
            this.Password2Box.TabIndex = 2;
            this.Password2Box.UseSystemPasswordChar = true;
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(20, 193);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(147, 20);
            this.label5.TabIndex = 6;
            this.label5.Text = "Jelszó mégegyszer:";
            // 
            // resetDataBaseBtn
            // 
            this.resetDataBaseBtn.BackColor = System.Drawing.Color.IndianRed;
            this.resetDataBaseBtn.Font = new System.Drawing.Font("Century Gothic", 10.8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.resetDataBaseBtn.Location = new System.Drawing.Point(603, 428);
            this.resetDataBaseBtn.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.resetDataBaseBtn.Name = "resetDataBaseBtn";
            this.resetDataBaseBtn.Size = new System.Drawing.Size(189, 70);
            this.resetDataBaseBtn.TabIndex = 5;
            this.resetDataBaseBtn.Text = "Adatbázis törlése";
            this.resetDataBaseBtn.UseVisualStyleBackColor = false;
            this.resetDataBaseBtn.Click += new System.EventHandler(this.resetDataBaseBtn_Click);
            // 
            // tabPage1
            // 
            this.tabPage1.Controls.Add(this.splitContainer2);
            this.tabPage1.Location = new System.Drawing.Point(4, 29);
            this.tabPage1.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.tabPage1.Name = "tabPage1";
            this.tabPage1.Padding = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.tabPage1.Size = new System.Drawing.Size(816, 520);
            this.tabPage1.TabIndex = 0;
            this.tabPage1.Text = "Kezdőlap";
            this.tabPage1.UseVisualStyleBackColor = true;
            // 
            // splitContainer2
            // 
            this.splitContainer2.Dock = System.Windows.Forms.DockStyle.Fill;
            this.splitContainer2.Location = new System.Drawing.Point(3, 2);
            this.splitContainer2.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.splitContainer2.Name = "splitContainer2";
            // 
            // splitContainer2.Panel1
            // 
            this.splitContainer2.Panel1.Controls.Add(this.logSwitchButton);
            this.splitContainer2.Panel1.Controls.Add(this.clearLogBtn);
            this.splitContainer2.Panel1.Controls.Add(this.startBox);
            this.splitContainer2.Panel1.Controls.Add(this.stopBox);
            this.splitContainer2.Panel1.Controls.Add(this.serverInfo);
            // 
            // splitContainer2.Panel2
            // 
            this.splitContainer2.Panel2.Controls.Add(this.serverLog);
            this.splitContainer2.Size = new System.Drawing.Size(810, 516);
            this.splitContainer2.SplitterDistance = 261;
            this.splitContainer2.TabIndex = 4;
            // 
            // logSwitchButton
            // 
            this.logSwitchButton.Location = new System.Drawing.Point(21, 9);
            this.logSwitchButton.Name = "logSwitchButton";
            this.logSwitchButton.Size = new System.Drawing.Size(104, 36);
            this.logSwitchButton.TabIndex = 5;
            this.logSwitchButton.Text = "Log kikapcs";
            this.logSwitchButton.UseVisualStyleBackColor = true;
            this.logSwitchButton.Click += new System.EventHandler(this.logSwitchButton_Click);
            // 
            // clearLogBtn
            // 
            this.clearLogBtn.Location = new System.Drawing.Point(148, 9);
            this.clearLogBtn.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.clearLogBtn.Name = "clearLogBtn";
            this.clearLogBtn.Size = new System.Drawing.Size(102, 36);
            this.clearLogBtn.TabIndex = 3;
            this.clearLogBtn.Text = "Log törlése";
            this.clearLogBtn.UseVisualStyleBackColor = true;
            this.clearLogBtn.Click += new System.EventHandler(this.clearLogBtn_Click);
            // 
            // startBox
            // 
            this.startBox.Controls.Add(this.domainCheckBox);
            this.startBox.Controls.Add(this.portNumber);
            this.startBox.Controls.Add(this.label1);
            this.startBox.Controls.Add(this.startServerBtn);
            this.startBox.Location = new System.Drawing.Point(30, 139);
            this.startBox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.startBox.Name = "startBox";
            this.startBox.Padding = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.startBox.Size = new System.Drawing.Size(177, 242);
            this.startBox.TabIndex = 0;
            this.startBox.TabStop = false;
            // 
            // domainCheckBox
            // 
            this.domainCheckBox.AutoSize = true;
            this.domainCheckBox.Checked = true;
            this.domainCheckBox.CheckState = System.Windows.Forms.CheckState.Checked;
            this.domainCheckBox.Location = new System.Drawing.Point(12, 101);
            this.domainCheckBox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.domainCheckBox.Name = "domainCheckBox";
            this.domainCheckBox.Size = new System.Drawing.Size(97, 24);
            this.domainCheckBox.TabIndex = 4;
            this.domainCheckBox.Text = "localhost";
            this.domainCheckBox.UseVisualStyleBackColor = true;
            // 
            // portNumber
            // 
            this.portNumber.Font = new System.Drawing.Font("Century Gothic", 10.8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.portNumber.Location = new System.Drawing.Point(68, 36);
            this.portNumber.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.portNumber.Maximum = new decimal(new int[] {
            10000,
            0,
            0,
            0});
            this.portNumber.Name = "portNumber";
            this.portNumber.Size = new System.Drawing.Size(87, 30);
            this.portNumber.TabIndex = 3;
            this.portNumber.Value = new decimal(new int[] {
            9999,
            0,
            0,
            0});
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Font = new System.Drawing.Font("Century Gothic", 10.8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label1.Location = new System.Drawing.Point(6, 38);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(51, 22);
            this.label1.TabIndex = 1;
            this.label1.Text = "Port:";
            // 
            // startServerBtn
            // 
            this.startServerBtn.Location = new System.Drawing.Point(10, 166);
            this.startServerBtn.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.startServerBtn.Name = "startServerBtn";
            this.startServerBtn.Size = new System.Drawing.Size(145, 66);
            this.startServerBtn.TabIndex = 0;
            this.startServerBtn.Text = "Szerver indítása";
            this.startServerBtn.UseVisualStyleBackColor = true;
            this.startServerBtn.Click += new System.EventHandler(this.startServer_Click);
            // 
            // stopBox
            // 
            this.stopBox.Controls.Add(this.stopServerBtn);
            this.stopBox.Location = new System.Drawing.Point(40, 400);
            this.stopBox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.stopBox.Name = "stopBox";
            this.stopBox.Padding = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.stopBox.Size = new System.Drawing.Size(150, 99);
            this.stopBox.TabIndex = 1;
            this.stopBox.TabStop = false;
            // 
            // stopServerBtn
            // 
            this.stopServerBtn.Location = new System.Drawing.Point(12, 24);
            this.stopServerBtn.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.stopServerBtn.Name = "stopServerBtn";
            this.stopServerBtn.Size = new System.Drawing.Size(129, 53);
            this.stopServerBtn.TabIndex = 1;
            this.stopServerBtn.Text = "Leállítás";
            this.stopServerBtn.UseVisualStyleBackColor = true;
            this.stopServerBtn.Click += new System.EventHandler(this.stopServer_Click);
            // 
            // serverInfo
            // 
            this.serverInfo.AutoSize = true;
            this.serverInfo.Font = new System.Drawing.Font("Century Gothic", 13.8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.serverInfo.Location = new System.Drawing.Point(25, 71);
            this.serverInfo.Name = "serverInfo";
            this.serverInfo.Size = new System.Drawing.Size(203, 30);
            this.serverInfo.TabIndex = 2;
            this.serverInfo.Text = "A szerver aktív...";
            // 
            // serverLog
            // 
            this.serverLog.Dock = System.Windows.Forms.DockStyle.Fill;
            this.serverLog.Font = new System.Drawing.Font("Arial", 10.2F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.serverLog.Location = new System.Drawing.Point(0, 0);
            this.serverLog.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.serverLog.Multiline = true;
            this.serverLog.Name = "serverLog";
            this.serverLog.ReadOnly = true;
            this.serverLog.ScrollBars = System.Windows.Forms.ScrollBars.Both;
            this.serverLog.Size = new System.Drawing.Size(545, 516);
            this.serverLog.TabIndex = 3;
            // 
            // tabControl1
            // 
            this.tabControl1.Controls.Add(this.tabPage1);
            this.tabControl1.Controls.Add(this.tabPage3);
            this.tabControl1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.tabControl1.Location = new System.Drawing.Point(0, 0);
            this.tabControl1.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.tabControl1.Name = "tabControl1";
            this.tabControl1.SelectedIndex = 0;
            this.tabControl1.Size = new System.Drawing.Size(824, 553);
            this.tabControl1.TabIndex = 0;
            // 
            // AdasVetelServer
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(9F, 20F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(824, 553);
            this.Controls.Add(this.tabControl1);
            this.Font = new System.Drawing.Font("Century Gothic", 9F);
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.MinimumSize = new System.Drawing.Size(842, 600);
            this.Name = "AdasVetelServer";
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
            this.Text = "Adás Vétel Szerver";
            this.Load += new System.EventHandler(this.AdasVetelServer_Load);
            this.tabPage3.ResumeLayout(false);
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.authBox)).EndInit();
            this.tabPage1.ResumeLayout(false);
            this.splitContainer2.Panel1.ResumeLayout(false);
            this.splitContainer2.Panel1.PerformLayout();
            this.splitContainer2.Panel2.ResumeLayout(false);
            this.splitContainer2.Panel2.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer2)).EndInit();
            this.splitContainer2.ResumeLayout(false);
            this.startBox.ResumeLayout(false);
            this.startBox.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.portNumber)).EndInit();
            this.stopBox.ResumeLayout(false);
            this.tabControl1.ResumeLayout(false);
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.TabPage tabPage3;
        private System.Windows.Forms.Button resetDataBaseBtn;
        private System.Windows.Forms.TabPage tabPage1;
        private System.Windows.Forms.SplitContainer splitContainer2;
        private System.Windows.Forms.Button logSwitchButton;
        private System.Windows.Forms.Button clearLogBtn;
        private System.Windows.Forms.GroupBox startBox;
        private System.Windows.Forms.CheckBox domainCheckBox;
        private System.Windows.Forms.NumericUpDown portNumber;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Button startServerBtn;
        private System.Windows.Forms.GroupBox stopBox;
        private System.Windows.Forms.Button stopServerBtn;
        private System.Windows.Forms.Label serverInfo;
        private System.Windows.Forms.TextBox serverLog;
        private System.Windows.Forms.TabControl tabControl1;
        private System.Windows.Forms.Button registerUserButton;
        private System.Windows.Forms.TextBox userNameBox;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.NumericUpDown authBox;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.TextBox passwordBox;
        private System.Windows.Forms.TextBox Password2Box;
        private System.Windows.Forms.Label warningLabel;
    }
}

