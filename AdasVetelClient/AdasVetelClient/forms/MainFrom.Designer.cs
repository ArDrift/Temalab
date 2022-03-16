namespace AdasVetelClient
{
    partial class AdasVetel
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
            this.components = new System.ComponentModel.Container();
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(AdasVetel));
            this.readContract = new MaterialSkin.Controls.MaterialRaisedButton();
            this.groupBox2 = new System.Windows.Forms.GroupBox();
            this.docLoadButton = new System.Windows.Forms.Button();
            this.imageLoadButton = new System.Windows.Forms.Button();
            this.txtLoadButton = new System.Windows.Forms.Button();
            this.pdfLoadButton = new System.Windows.Forms.Button();
            this.splitContainer1 = new System.Windows.Forms.SplitContainer();
            this.fileFlowLayout = new System.Windows.Forms.FlowLayoutPanel();
            this.progressLabel = new System.Windows.Forms.Label();
            this.progressBar = new System.Windows.Forms.ProgressBar();
            this.panel1 = new System.Windows.Forms.Panel();
            this.connectionSignal = new System.Windows.Forms.PictureBox();
            this.connectButton = new System.Windows.Forms.Button();
            this.clearListBtn = new System.Windows.Forms.Button();
            this.viewContracts = new MaterialSkin.Controls.MaterialRaisedButton();
            this.timer1 = new System.Windows.Forms.Timer(this.components);
            this.limitLabel = new System.Windows.Forms.Label();
            this.userNameLabel = new MaterialSkin.Controls.MaterialLabel();
            this.groupBox2.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer1)).BeginInit();
            this.splitContainer1.Panel1.SuspendLayout();
            this.splitContainer1.Panel2.SuspendLayout();
            this.splitContainer1.SuspendLayout();
            this.panel1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.connectionSignal)).BeginInit();
            this.SuspendLayout();
            // 
            // readContract
            // 
            this.readContract.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.readContract.AutoSize = true;
            this.readContract.BackColor = System.Drawing.Color.DarkTurquoise;
            this.readContract.Cursor = System.Windows.Forms.Cursors.Hand;
            this.readContract.Depth = 0;
            this.readContract.FlatAppearance.BorderColor = System.Drawing.Color.Black;
            this.readContract.FlatAppearance.BorderSize = 2;
            this.readContract.FlatAppearance.MouseOverBackColor = System.Drawing.Color.FromArgb(((int)(((byte)(0)))), ((int)(((byte)(64)))), ((int)(((byte)(64)))));
            this.readContract.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.readContract.Location = new System.Drawing.Point(18, 349);
            this.readContract.MouseState = MaterialSkin.MouseState.HOVER;
            this.readContract.Name = "readContract";
            this.readContract.Primary = true;
            this.readContract.Size = new System.Drawing.Size(204, 60);
            this.readContract.TabIndex = 12;
            this.readContract.Text = "Szerződés(ek) beolvasása";
            this.readContract.UseVisualStyleBackColor = false;
            this.readContract.Click += new System.EventHandler(this.readContract_Click);
            // 
            // groupBox2
            // 
            this.groupBox2.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.groupBox2.Controls.Add(this.docLoadButton);
            this.groupBox2.Controls.Add(this.imageLoadButton);
            this.groupBox2.Controls.Add(this.txtLoadButton);
            this.groupBox2.Controls.Add(this.pdfLoadButton);
            this.groupBox2.Location = new System.Drawing.Point(126, 70);
            this.groupBox2.Name = "groupBox2";
            this.groupBox2.Size = new System.Drawing.Size(240, 229);
            this.groupBox2.TabIndex = 18;
            this.groupBox2.TabStop = false;
            // 
            // docLoadButton
            // 
            this.docLoadButton.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(192)))), ((int)(((byte)(192)))), ((int)(((byte)(255)))));
            this.docLoadButton.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("docLoadButton.BackgroundImage")));
            this.docLoadButton.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Stretch;
            this.docLoadButton.Cursor = System.Windows.Forms.Cursors.Hand;
            this.docLoadButton.FlatAppearance.MouseOverBackColor = System.Drawing.Color.FromArgb(((int)(((byte)(128)))), ((int)(((byte)(128)))), ((int)(((byte)(255)))));
            this.docLoadButton.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.docLoadButton.Location = new System.Drawing.Point(23, 21);
            this.docLoadButton.Name = "docLoadButton";
            this.docLoadButton.Size = new System.Drawing.Size(90, 90);
            this.docLoadButton.TabIndex = 14;
            this.docLoadButton.UseVisualStyleBackColor = false;
            this.docLoadButton.Click += new System.EventHandler(this.docLoadButton_Click);
            // 
            // imageLoadButton
            // 
            this.imageLoadButton.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(192)))), ((int)(((byte)(192)))), ((int)(((byte)(255)))));
            this.imageLoadButton.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("imageLoadButton.BackgroundImage")));
            this.imageLoadButton.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Stretch;
            this.imageLoadButton.Cursor = System.Windows.Forms.Cursors.Hand;
            this.imageLoadButton.FlatAppearance.MouseOverBackColor = System.Drawing.Color.FromArgb(((int)(((byte)(128)))), ((int)(((byte)(128)))), ((int)(((byte)(255)))));
            this.imageLoadButton.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.imageLoadButton.Location = new System.Drawing.Point(134, 121);
            this.imageLoadButton.Name = "imageLoadButton";
            this.imageLoadButton.Size = new System.Drawing.Size(90, 90);
            this.imageLoadButton.TabIndex = 17;
            this.imageLoadButton.UseVisualStyleBackColor = false;
            this.imageLoadButton.Click += new System.EventHandler(this.imageLoadButton_Click);
            // 
            // txtLoadButton
            // 
            this.txtLoadButton.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(192)))), ((int)(((byte)(192)))), ((int)(((byte)(255)))));
            this.txtLoadButton.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("txtLoadButton.BackgroundImage")));
            this.txtLoadButton.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Stretch;
            this.txtLoadButton.Cursor = System.Windows.Forms.Cursors.Hand;
            this.txtLoadButton.FlatAppearance.MouseOverBackColor = System.Drawing.Color.FromArgb(((int)(((byte)(128)))), ((int)(((byte)(128)))), ((int)(((byte)(255)))));
            this.txtLoadButton.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.txtLoadButton.Location = new System.Drawing.Point(134, 21);
            this.txtLoadButton.Name = "txtLoadButton";
            this.txtLoadButton.Size = new System.Drawing.Size(90, 90);
            this.txtLoadButton.TabIndex = 15;
            this.txtLoadButton.UseVisualStyleBackColor = false;
            this.txtLoadButton.Click += new System.EventHandler(this.txtLoadButton_Click);
            // 
            // pdfLoadButton
            // 
            this.pdfLoadButton.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(192)))), ((int)(((byte)(192)))), ((int)(((byte)(255)))));
            this.pdfLoadButton.BackgroundImage = global::AdasVetelClient.Properties.Resources.pdf_icon;
            this.pdfLoadButton.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Stretch;
            this.pdfLoadButton.Cursor = System.Windows.Forms.Cursors.Hand;
            this.pdfLoadButton.FlatAppearance.MouseOverBackColor = System.Drawing.Color.FromArgb(((int)(((byte)(128)))), ((int)(((byte)(128)))), ((int)(((byte)(255)))));
            this.pdfLoadButton.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.pdfLoadButton.Location = new System.Drawing.Point(23, 121);
            this.pdfLoadButton.Name = "pdfLoadButton";
            this.pdfLoadButton.Size = new System.Drawing.Size(90, 90);
            this.pdfLoadButton.TabIndex = 16;
            this.pdfLoadButton.UseVisualStyleBackColor = false;
            this.pdfLoadButton.Click += new System.EventHandler(this.pdfLoadButton_Click);
            // 
            // splitContainer1
            // 
            this.splitContainer1.Dock = System.Windows.Forms.DockStyle.Bottom;
            this.splitContainer1.Location = new System.Drawing.Point(0, 65);
            this.splitContainer1.Name = "splitContainer1";
            // 
            // splitContainer1.Panel1
            // 
            this.splitContainer1.Panel1.Controls.Add(this.fileFlowLayout);
            // 
            // splitContainer1.Panel2
            // 
            this.splitContainer1.Panel2.BackColor = System.Drawing.Color.White;
            this.splitContainer1.Panel2.Controls.Add(this.progressLabel);
            this.splitContainer1.Panel2.Controls.Add(this.progressBar);
            this.splitContainer1.Panel2.Controls.Add(this.panel1);
            this.splitContainer1.Panel2.Controls.Add(this.clearListBtn);
            this.splitContainer1.Panel2.Controls.Add(this.readContract);
            this.splitContainer1.Panel2.Controls.Add(this.viewContracts);
            this.splitContainer1.Panel2.Controls.Add(this.groupBox2);
            this.splitContainer1.Size = new System.Drawing.Size(950, 421);
            this.splitContainer1.SplitterDistance = 489;
            this.splitContainer1.TabIndex = 19;
            // 
            // fileFlowLayout
            // 
            this.fileFlowLayout.AutoScroll = true;
            this.fileFlowLayout.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(192)))), ((int)(((byte)(192)))), ((int)(((byte)(255)))));
            this.fileFlowLayout.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.fileFlowLayout.Dock = System.Windows.Forms.DockStyle.Fill;
            this.fileFlowLayout.Location = new System.Drawing.Point(0, 0);
            this.fileFlowLayout.Name = "fileFlowLayout";
            this.fileFlowLayout.Size = new System.Drawing.Size(489, 421);
            this.fileFlowLayout.TabIndex = 0;
            // 
            // progressLabel
            // 
            this.progressLabel.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.progressLabel.AutoSize = true;
            this.progressLabel.Location = new System.Drawing.Point(15, 304);
            this.progressLabel.Name = "progressLabel";
            this.progressLabel.Size = new System.Drawing.Size(0, 17);
            this.progressLabel.TabIndex = 24;
            // 
            // progressBar
            // 
            this.progressBar.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.progressBar.Location = new System.Drawing.Point(18, 327);
            this.progressBar.Name = "progressBar";
            this.progressBar.Size = new System.Drawing.Size(424, 16);
            this.progressBar.TabIndex = 23;
            // 
            // panel1
            // 
            this.panel1.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.panel1.Controls.Add(this.connectionSignal);
            this.panel1.Controls.Add(this.connectButton);
            this.panel1.Location = new System.Drawing.Point(361, 9);
            this.panel1.Name = "panel1";
            this.panel1.Size = new System.Drawing.Size(84, 55);
            this.panel1.TabIndex = 22;
            // 
            // connectionSignal
            // 
            this.connectionSignal.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.connectionSignal.BackColor = System.Drawing.Color.Red;
            this.connectionSignal.BackgroundImageLayout = System.Windows.Forms.ImageLayout.None;
            this.connectionSignal.Location = new System.Drawing.Point(4, 15);
            this.connectionSignal.Name = "connectionSignal";
            this.connectionSignal.Size = new System.Drawing.Size(23, 22);
            this.connectionSignal.TabIndex = 21;
            this.connectionSignal.TabStop = false;
            // 
            // connectButton
            // 
            this.connectButton.BackColor = System.Drawing.SystemColors.InactiveCaption;
            this.connectButton.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("connectButton.BackgroundImage")));
            this.connectButton.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Stretch;
            this.connectButton.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.connectButton.Location = new System.Drawing.Point(33, 4);
            this.connectButton.Name = "connectButton";
            this.connectButton.Size = new System.Drawing.Size(48, 47);
            this.connectButton.TabIndex = 20;
            this.connectButton.UseVisualStyleBackColor = false;
            this.connectButton.Click += new System.EventHandler(this.connectButton_Click);
            // 
            // clearListBtn
            // 
            this.clearListBtn.BackColor = System.Drawing.SystemColors.InactiveCaption;
            this.clearListBtn.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.clearListBtn.Location = new System.Drawing.Point(18, 18);
            this.clearListBtn.Name = "clearListBtn";
            this.clearListBtn.Size = new System.Drawing.Size(113, 29);
            this.clearListBtn.TabIndex = 19;
            this.clearListBtn.Text = "Lista törlése";
            this.clearListBtn.UseVisualStyleBackColor = false;
            this.clearListBtn.Click += new System.EventHandler(this.clearListBtn_Click);
            // 
            // viewContracts
            // 
            this.viewContracts.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.viewContracts.BackColor = System.Drawing.Color.DarkTurquoise;
            this.viewContracts.Cursor = System.Windows.Forms.Cursors.Hand;
            this.viewContracts.Depth = 0;
            this.viewContracts.FlatAppearance.BorderColor = System.Drawing.Color.Black;
            this.viewContracts.FlatAppearance.BorderSize = 2;
            this.viewContracts.FlatAppearance.MouseOverBackColor = System.Drawing.Color.FromArgb(((int)(((byte)(0)))), ((int)(((byte)(64)))), ((int)(((byte)(64)))));
            this.viewContracts.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.viewContracts.Location = new System.Drawing.Point(248, 349);
            this.viewContracts.MouseState = MaterialSkin.MouseState.HOVER;
            this.viewContracts.Name = "viewContracts";
            this.viewContracts.Primary = true;
            this.viewContracts.Size = new System.Drawing.Size(194, 60);
            this.viewContracts.TabIndex = 13;
            this.viewContracts.Text = "Szerződések megtekintése";
            this.viewContracts.UseVisualStyleBackColor = false;
            this.viewContracts.Click += new System.EventHandler(this.viewContracts_Click);
            // 
            // timer1
            // 
            this.timer1.Interval = 10000;
            this.timer1.Tick += new System.EventHandler(this.timer1_Tick);
            // 
            // limitLabel
            // 
            this.limitLabel.AutoSize = true;
            this.limitLabel.ForeColor = System.Drawing.Color.Red;
            this.limitLabel.Location = new System.Drawing.Point(203, 45);
            this.limitLabel.Name = "limitLabel";
            this.limitLabel.Size = new System.Drawing.Size(258, 17);
            this.limitLabel.TabIndex = 0;
            this.limitLabel.Text = "Maximum 10 elem tölthető be egyszerre";
            this.limitLabel.Visible = false;
            // 
            // userNameLabel
            // 
            this.userNameLabel.AutoSize = true;
            this.userNameLabel.BackColor = System.Drawing.SystemColors.Desktop;
            this.userNameLabel.Depth = 0;
            this.userNameLabel.Font = new System.Drawing.Font("Microsoft Sans Serif", 11F, System.Drawing.FontStyle.Bold);
            this.userNameLabel.ForeColor = System.Drawing.Color.White;
            this.userNameLabel.Location = new System.Drawing.Point(714, 31);
            this.userNameLabel.MouseState = MaterialSkin.MouseState.HOVER;
            this.userNameLabel.Name = "userNameLabel";
            this.userNameLabel.Size = new System.Drawing.Size(0, 24);
            this.userNameLabel.TabIndex = 20;
            // 
            // AdasVetel
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 16F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(950, 486);
            this.Controls.Add(this.userNameLabel);
            this.Controls.Add(this.limitLabel);
            this.Controls.Add(this.splitContainer1);
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.MinimumSize = new System.Drawing.Size(853, 486);
            this.Name = "AdasVetel";
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
            this.Text = "Adás Vétel";
            this.Load += new System.EventHandler(this.AdasVetel_Load);
            this.groupBox2.ResumeLayout(false);
            this.splitContainer1.Panel1.ResumeLayout(false);
            this.splitContainer1.Panel2.ResumeLayout(false);
            this.splitContainer1.Panel2.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer1)).EndInit();
            this.splitContainer1.ResumeLayout(false);
            this.panel1.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.connectionSignal)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion
        private MaterialSkin.Controls.MaterialRaisedButton readContract;
        private MaterialSkin.Controls.MaterialRaisedButton viewContracts;
        private System.Windows.Forms.Button docLoadButton;
        private System.Windows.Forms.Button txtLoadButton;
        private System.Windows.Forms.Button pdfLoadButton;
        private System.Windows.Forms.Button imageLoadButton;
        private System.Windows.Forms.GroupBox groupBox2;
        private System.Windows.Forms.SplitContainer splitContainer1;
        private System.Windows.Forms.FlowLayoutPanel fileFlowLayout;
        private System.Windows.Forms.Button clearListBtn;
        private System.Windows.Forms.Button connectButton;
        private System.Windows.Forms.PictureBox connectionSignal;
        private System.Windows.Forms.Timer timer1;
        private System.Windows.Forms.Label limitLabel;
        private System.Windows.Forms.Panel panel1;
        private System.Windows.Forms.Label progressLabel;
        private System.Windows.Forms.ProgressBar progressBar;
        private MaterialSkin.Controls.MaterialLabel userNameLabel;
    }
}

