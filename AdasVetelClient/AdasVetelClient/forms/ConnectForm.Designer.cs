
namespace AdasVetelClient
{
    partial class ConnectForm
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
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(ConnectForm));
            this.connectionProgress = new System.Windows.Forms.ProgressBar();
            this.closeBtn = new MaterialSkin.Controls.MaterialRaisedButton();
            this.SuspendLayout();
            // 
            // connectionProgress
            // 
            this.connectionProgress.Location = new System.Drawing.Point(87, 97);
            this.connectionProgress.MarqueeAnimationSpeed = 25;
            this.connectionProgress.Name = "connectionProgress";
            this.connectionProgress.Size = new System.Drawing.Size(251, 43);
            this.connectionProgress.Style = System.Windows.Forms.ProgressBarStyle.Marquee;
            this.connectionProgress.TabIndex = 0;
            // 
            // closeBtn
            // 
            this.closeBtn.Cursor = System.Windows.Forms.Cursors.Hand;
            this.closeBtn.Depth = 0;
            this.closeBtn.Location = new System.Drawing.Point(284, 168);
            this.closeBtn.MouseState = MaterialSkin.MouseState.HOVER;
            this.closeBtn.Name = "closeBtn";
            this.closeBtn.Primary = true;
            this.closeBtn.Size = new System.Drawing.Size(158, 34);
            this.closeBtn.TabIndex = 4;
            this.closeBtn.Text = "Offline mód";
            this.closeBtn.UseVisualStyleBackColor = true;
            this.closeBtn.Click += new System.EventHandler(this.closeBtn_Click);
            // 
            // ConnectForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 16F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(454, 214);
            this.Controls.Add(this.closeBtn);
            this.Controls.Add(this.connectionProgress);
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.ImeMode = System.Windows.Forms.ImeMode.Disable;
            this.Name = "ConnectForm";
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
            this.Text = "Csatlakozás a szerverhez";
            this.Load += new System.EventHandler(this.ConnectForm_Load);
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.ProgressBar connectionProgress;
        private MaterialSkin.Controls.MaterialRaisedButton closeBtn;
    }
}