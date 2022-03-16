
namespace AdasVetelClient
{
    partial class FileIconControl
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

        #region Component Designer generated code

        /// <summary> 
        /// Required method for Designer support - do not modify 
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.fileLabel = new System.Windows.Forms.TextBox();
            this.fileButton = new System.Windows.Forms.Button();
            this.SuspendLayout();
            // 
            // fileLabel
            // 
            this.fileLabel.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(192)))), ((int)(((byte)(192)))), ((int)(((byte)(255)))));
            this.fileLabel.BorderStyle = System.Windows.Forms.BorderStyle.None;
            this.fileLabel.Dock = System.Windows.Forms.DockStyle.Fill;
            this.fileLabel.Location = new System.Drawing.Point(0, 74);
            this.fileLabel.Multiline = true;
            this.fileLabel.Name = "fileLabel";
            this.fileLabel.ReadOnly = true;
            this.fileLabel.Size = new System.Drawing.Size(85, 93);
            this.fileLabel.TabIndex = 2;
            // 
            // fileButton
            // 
            this.fileButton.BackColor = System.Drawing.SystemColors.Control;
            this.fileButton.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Stretch;
            this.fileButton.Dock = System.Windows.Forms.DockStyle.Top;
            this.fileButton.FlatAppearance.BorderColor = System.Drawing.Color.White;
            this.fileButton.FlatAppearance.BorderSize = 2;
            this.fileButton.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.fileButton.Location = new System.Drawing.Point(0, 0);
            this.fileButton.Name = "fileButton";
            this.fileButton.Size = new System.Drawing.Size(85, 74);
            this.fileButton.TabIndex = 1;
            this.fileButton.UseVisualStyleBackColor = false;
            this.fileButton.Click += new System.EventHandler(this.fileButton_Click);
            // 
            // FileIconControl
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 16F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.Controls.Add(this.fileLabel);
            this.Controls.Add(this.fileButton);
            this.Margin = new System.Windows.Forms.Padding(8);
            this.Name = "FileIconControl";
            this.Size = new System.Drawing.Size(85, 167);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion
        private System.Windows.Forms.TextBox fileLabel;
        private System.Windows.Forms.Button fileButton;
    }
}
