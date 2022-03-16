using System;

namespace AdasVetelClient
{
    public partial class LoadingForm : MaterialSkin.Controls.MaterialForm
    {
        private int setCount;
        public LoadingForm(int fileCount)
        {
            InitializeComponent();
            setCount = fileCount * 2;
            progressBar1.Step = progressBar1.Maximum / (fileCount * 2);
        }

        public void setLabelText(string text)
        {
            progressLabel.Text = text;
        }

        public void setProgress(string text)
        {
            setCount--;
            progressBar1.PerformStep();
            progressLabel.Text = text + "...";
            progressLabel.Font = new System.Drawing.Font(progressLabel.Font.FontFamily, 9);
            if (setCount <= 0)
            {
                progressBar1.Value = progressBar1.Maximum;
                progressLabel.Text = "Beolvasás kész!";
                closeBtn.Text = "Ok";
            }
        }
        public void stepBack()
        {
            setCount++;
            progressBar1.Value -= progressBar1.Value - progressBar1.Step;
        }

        private void closeBtn_Click(object sender, EventArgs e)
        {
            Close();
        }
    }
}
