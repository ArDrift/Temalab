using AdasVetelClient.display;
using System.Drawing;
using System.Windows.Forms;


namespace AdasVetelClient.layout
{
    public class MyDataGridView : DataGridView
    {
        public string Id { get; set; }

        public MyDataGridView() : base()
        {

            InitializeComponent();

            ColumnHeadersDefaultCellStyle.BackColor = Color.Navy;
            ColumnHeadersDefaultCellStyle.ForeColor = Color.White;
            ColumnHeadersDefaultCellStyle.Font =
                new Font(Font, FontStyle.Bold);

            Name = "dataView";
            Location = new Point(8, 8);
            Size = new Size(500, 250);

            ColumnHeadersBorderStyle =
                DataGridViewHeaderBorderStyle.Single;
            CellBorderStyle = DataGridViewCellBorderStyle.Single;
            GridColor = Color.Black;
            RowHeadersVisible = false;

            SelectionMode =
                DataGridViewSelectionMode.FullRowSelect;
            MultiSelect = false;
            Dock = DockStyle.Fill;

        }

        private void InitializeComponent()
        {
            ((System.ComponentModel.ISupportInitialize)(this)).BeginInit();
            this.SuspendLayout();
            // 
            // MyDataGridView
            // 
            this.AllowUserToOrderColumns = true;
            this.AutoSizeColumnsMode = System.Windows.Forms.DataGridViewAutoSizeColumnsMode.AllCells;
            this.AutoSizeRowsMode = System.Windows.Forms.DataGridViewAutoSizeRowsMode.AllCells;
            this.Dock = System.Windows.Forms.DockStyle.Fill;
            this.RowTemplate.Height = 35;
            this.CellDoubleClick += new System.Windows.Forms.DataGridViewCellEventHandler(this.MyDataGridView_CellDoubleClick);
            ((System.ComponentModel.ISupportInitialize)(this)).EndInit();
            this.ResumeLayout(false);

        }

        private void MyDataGridView_CellDoubleClick(object sender, DataGridViewCellEventArgs e)
        {
            if (Id == "szemelyek")
                new SzemelyekDisplay(SelectedRows[0]).Show();
            else if (Id == "szervezetek")
                new SzervezetekDisplay(SelectedRows[0]).Show();
            else if (Id == "szerzodesek")
                new SzerzodesekDisplay(SelectedRows[0]).Show();

        }

        public virtual void FillGrid(byte[] content) { }
        public virtual void AddToGrid(byte[] content) { }

    }
}
