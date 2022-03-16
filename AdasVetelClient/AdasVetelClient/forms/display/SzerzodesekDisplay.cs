
using AdasVetelClient.layout;
using AdasVetelServer.model;
using System.Windows.Forms;


namespace AdasVetelClient.display
{
    public partial class SzerzodesekDisplay : DisplayForm
    {

        public SzerzodesekDisplay(DataGridViewRow clickedRow) : base(clickedRow)
        {
            Szerzodes sz = (Szerzodes)clickedRow.Tag;
            InitializeComponent();
            Text = $"{sz.Az} - {sz.Tipus} : {sz.KeltDatum}";
            Route = $"{Route}/szerzodesek/{sz.Az}";
        }
        protected override void createGridViews()
        {
            listView1.Items.AddRange(new ListViewItem[] {
            listViewItem3,
            listViewItem4,
            listViewItem10,
            listViewItem5,
            listViewItem6,
            listViewItem7,
            listViewItem8,
            listViewItem9
            });
            dataGrids.Add(new DbHolderGridView<Szemely>((string)listViewItem3.Tag));
            dataGrids.Add(new DbHolderGridView<Szervezet>((string)listViewItem4.Tag));
            dataGrids.Add(new DbHolderGridView<Reszvetel>((string)listViewItem10.Tag));
            dataGrids.Add(new DbHolderGridView<Ingatlan>((string)listViewItem5.Tag));
            dataGrids.Add(new DbHolderGridView<Gepjarmu>((string)listViewItem6.Tag));
            dataGrids.Add(new DbHolderGridView<Ingo>((string)listViewItem7.Tag));
            dataGrids.Add(new DbHolderGridView<Teljesites>((string)listViewItem8.Tag));
            dataGrids.Add(new DbHolderGridView<SzerzodesTargy>((string)listViewItem9.Tag));
        }


    }
}
