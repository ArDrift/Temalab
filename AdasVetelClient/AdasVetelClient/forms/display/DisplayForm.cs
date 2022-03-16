
using AdasVetelClient.layout;
using AdasVetelClient.tcp;
using AdasVetelServer.messages;
using AdasVetelServer.model;
using System;
using System.Collections.Generic;
using System.Windows.Forms;

namespace AdasVetelClient
{

    public partial class DisplayForm : Form, ClientView
    {

        protected List<MyDataGridView> dataGrids = new List<MyDataGridView>();
        protected MyDataGridView currentGrid;
        protected List<DataGridViewRow> currentRows = new List<DataGridViewRow>();
        protected DataGridViewRow clickedRow;
        protected string Route = "";
        protected string Label = "";
        public DisplayForm(DataGridViewRow clickedRow = null)
        {

            InitializeComponent();
            CenterToScreen();
            this.clickedRow = clickedRow;
            currentGrid = null;

            listViewItem1.Tag = "cimek";
            listViewItem2.Tag = "szerzodesek";
            listViewItem3.Tag = "szemelyek";
            listViewItem4.Tag = "szervezetek";
            listViewItem5.Tag = "ingatlanok";
            listViewItem6.Tag = "gepjarmuvek";
            listViewItem7.Tag = "ingok";
            listViewItem8.Tag = "teljesitesek";
            listViewItem9.Tag = "szerzodestargyak";
            listViewItem10.Tag = "reszvetelek";
            createGridViews();
            TcpClientWrapper.Instance.views.Add(this);


        }

        public DisplayForm()
        {
            InitializeComponent();
        }
        private void DisplayForm_Load(object sender, EventArgs e)
        {
        }
        virtual protected void createGridViews()
        {
            listView1.Items.AddRange(new ListViewItem[] {
            listViewItem1,
            listViewItem2,
            listViewItem3,
            listViewItem4,
            listViewItem5,
            listViewItem6,
            listViewItem7,
            listViewItem8,
            listViewItem9,
            listViewItem10});
            dataGrids.Add(new DbHolderGridView<Cim>((string)listViewItem1.Tag));
            dataGrids.Add(new DbHolderGridView<Szerzodes>((string)listViewItem2.Tag));
            dataGrids.Add(new DbHolderGridView<Szemely>((string)listViewItem3.Tag));
            dataGrids.Add(new DbHolderGridView<Szervezet>((string)listViewItem4.Tag));
            dataGrids.Add(new DbHolderGridView<Ingatlan>((string)listViewItem5.Tag));
            dataGrids.Add(new DbHolderGridView<Gepjarmu>((string)listViewItem6.Tag));
            dataGrids.Add(new DbHolderGridView<Ingo>((string)listViewItem7.Tag));
            dataGrids.Add(new DbHolderGridView<Teljesites>((string)listViewItem8.Tag));
            dataGrids.Add(new DbHolderGridView<SzerzodesTargy>((string)listViewItem9.Tag));
            dataGrids.Add(new DbHolderGridView<Reszvetel>((string)listViewItem10.Tag));
        }


        protected void listView1_DoubleClick(object sender, EventArgs e)
        {
            if (TcpClientWrapper.Instance.isConnected() && listView1.SelectedItems.Count > 0)
            {
                splitContainer2.Panel2.Controls.Remove(currentGrid);
                Label = (string)listView1.SelectedItems[0].Tag;
                foreach (var grid in dataGrids)
                {
                    if (listView1.SelectedItems.Count > 0 && Label == grid.Id)
                    {

                        splitContainer2.Panel2.Controls.Add(grid);
                        currentGrid = grid;

                        break;
                    }
                }
                GetRecordMessage msg = new GetRecordMessage($"{Route}/{Label}", new byte[0], 0, "");
                TcpClientWrapper.Instance.send<GetRecordMessage>(msg);
            }

        }

        private void searchButton_Click(object sender, EventArgs e)
        {
            searchLogic();
        }
        private void searchBox_TextChanged(object sender, EventArgs e)
        {
            searchLogic();
        }
        private void searchLogic()
        {
            if (currentGrid != null)
            {

                currentGrid.Rows.Clear();

                if (searchBox.Text == "")
                {
                    currentGrid.Rows.AddRange(currentRows.ToArray());
                }
                else
                {
                    foreach (var item in currentRows)
                    {
                        bool valid = false;
                        foreach (DataGridViewCell data in item.Cells)
                        {

                            if (data.Value.ToString().ToLower().Contains(searchBox.Text.ToLower()))
                            {
                                valid = true;
                                break;
                            }
                        }
                        if (valid)
                            currentGrid.Rows.Add(item);
                    }
                }
                currentGrid.Refresh();
            }
        }

        public void connected()
        { }
        public bool isActive()
        {
            return !IsDisposed;
        }

        public void clientLoggedIn() { }
        public void disconnected()
        { }
        public void handleMessage(MessageBase message, string type)
        {
            Invoke((MethodInvoker)delegate
            {
                if (type == "RecordRequest")
                {
                    GetRecordMessage grmsg = (GetRecordMessage)message;
                    string baseRoute = grmsg.Route.Substring(0, grmsg.Route.LastIndexOf('/'));
                    if (Route == baseRoute)
                    {
                        if (grmsg.Index == 0)
                        {
                            currentGrid.Rows.Clear();
                            currentRows.Clear();
                        }
                        else
                        {
                            currentGrid.AddToGrid(grmsg.Record);
                            currentRows.Add(currentGrid.Rows[currentGrid.Rows.Count - 1]);
                        }
                    }
                }
                else if (type == "DataChanged")
                {
                    GetRecordMessage msg = new GetRecordMessage($"{Route}/{Label}", new byte[0], 0, "");
                    TcpClientWrapper.Instance.send<GetRecordMessage>(msg);
                }
            });
        }
    }
}
