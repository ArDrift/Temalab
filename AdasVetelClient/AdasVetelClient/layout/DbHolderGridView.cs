using AdasVetelServer.model;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Text.Json;

using System.Windows.Forms;

namespace AdasVetelClient.layout
{
    public class DbHolderGridView<T> : MyDataGridView where T : DbElement, new()
    {

        private List<T> data = new List<T>();


        public DbHolderGridView(string id) : base()
        {
            Id = id;
            InitializeComponent();

            ColumnHeadersDefaultCellStyle.BackColor = Color.Navy;
            ColumnHeadersDefaultCellStyle.ForeColor = Color.White;
            ColumnHeadersDefaultCellStyle.Font =
                new Font(Font, FontStyle.Bold);

            Name = "dataView2";
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

            T obj = new T();
            ColumnCount = obj.Labels.Count;
            for (int i = 0; i < ColumnCount; i++)
            {
                Columns[i].Name = obj.Labels[i];
            }

        }

        private void InitializeComponent()
        {
            ((System.ComponentModel.ISupportInitialize)(this)).BeginInit();
            this.SuspendLayout();
            // 
            // DbHolderGridView
            // 
            this.AllowUserToAddRows = false;
            this.AllowUserToDeleteRows = false;
            this.RowTemplate.Height = 35;
            ((System.ComponentModel.ISupportInitialize)(this)).EndInit();
            this.ResumeLayout(false);

        }
        public override void FillGrid(byte[] content)
        {
            data = JsonSerializer.Deserialize<List<T>>(new ReadOnlySpan<byte>(content));
            Rows.Clear();

            for (int i = 0; i < data.Count; i++)
            {
                Rows.Add(data[i].getData().ToArray());
                Rows[i].Tag = data[i];
            }
            Refresh();
        }
        public override void AddToGrid(byte[] content)
        {
            T data = JsonSerializer.Deserialize<T>(new ReadOnlySpan<byte>(content));
            Rows.Add(data.getData().ToArray());
            Rows[Rows.Count - 1].Tag = data;
        }
    }
}
