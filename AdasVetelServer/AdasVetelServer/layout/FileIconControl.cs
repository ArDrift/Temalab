using AdasVetelServer.logic.nep;
using DocumentFormat.OpenXml.Packaging;
using DocumentFormat.OpenXml.Wordprocessing;
using System;
using System.Drawing;
using System.IO;
using System.Windows.Forms;

namespace AdasVetelServer
{
    public delegate void KeySetChanged();
    public partial class FileIconControl : UserControl
    {
        public event KeySetChanged OnKeySetChanged;

        private bool keySet = false;
        private readonly string fileName;
        public int SelectionIndex { get; set; } = 0;
        public string KeyString { get; set; } = "";
        public string ContextBefore { get; set; } = "";
        public string ContextAfter { get; set; } = "";
        public bool KeySet
        {
            get
            {
                return keySet;
            }
            set
            {
                keySet = value;
                OnKeySetChanged?.Invoke();
                if (keySet)
                    fileButton.FlatAppearance.BorderColor = System.Drawing.Color.Lime;
                else
                    fileButton.FlatAppearance.BorderColor = System.Drawing.Color.Red;

            }
        }
        public bool KeyCounts { get; set; } = false;
        public bool BeforeContextCounts { get; set; } = true;
        public bool AfterContextCounts { get; set; } = true;

        public FileIconControl(string filename)
        {
            InitializeComponent();
            fileLabel.Text = Path.GetFileName(filename);
            fileName = filename;
            fileButton.BackgroundImage = new Bitmap(fileButton.BackgroundImage, new Size(fileButton.Width, fileButton.Height));
        }

        public void setContext(string ca, string cb, string key, int index)
        {
            ContextAfter = ca;
            ContextBefore = cb;
            KeyString = key;
            SelectionIndex = index;
            KeySet = true;

        }
        private string readWordDoc(string filename)
        {
            try
            {
                using (WordprocessingDocument wordDocument =
                WordprocessingDocument.Open(filename, false))
                {
                    // Assign a reference to the existing document body.  
                    string data = "";
                    var paragraphs = wordDocument.MainDocumentPart.RootElement.Descendants<Paragraph>();
                    foreach (var paragraph in paragraphs)
                    {
                        data += paragraph.InnerText + " \n ";
                    }
                    return data;
                }
            }
            catch (Exception)
            {
                string message = $"A {filename} útvonalú fájl nem található vagy meg van nyitva egy másik program álltal, így a beolvasás nem hajtható végre!";
                // Show message box
                MessageBox.Show(message);
                return "";
            }
        }
        public void train(string xmlFile)
        {
            if (!(KeyString == "" && ContextBefore == "" && ContextAfter == ""))
            {

                EntityPattern entityP = EntityPattern.load(xmlFile);

                if (entityP.OnlyPattern)
                {
                    string CA = $"(?=({ContextAfter}))";
                    string CB = $"(?<=({ContextBefore}))";
                    string pattern;
                    string div;
                    double keySize = KeyString.Split(' ').Length;
                    if (KeyCounts)
                    {
                        pattern = KeyString;
                        div = " ";
                    }
                    else
                    {
                        pattern = $" ([^ ]+ ){{{(int)(keySize / 2)},{(int)(keySize * 3 / 2)}}}?";
                        div = "";
                    }
                    if (AfterContextCounts)
                        pattern = $"{pattern}{div}{CA}";
                    if (BeforeContextCounts)
                        pattern = $"{CB}{div}{pattern}";
                    entityP.Patterns.Add(pattern);
                }
                else
                {
                    if (KeyCounts)
                    {
                        entityP.Patterns.Add(KeyString);
                    }
                    if (AfterContextCounts)
                        entityP.ContextsAfter.Add(ContextAfter);
                    if (BeforeContextCounts)
                        entityP.ContextsBefore.Add(ContextBefore);
                }

                entityP.save(xmlFile);
            }
        }
    }
}
