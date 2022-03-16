using DocumentFormat.OpenXml.Packaging;
using DocumentFormat.OpenXml.Wordprocessing;
using IronOcr;
using System;
using System.Drawing;
using System.IO;
using System.Text;
using System.Windows.Forms;

namespace AdasVetelClient
{
    public delegate string GetText(string fileName);
    public partial class FileIconControl : UserControl
    {
        private GetText getTextDelegate;
        private IronTesseract ocr = null;
        private readonly ControlCollection parent;
        public IronTesseract Ocr
        {
            get
            {
                if (ocr == null)
                {
                    ocr = new IronTesseract();
                    // Configure for speed.  35% faster and only 0.2% loss of accuracy
                    ocr.Configuration.BlackListCharacters = "~`$#^*_}{][|\\@¢©«»°±·×‑–—‘’“”•…′″€™←↑→↓↔⇄⇒∅∼≅≈≠≤≥≪≫⌁⌘○◔◑◕●☐☑☒☕☮☯☺♡⚓✓✰";
                    ocr.Configuration.PageSegmentationMode = TesseractPageSegmentationMode.Auto;
                    ocr.Configuration.TesseractVersion = TesseractVersion.Tesseract5;
                    ocr.Configuration.EngineMode = TesseractEngineMode.LstmOnly;
                    ocr.Configuration.ReadBarCodes = false;
                    ocr.Language = OcrLanguage.HungarianFast;

                }
                return ocr;
            }
        }
        public string FileName { get; set; }
        public string SafeFileName { get; set; }


        public FileIconControl(string filename, string safeFilename, ControlCollection parent)
        {
            InitializeComponent();
            this.parent = parent;
            FileName = filename;
            SafeFileName = safeFilename;
            fileLabel.Text = safeFilename;
            setGetTextDelegate();
            fileButton.BackgroundImage = new Bitmap(fileButton.BackgroundImage, new Size(fileButton.Width, fileButton.Height));
        }

        public string getText()
        {
            return getTextDelegate(FileName);
        }
        private void setGetTextDelegate()
        {
            string extension = Path.GetExtension(FileName).ToLower();
            if (extension == ".txt")
            {
                getTextDelegate = readTxt;
                fileButton.BackgroundImage = Properties.Resources.txt_icon;
            }
            else if (extension == ".docx" || extension == ".doc")
            {
                getTextDelegate = readWordDoc;
                fileButton.BackgroundImage = Properties.Resources.doc_icon;
            }
            else if (extension == ".pdf")
            {
                getTextDelegate = readPdf;
                fileButton.BackgroundImage = Properties.Resources.pdf_icon;
            }
            else if (extension == ".png" || extension == ".jpeg" || extension == ".jpg")
            {
                getTextDelegate = readImage;
                fileButton.BackgroundImage = Properties.Resources.image_icon;
            }
        }

        private string readPdf(string filename)
        {
            using (var Input = new OcrInput(filename))
            {
                var Result = Ocr.Read(Input);
                return Result.Text;
            }

        }
        private string readImage(string name)
        {
            using (var Input = new OcrInput(name))
            {
                var Result = Ocr.Read(Input);
                return Result.Text;
            }

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
                string message = $"A {SafeFileName} nevű fájl meg van nyitva egy másik program álltal, így a beolvasás nem hajtható végre!";
                // Show message box
                MessageBox.Show(message);
                return "";
            }

        }
        private string readTxt(string filename)
        {
            string data = "";
            var lines = File.ReadAllLines(filename, Encoding.UTF8);
            foreach (var line in lines)
            {
                data += line + " \n ";
            }

            return data;

        }

        private void fileButton_Click(object sender, EventArgs e)
        {
            parent.Remove(this);
        }
    }
}
