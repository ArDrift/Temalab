using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
/*
namespace AdasVetelServer.analyzing.ner
{
    
    public class NerController
    {
        public Ner MyNer { get; set; }
        public Predictor MyPredictor { get; set; }
        public string Text { get; set; }
        public string TextToPredict { get; set; }
        public string Output { get; set; }
        public string OutputFileName { get; set; }
       
        public NerController(Ner myner,string text,string outPutFileName)
        {
            MyNer = myner;
            MyPredictor = new Predictor(MyNer.BuildAndTrain("model\\model.onnx", false));
            Text = text;
            TextToPredict = "";
            Output = "[";
            OutputFileName = outPutFileName;
        }

        public void writeToFile()
        {
            for (int i = 0; i < Text.Length; i++)
            {
                TextToPredict += Text[i] + " ";
                if (i == Text.Length - 1 || i + 1 % 500 == 0)
                {
                    string result = MyPredictor.Predict(TextToPredict.TrimEnd(' '));
                    TextToPredict = "";
                    Output += result.Trim('[').Trim(']').Trim(' ') + ',';
                }
            }
            Output = Output.TrimStart(',')+']';
            File.WriteAllText(OutputFileName, Output);
           
        }
    }
}
*/