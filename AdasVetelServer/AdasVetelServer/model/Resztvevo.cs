using System.Collections.Generic;

namespace AdasVetelServer.model
{

    public abstract class Resztvevo : DbElement
    {

        public Resztvevo() : base()
        {
            Labels.Add("Résztvevő Típus");
        }


        public string ResztvevoTipus { get; set; } = "Ismeretlen";
        public override List<string> getData()
        {
            List<string> data = base.getData();
            data.Add(ResztvevoTipus.ToString());
            return data;
        }
    }
}
