using System.Collections.Generic;

namespace AdasVetelServer.model
{
    public abstract class Targy : DbElement
    {
        public Targy() : base()
        {
            Labels.Add("Targy Típus");

        }
        public string TargyTipus { get; set; } = "Ismeretlen";
        public override List<string> getData()
        {
            List<string> data = base.getData();
            data.Add(TargyTipus.ToString());
            return data;
        }
    }
}
