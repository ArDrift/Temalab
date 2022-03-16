using System;
using System.Collections.Generic;
using System.Data.Linq.Mapping;

namespace AdasVetelServer.model
{
    [Table(Name = "dbo.Teljesítés")]
    public class Teljesites : DbElement, ICheckable<Teljesites>
    {
        public Teljesites() : base()
        {
            Labels.Add("Vételár");
            Labels.Add("Áfa");
            Labels.Add("Banki átutalás-e");
            Labels.Add("Részteljesítés");
            Labels.Add("Határidő");


        }
        [Column(IsPrimaryKey = true, Name = "az", IsDbGenerated = true, DbType = "Int NOT NULL IDENTITY")]
        public int Az { get { return az; } set { az = value; } }

        [Column(Name = "létrehozás_dátum")]
        public DateTime? LetrehozasDatum { get { return letrehozasdatum; } set { letrehozasdatum = value; } }


        [Column(Name = "vételár")]
        public float Vetelar { get; set; } = -1;
        [Column(Name = "áfa")]
        public string Afa { get; set; } = "Ismeretlen";
        [Column(Name = "banki_átutalás_e")]
        public bool BankiAtutalasE { get; set; } = false;
        [Column(Name = "részteljesítés")]
        public int Reszteljesites { get; set; } = -1;
        [Column(Name = "határidő", CanBeNull = true)]
        public DateTime? Hatarido { get; set; } = null;

        public bool equals(Teljesites element)
        {
            return Vetelar == element.Vetelar &&
                Afa == element.Afa &&
                BankiAtutalasE == element.BankiAtutalasE &&
                Reszteljesites == element.Reszteljesites &&
               Hatarido.ToString() == element.Hatarido.ToString();
        }

        public override List<string> getData()
        {
            List<string> data = base.getData();

            data.Add(Vetelar.ToString("r"));
            data.Add(Afa.ToString());
            data.Add(BankiAtutalasE.ToString());
            data.Add(Reszteljesites.ToString());
            data.Add(Hatarido.ToString());
            return data;
        }
    }
}
