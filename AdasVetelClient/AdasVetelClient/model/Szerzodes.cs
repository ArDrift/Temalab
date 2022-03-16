
using System;
using System.Collections.Generic;
using System.Data.Linq;
using System.Data.Linq.Mapping;

namespace AdasVetelServer.model
{
    [Table(Name = "dbo.Szerződés")]
    public class Szerzodes : DbElement, ICheckable<Szerzodes>
    {

        private readonly EntitySet<SzerzodesTargy> szerzodesTargyak = new EntitySet<SzerzodesTargy>();
        [Association(Name = "szerződésTárgyak", OtherKey = "SzerzodesAz")]
        public EntitySet<SzerzodesTargy> Szerzodestargyak
        {
            get { return szerzodesTargyak; }
            set { szerzodesTargyak.Assign(value); }
        }

        public Szerzodes() : base()
        {
            Labels.Add("Kelt dátum");
            Labels.Add("Típus");
            Labels.Add("Ellenjegyző iroda");


        }
        [Column(IsPrimaryKey = true, Name = "az", IsDbGenerated = true, DbType = "Int NOT NULL IDENTITY")]
        public int Az { get { return az; } set { az = value; } }

        [Column(Name = "létrehozás_dátum")]
        public DateTime? LetrehozasDatum { get { return letrehozasdatum; } set { letrehozasdatum = value; } }


        [Column(Name = "kelt_dátum", CanBeNull = true)]
        public DateTime? KeltDatum { get; set; } = null;
        [Column(Name = "típus")]
        public string Tipus { get; set; } = "Ismeretlen";
        [Column(Name = "ellenjegyző_iroda")]
        public string EllenjegyzoIroda { get; set; } = "Ismeretlen";
        [Column(Name = "fájl_név")]
        public string FajlNev { get; set; } = "Ismeretlen";

        public bool equals(Szerzodes element)
        {

            return KeltDatum.ToString() == element.KeltDatum.ToString() &&
                Tipus == element.Tipus &&
                 EllenjegyzoIroda == element.EllenjegyzoIroda &&
                  FajlNev == element.FajlNev;
        }

        public override List<string> getData()
        {
            List<string> data = base.getData();

            data.Add(KeltDatum.ToString());
            data.Add(Tipus.ToString());
            data.Add(EllenjegyzoIroda.ToString());

            return data;
        }
    }
}
