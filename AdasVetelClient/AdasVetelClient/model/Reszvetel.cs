using System;
using System.Collections.Generic;
using System.Data.Linq;
using System.Data.Linq.Mapping;

namespace AdasVetelServer.model
{
    [Table(Name = "dbo.Részvétel")]
    public class Reszvetel : DbElement, ICheckable<Reszvetel>
    {
        private EntityRef<Szerzodes> szerzodes = new EntityRef<Szerzodes>();
        private EntityRef<Resztvevo> resztvevo = new EntityRef<Resztvevo>();
        public Reszvetel() : base()
        {
            Labels.Add("Szerződés az");
            Labels.Add("Személy az");
            Labels.Add("Szervezet az");
            Labels.Add("Résztvevő Típus");
            Labels.Add("Szerep");


        }
        [Column(IsPrimaryKey = true, Name = "az", IsDbGenerated = true, DbType = "Int NOT NULL IDENTITY")]
        public int Az { get { return az; } set { az = value; } }

        [Column(Name = "létrehozás_dátum")]
        public DateTime? LetrehozasDatum { get { return letrehozasdatum; } set { letrehozasdatum = value; } }



        [Column(Name = "résztvevő_kód")]
        public string ResztvevoKod { get; set; } = "Ismeretlen";

        [Column(Name = "szerep")]
        public string Szerep { get; set; } = "Ismeretlen";



        [Column(Name = "szerződés_az", CanBeNull = true, DbType = "int NULL")]
        public int? SzerzodesAz { get; set; } = null;
        [Association(Name = "szerződés", ThisKey = "SzerzodesAz", IsForeignKey = true, DeleteOnNull = false)]
        public Szerzodes Szerzodes { get { return szerzodes.Entity; } set { szerzodes.Entity = value; } }



        [Column(Name = "személy_az", CanBeNull = true, DbType = "int NULL")]
        public int? SzemelyAz { get; set; } = null;
        [Column(Name = "szervezet_az", CanBeNull = true, DbType = "int NULL")]
        public int? SzervezetAz { get; set; } = null;
        [Association(Name = "szemely", ThisKey = "SzemelyAz", IsForeignKey = true, DeleteOnNull = false)]
        public Szemely Szemely { get { return resztvevo.Entity as Szemely; } set { resztvevo.Entity = value; } }
        [Association(Name = "szervezet", ThisKey = "SzervezetAz", IsForeignKey = true, DeleteOnNull = false)]
        public Szervezet Szervezet { get { return resztvevo.Entity as Szervezet; } set { resztvevo.Entity = value; } }




        public bool equals(Reszvetel element)
        {
            return SzerzodesAz == element.SzerzodesAz &&
                SzemelyAz == element.SzemelyAz &&
                 SzervezetAz == element.SzervezetAz &&
               ResztvevoKod == element.ResztvevoKod &&
               Szerep == element.Szerep;
        }

        public override List<string> getData()
        {
            List<string> data = base.getData();

            data.Add(SzerzodesAz.ToString());
            data.Add(SzemelyAz.ToString());
            data.Add(SzervezetAz.ToString());
            data.Add(ResztvevoKod);
            data.Add(Szerep);

            return data;
        }
    }
}
