using System;
using System.Collections.Generic;
using System.Data.Linq;
using System.Data.Linq.Mapping;

namespace AdasVetelServer.model
{
    [Table(Name = "SzerződésTárgy")]
    public class SzerzodesTargy : DbElement, ICheckable<SzerzodesTargy>
    {
        private EntityRef<Szerzodes> szerzodes = new EntityRef<Szerzodes>();
        private EntityRef<Teljesites> teljesites = new EntityRef<Teljesites>();
        private EntityRef<Targy> targy = new EntityRef<Targy>();

        public SzerzodesTargy() : base()
        {
            Labels.Add("Ingatlan az");
            Labels.Add("Ingó az");
            Labels.Add("Gépjármű az");
            Labels.Add("Szerződés az");
            Labels.Add("Eladó tulajdon arány");
            Labels.Add("Teljesítés az");


        }
        [Column(IsPrimaryKey = true, Name = "az", IsDbGenerated = true, DbType = "Int NOT NULL IDENTITY")]
        public int Az { get { return az; } set { az = value; } }

        [Column(Name = "létrehozás_dátum")]
        public DateTime? LetrehozasDatum { get { return letrehozasdatum; } set { letrehozasdatum = value; } }



        [Column(Name = "tárgy_kód")]
        public string TargyKod { get; set; } = "Ismeretlen";


        [Column(Name = "eladó_tulajdon_arány")]
        public string EladoTulajdonArany { get; set; } = "100%";



        [Column(Name = "szerződés_az", CanBeNull = true, DbType = "int NULL")]
        public int? SzerzodesAz { get; set; } = null;
        [Association(Name = "szerződés", ThisKey = "SzerzodesAz", IsForeignKey = true, DeleteOnNull = false)]
        public Szerzodes Szerzodes { get { return szerzodes.Entity; } set { szerzodes.Entity = value; } }


        [Column(Name = "teljesítés_az", CanBeNull = true, DbType = "int NULL")]
        public int? TeljesitesAz { get; set; } = null;
        [Association(Name = "teljesítés", ThisKey = "TeljesitesAz", IsForeignKey = true, DeleteOnNull = false)]
        public Teljesites Teljesites { get { return teljesites.Entity; } set { teljesites.Entity = value; } }




        [Column(Name = "ingatlan_az", CanBeNull = true, DbType = "int NULL")]
        public int? IngatlanAz { get; set; } = null;
        [Column(Name = "ingó_az", CanBeNull = true, DbType = "int NULL")]
        public int? IngoAz { get; set; } = null;
        [Column(Name = "gépjármű_az", CanBeNull = true, DbType = "int NULL")]
        public int? GepjarmuAz { get; set; } = null;
        [Association(Name = "ingatlan", ThisKey = "IngatlanAz", IsForeignKey = true, DeleteOnNull = false)]
        public Ingatlan Ingatlan { get { return targy.Entity as Ingatlan; } set { targy.Entity = value; } }
        [Association(Name = "ingó", ThisKey = "IngoAz", IsForeignKey = true, DeleteOnNull = false)]
        public Ingo Ingo { get { return targy.Entity as Ingo; } set { targy.Entity = value; } }
        [Association(Name = "gépjármű", ThisKey = "GepjarmuAz", IsForeignKey = true, DeleteOnNull = false)]
        public Gepjarmu Gepjarmu { get { return targy.Entity as Gepjarmu; } set { targy.Entity = value; } }



        public bool equals(SzerzodesTargy element)
        {
            return IngatlanAz == element.IngatlanAz &&
                IngoAz == element.IngoAz &&
                GepjarmuAz == element.GepjarmuAz &&
                SzerzodesAz == element.SzerzodesAz &&
                EladoTulajdonArany == element.EladoTulajdonArany &&
                 TargyKod == element.TargyKod &&
                TeljesitesAz == element.TeljesitesAz;

        }

        public override List<string> getData()
        {
            List<string> data = base.getData();

            data.Add(IngatlanAz.ToString());
            data.Add(IngoAz.ToString());
            data.Add(GepjarmuAz.ToString());
            data.Add(SzerzodesAz.ToString());
            data.Add(EladoTulajdonArany.ToString());
            data.Add(TeljesitesAz.ToString());

            return data;
        }
    }
}
