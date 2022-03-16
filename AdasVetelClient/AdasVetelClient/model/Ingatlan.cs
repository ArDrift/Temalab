

using System;
using System.Collections.Generic;
using System.Data.Linq;
using System.Data.Linq.Mapping;

namespace AdasVetelServer.model
{
    [Table(Name = "dbo.Ingatlan")]
    public class Ingatlan : Targy, ICheckable<Ingatlan>
    {
        private EntityRef<Cim> cim = new EntityRef<Cim>();

        public Ingatlan()
        {
            TargyTipus = "Ingatlan";

            Labels.Add("Földhivatal");
            Labels.Add("Belterület");
            Labels.Add("Helyrajzi szám");
            Labels.Add("Terület");
            Labels.Add("Szobák száma");
            Labels.Add("Típus");
            Labels.Add("Közös tulajdoni hányad");
            Labels.Add("Cím az");


        }
        [Column(IsPrimaryKey = true, Name = "az", IsDbGenerated = true, DbType = "Int NOT NULL IDENTITY")]
        public int Az { get { return az; } set { az = value; } }

        [Column(Name = "létrehozás_dátum")]
        public DateTime? LetrehozasDatum { get { return letrehozasdatum; } set { letrehozasdatum = value; } }


        [Column(Name = "földhivatal")]
        public string Foldhivatal { get; set; } = "Ismeretlen";
        [Column(Name = "belterület")]
        public bool Belterulet { get; set; } = true;
        [Column(Name = "helyrajzi_szám")]
        public string HelyrajziSzam { get; set; } = "Ismeretlen";
        [Column(Name = "terület")]
        public string Terulet { get; set; } = "Ismeretlen";
        [Column(Name = "szobák_száma")]
        public int SzobakSzama { get; set; } = -1;
        [Column(Name = "típus")]
        public string Tipus { get; set; } = "Ismeretlen";
        [Column(Name = "közös_tulajdoni_hányad")]
        public string KozosTulajdoniHanyad { get; set; } = "Ismeretlen";






        [Column(Name = "cím_az", CanBeNull = true, DbType = "int NULL")]
        public int? CimAz { get; set; } = null;
        [Association(Name = "cím", ThisKey = "CimAz", IsForeignKey = true)]
        public Cim Cim { get { return cim.Entity; } set { cim.Entity = value; } }





        public bool equals(Ingatlan element)
        {

            return CimAz == element.CimAz &&
                Foldhivatal == element.Foldhivatal &&
                Belterulet == element.Belterulet &&
                 HelyrajziSzam == element.HelyrajziSzam &&
                Terulet == element.Terulet &&
                SzobakSzama == element.SzobakSzama &&
                Tipus == element.Tipus &&
                 KozosTulajdoniHanyad == element.KozosTulajdoniHanyad;
        }

        public override List<string> getData()
        {
            List<string> data = base.getData();

            data.Add(Foldhivatal.ToString());
            data.Add(Belterulet.ToString());
            data.Add(HelyrajziSzam);
            data.Add(Terulet.ToString());
            data.Add(SzobakSzama.ToString());
            data.Add(Tipus.ToString());
            data.Add(KozosTulajdoniHanyad);
            data.Add(CimAz.ToString());

            return data;
        }
    }
}
