using System;
using System.Collections.Generic;
using System.Data.Linq;
using System.Data.Linq.Mapping;

namespace AdasVetelServer.model
{
    [Table(Name = "dbo.Szervezet")]
    public class Szervezet : Resztvevo, ICheckable<Szervezet>
    {
        private EntityRef<Cim> szekhely = new EntityRef<Cim>();

        private readonly EntitySet<Reszvetel> reszvetelek = new EntitySet<Reszvetel>();
        [Association(Name = "részvételek", OtherKey = "SzervezetAz")]
        public EntitySet<Reszvetel> Reszvetelek
        {
            get { return reszvetelek; }
            set { reszvetelek.Assign(value); }
        }


        public Szervezet() : base()
        {
            ResztvevoTipus = "Szervezet";
            Labels.Add("Név");
            Labels.Add("Típus");
            Labels.Add("Székhely Cím az");
            Labels.Add("Cégjegyzékszám");
            Labels.Add("Statisztikai azonosító");
            Labels.Add("Adószám");
            Labels.Add("Számlaszám");


        }
        [Column(IsPrimaryKey = true, Name = "az", IsDbGenerated = true, DbType = "Int NOT NULL IDENTITY")]
        public int Az { get { return az; } set { az = value; } }

        [Column(Name = "létrehozás_dátum")]
        public DateTime? LetrehozasDatum { get { return letrehozasdatum; } set { letrehozasdatum = value; } }


        [Column(Name = "név")]
        public string Nev { get; set; } = "Ismeretlen";


        [Column(Name = "típus")]
        public string Tipus { get; set; } = "Ismeretlen";

        [Column(Name = "cégjegyzékszám")]
        public string Cegjegyzekszam { get; set; } = "Ismeretlen";


        [Column(Name = "stat_az")]
        public string StatAz { get; set; } = "Ismeretlen";


        [Column(Name = "adószám")]
        public string Adoszam { get; set; } = "Ismeretlen";


        [Column(Name = "számlaszám")]
        public string Szamlaszam { get; set; } = "Ismeretlen";





        [Column(Name = "székhely_cím_az", CanBeNull = true, DbType = "int NULL")]
        public int? SzekhelyAz { get; set; } = null;
        [Association(Name = "székhely", ThisKey = "SzekhelyAz", IsForeignKey = true)]
        public Cim Szekhely { get { return szekhely.Entity; } set { szekhely.Entity = value; } }


        public bool equals(Szervezet element)
        {
            return (Cegjegyzekszam == element.Cegjegyzekszam ||
                  Adoszam == element.Adoszam) &&
                Nev == element.Nev &&
                   Szamlaszam == element.Szamlaszam &&
                   StatAz == element.StatAz &&
                    Tipus == element.Tipus &&
                    SzekhelyAz == element.SzekhelyAz;
        }

        public override List<string> getData()
        {
            List<string> data = base.getData();
            data.Add(Nev.ToString());
            data.Add(Tipus.ToString());
            data.Add(SzekhelyAz.ToString());
            data.Add(Cegjegyzekszam.ToString());
            data.Add(StatAz.ToString());
            data.Add(Adoszam.ToString());
            data.Add(Szamlaszam.ToString());

            return data;
        }
    }
}
