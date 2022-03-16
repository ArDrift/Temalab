using System;
using System.Collections.Generic;
using System.Data.Linq;
using System.Data.Linq.Mapping;

namespace AdasVetelServer.model
{
    [Table(Name = "dbo.Személy")]
    public class Szemely : Resztvevo, ICheckable<Szemely>
    {
        private EntityRef<Cim> lakcim = new EntityRef<Cim>();



        private readonly EntitySet<Reszvetel> reszvetelek = new EntitySet<Reszvetel>();
        [Association(Name = "részvételek", OtherKey = "SzemelyAz")]
        public EntitySet<Reszvetel> Reszvetelek
        {
            get { return reszvetelek; }
            set { reszvetelek.Assign(value); }
        }


        public Szemely() : base()
        {
            ResztvevoTipus = "Személy";

            Labels.Add("Családi név");
            Labels.Add("Utónév");
            Labels.Add("Születési hely");
            Labels.Add("Születési idő");
            Labels.Add("Anyja családi neve");
            Labels.Add("Anyja utóneve");
            Labels.Add("Személyi szám");
            Labels.Add("Adóazonosító");
            Labels.Add("Lakcím azonosító");
            Labels.Add("Számlaszám");

        }
        [Column(IsPrimaryKey = true, Name = "az", IsDbGenerated = true, DbType = "Int NOT NULL IDENTITY")]
        public int Az { get { return az; } set { az = value; } }

        [Column(Name = "létrehozás_dátum")]
        public DateTime? LetrehozasDatum { get { return letrehozasdatum; } set { letrehozasdatum = value; } }

        [Column(Name = "családi_név")]
        public string CsaladiNev { get; set; } = "Ismeretlen";


        [Column(Name = "utó_név")]
        public string UtoNev { get; set; } = "Ismeretlen";


        [Column(Name = "szül_hely")]
        public string SzulHely { get; set; } = "Ismeretlen";


        [Column(Name = "szül_idő", CanBeNull = true)]
        public DateTime? SzulIdo { get; set; } = null;


        [Column(Name = "anyja_családi_neve")]
        public string AnyjaCsaladiNeve { get; set; } = "Ismeretlen";


        [Column(Name = "anyja_utó_neve")]
        public string AnyjaUtoNeve { get; set; } = "Ismeretlen";


        [Column(Name = "személyi_szám")]
        public string SzemelyiSzam { get; set; } = "Ismeretlen";


        [Column(Name = "adóazonosító")]
        public string Adoazonosito { get; set; } = "Ismeretlen";



        [Column(Name = "számlaszám")]
        public string Szamlaszam { get; set; } = "Ismeretlen";





        [Column(Name = "lakcím_az", CanBeNull = true, DbType = "int NULL")]
        public int? LakcimAz { get; set; } = null;
        [Association(Name = "lakcím", ThisKey = "LakcimAz", IsForeignKey = true)]
        public Cim Lakcim { get { return lakcim.Entity; } set { lakcim.Entity = value; } }


        public override List<string> getData()
        {
            List<string> data = base.getData();

            data.Add(CsaladiNev.ToString());
            data.Add(UtoNev.ToString());
            data.Add(SzulHely.ToString());
            data.Add(SzulIdo.ToString());
            data.Add(AnyjaCsaladiNeve.ToString());
            data.Add(AnyjaUtoNeve.ToString());
            data.Add(SzemelyiSzam.ToString());
            data.Add(Adoazonosito.ToString());
            data.Add(LakcimAz.ToString());
            data.Add(Szamlaszam.ToString());
            return data;
        }

        public bool equals(Szemely element)
        {

            return (Adoazonosito == element.Adoazonosito ||
                SzemelyiSzam == element.SzemelyiSzam) &&
                 SzemelyiSzam == element.SzemelyiSzam &&
                 UtoNev == element.UtoNev &&
                 Szamlaszam == element.Szamlaszam &&
                 LakcimAz == element.LakcimAz &&
                 SzulIdo.ToString() == element.SzulIdo.ToString() &&
                 SzulHely == element.SzulHely;
        }
    }

}
