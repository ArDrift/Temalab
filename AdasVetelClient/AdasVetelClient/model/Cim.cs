
using System;
using System.Collections.Generic;
using System.Data.Linq;
using System.Data.Linq.Mapping;


namespace AdasVetelServer.model
{
    [Table(Name = "dbo.Cím")]
    public class Cim : DbElement, ICheckable<Cim>
    {

        private readonly EntitySet<Szemely> szemelyek = new EntitySet<Szemely>();
        [Association(Name = "személyek", OtherKey = "LakcimAz")]
        public EntitySet<Szemely> Szemelyek
        {
            get { return szemelyek; }
            set { szemelyek.Assign(value); }
        }

        private readonly EntitySet<Szervezet> szervezetek = new EntitySet<Szervezet>();
        [Association(Name = "szervezetek", OtherKey = "SzekhelyAz")]
        public EntitySet<Szervezet> Szervezetek
        {
            get { return szervezetek; }
            set { szervezetek.Assign(value); }
        }

        private readonly EntitySet<Ingatlan> ingatlanok = new EntitySet<Ingatlan>();
        [Association(Name = "ingatlanok", OtherKey = "CimAz")]
        public EntitySet<Ingatlan> Ingatlanok
        {
            get { return ingatlanok; }
            set { ingatlanok.Assign(value); }
        }

        public Cim() : base()
        {
            Labels.Add("Ország");
            Labels.Add("Irányítószám");
            Labels.Add("Város");
            Labels.Add("Utca");
            Labels.Add("Házszám");
            Labels.Add("Emelet");

        }
        [Column(IsPrimaryKey = true, Name = "az", IsDbGenerated = true, DbType = "Int NOT NULL IDENTITY")]
        public int Az { get { return az; } set { az = value; } }

        [Column(Name = "létrehozás_dátum")]
        public DateTime? LetrehozasDatum { get { return letrehozasdatum; } set { letrehozasdatum = value; } }


        [Column(Name = "ország")]
        public string Orszag { get; set; } = "Magyarország";
        [Column(Name = "irányítószám")]
        public int Iranyitoszam { get; set; } = -1;
        [Column(Name = "város")]
        public string Varos { get; set; } = "Ismeretlen";
        [Column(Name = "utca")]
        public string Utca { get; set; } = "Ismeretlen";
        [Column(Name = "házszám")]
        public string Hazszam { get; set; } = "Ismeretlen";
        [Column(Name = "emelet")]
        public int Emelet { get; set; } = -1;

        public bool equals(Cim element)
        {

            return Orszag == element.Orszag &&
               (Iranyitoszam == element.Iranyitoszam || Varos == element.Varos) &&
               Utca == element.Utca &&
              Hazszam == element.Hazszam &&
                Emelet == element.Emelet;
        }

        public override List<string> getData()
        {
            List<string> data = base.getData();

            data.Add(Orszag.ToString());
            data.Add(Iranyitoszam.ToString());
            data.Add(Varos.ToString());
            data.Add(Utca.ToString());
            data.Add(Hazszam.ToString());
            data.Add(Emelet.ToString());
            return data;
        }
    }
}
