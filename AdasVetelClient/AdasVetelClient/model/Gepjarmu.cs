using System;
using System.Collections.Generic;
using System.Data.Linq.Mapping;


namespace AdasVetelServer.model
{
    [Table(Name = "dbo.Gépjármű")]
    public class Gepjarmu : Targy, ICheckable<Gepjarmu>
    {

        public Gepjarmu() : base()
        {
            TargyTipus = "Gépjármű";

            Labels.Add("Rendszám");
            Labels.Add("Típus");
            Labels.Add("Alvázszám");
            Labels.Add("Motorszám");
            Labels.Add("Forgalmiszám");
            Labels.Add("Extra Felszerelések");
            Labels.Add("Első Forgalomba Helyezés");
            Labels.Add("Kilóméterszám");
            Labels.Add("Használtság");


        }
        [Column(IsPrimaryKey = true, Name = "az", IsDbGenerated = true, DbType = "Int NOT NULL IDENTITY")]
        public int Az { get { return az; } set { az = value; } }

        [Column(Name = "létrehozás_dátum")]
        public DateTime? LetrehozasDatum { get { return letrehozasdatum; } set { letrehozasdatum = value; } }


        [Column(Name = "rendszám")]
        public string Rendszam { get; set; } = "Ismeretlen";
        [Column(Name = "típus")]
        public string Tipus { get; set; } = "Ismeretlen";
        [Column(Name = "alvázszám")]
        public string Alvazszam { get; set; } = "Ismeretlen";
        [Column(Name = "motorszám")]
        public string Motorszam { get; set; } = "Ismeretlen";
        [Column(Name = "extra_felsz")]
        public string ExtraFelsz { get; set; } = "Ismeretlen";
        [Column(Name = "első_forgalomba_hely")]
        public int ElsoForgalombaHely { get; set; } = -1;
        [Column(Name = "kilóméterszám")]
        public int Kilometerszam { get; set; } = -1;
        [Column(Name = "gyártás_éve")]
        public int GyartasEve { get; set; } = -1;

        [Column(Name = "törzskönyv_szám")]
        public string TorzskonyvSzam { get; set; } = "Ismeretlen";

        [Column(Name = "forgalmi_szám")]
        public string Forgalmiszam { get; set; } = "Ismeretlen";
        [Column(Name = "használt_e")]
        public bool HasznaltE { get; set; } = false;

        public bool equals(Gepjarmu element)
        {
            return Rendszam == element.Rendszam &&
              Alvazszam == element.Alvazszam &&
              Motorszam == element.Motorszam &&
                 Tipus == element.Tipus &&
               ExtraFelsz == element.ExtraFelsz &&
                TorzskonyvSzam == element.TorzskonyvSzam &&
                ElsoForgalombaHely == element.ElsoForgalombaHely &&
                Kilometerszam == element.Kilometerszam &&
                 GyartasEve == element.GyartasEve &&
                 HasznaltE == element.HasznaltE &&
                  Forgalmiszam == element.Forgalmiszam;
        }

        public override List<string> getData()
        {
            List<string> data = base.getData();

            data.Add(Rendszam);
            data.Add(Tipus);
            data.Add(Alvazszam);
            data.Add(Motorszam);
            data.Add(Forgalmiszam);
            data.Add(ExtraFelsz);
            data.Add(ElsoForgalombaHely.ToString());
            data.Add(Kilometerszam.ToString());
            data.Add(HasznaltE.ToString());

            return data;
        }
    }
}
