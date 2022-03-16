using System;
using System.Data.Linq.Mapping;

namespace AdasVetelServer.model
{
    [Table(Name = "dbo.Ingóság")]
    public class Ingo : Targy, ICheckable<Ingo>
    {
        public Ingo()
        {
            TargyTipus = "Ingó";
        }

        public bool equals(Ingo element)
        {
            return false;
        }
        [Column(IsPrimaryKey = true, Name = "az", IsDbGenerated = true, DbType = "Int NOT NULL IDENTITY")]
        public int Az { get { return az; } set { az = value; } }

        [Column(Name = "létrehozás_dátum")]
        public DateTime? LetrehozasDatum { get { return letrehozasdatum; } set { letrehozasdatum = value; } }

    }
}
