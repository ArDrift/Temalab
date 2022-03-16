using System;
using System.Collections.Generic;

namespace AdasVetelServer.model
{
    public abstract class DbElement
    {
        public DbElement()
        {
            Labels.Add("Az");
            Labels.Add("Létrehozás dátuma");
        }

        protected int az = -1;
        protected DateTime? letrehozasdatum = DateTime.Now;
        public bool isValid { get; set; } = true;
        public List<string> Labels { get; set; } = new List<string>();
        public virtual List<string> getData()
        {
            List<string> data = new List<string>();
            data.Add(az.ToString());
            data.Add(letrehozasdatum.ToString());
            return data;
        }


    }
}
