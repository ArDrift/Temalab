using AdasVetelServer.dbHandler;
using AdasVetelServer.logic.nep;
using AdasVetelServer.model;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text.RegularExpressions;

namespace AdasVetelServer.logic
{
    class TeljesitesFinder : FinderBase<Teljesites>
    {
        public static TeljesitesFinder Instance { get; } = new TeljesitesFinder();
        private TeljesitesFinder() { }
        protected override void fillFindable()
        {
            if (findAfa()) validValues++;
            if (findHatarido()) validValues++;
            if (findVetelar()) validValues++;
            if (findReszteljesites()) validValues++;
            findAtutalasE();
        }
        private bool findAtutalasE()
        {
            MatchResult m = RegexEntityMatcher.getBestMatchInInterval(matchesDict["ATUTALASE"], start, end);
            if (m != null)
            {
                findable.BankiAtutalasE = true;
                return true;
            }
            return false;
        }
        private bool findAfa()
        {
            MatchResult m = RegexEntityMatcher.getBestMatchInInterval(matchesDict["AFA"], start, end);
            if (m != null)
            {
                findable.Afa = m.Value;
                return true;
            }
            return false;
        }
        private bool findHatarido()
        {           
            MatchResult date = RegexEntityMatcher.getBestMatchInInterval(matchesDict["HATARIDO"], start, end);
            if (date != null)
            {
                MatchCollection m = Regex.Matches(date.Value, "[0-9]+");
                findable.Hatarido = new DateTime(int.Parse(m[0].Value), int.Parse(m[1].Value), int.Parse(m[2].Value));
                if (findable.Hatarido > new DateTime(1900, 01, 01) && findable.Hatarido < new DateTime(3000, 01, 01))
                    return true;
            }
            findable.Hatarido = null;
            return false;
        }
        private bool findVetelar()
        {
            MatchResult m = RegexEntityMatcher.getBestMatchInInterval(matchesDict["VETELAR"], start, end);
            if (m != null)
            {
                string vetelar = m.Value;
                if (vetelar.Contains(","))
                {
                    vetelar = vetelar.Remove(' ');
                }
                vetelar = vetelar.Replace(',', '.');
                findable.Vetelar = float.Parse(vetelar, NumberStyles.Currency);
                return true;
            }
            return false;
        }
        private bool findReszteljesites()
        {
            return true;
        }
        protected override void formatAttribute()
        { }

        protected override void validateFindable()
        {
            if (validValues < 2)
                findable.isValid = false;
        }
        protected override List<Teljesites> getDatabaseElements()
        {
            return dbHandler.getTeljesitesek();
        }
        protected override void insertFindableToDatabase()
        {
            dbHandler.insertTeljesitesToDatabase(findable);
        }
    }
}
