using AdasVetelServer.dbHandler;
using AdasVetelServer.logic.nep;
using AdasVetelServer.model;
using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;

namespace AdasVetelServer.logic
{
    class SzemelyFinder : FinderBase<Szemely>
    {
        public static SzemelyFinder Instance { get; } = new SzemelyFinder();
        private SzemelyFinder() { }

        protected override void fillFindable()
        {

            if (findCsaladiNev()) validValues++;
            if (findUtoNev()) validValues++;
            if (findAnyja()) validValues++;
            if (findSzemelyi()) validValues += 2;
            if (findAdo()) validValues += 2;
            if (findSzulIdo()) validValues++;
            if (findSzulHely()) validValues++;
            if (findSzamla()) validValues++;
            if (findCim()) validValues++;

        }
        private bool findCim()
        {
            findable.Lakcim = CimFinder.Instance.findOne(matchesDict, start, end);
            if (findable.Lakcim != null)
            {
                findable.LakcimAz = findable.Lakcim.Az;
                return true;
            }
            return false;
        }
        private bool findAdo()
        {
            MatchResult m = RegexEntityMatcher.getBestMatchInInterval(matchesDict["ADOSZAM"], start, end);
            if (m != null)
            {
                findable.Adoazonosito = m.Value.Replace(" ", "");
                return true;
            }
            return false;
        }
        private bool findSzemelyi()
        {
            MatchResult m = RegexEntityMatcher.getBestMatchInInterval(matchesDict["SZEMELYISZAM"], start, end);
            if (m != null)
            {
                findable.SzemelyiSzam = m.Value.Replace(" ", "");
                return true;
            }
            return false;
        }
        private bool findSzamla()
        {
            MatchResult m = RegexEntityMatcher.getBestMatchInInterval(matchesDict["SZAMLASZAM"], start, end);
            if (m != null)
            {
                findable.Szamlaszam = m.Value.Replace(" ", "-");
                return true;
            }
            return false;
        }
        private bool findAnyja()
        {
            MatchResult m = RegexEntityMatcher.getBestMatchInInterval(matchesDict["ANYJANEVE"], start, end);
            if (m != null)
            {
                findable.AnyjaCsaladiNeve = m.Value.Split(' ')[0];
                findable.AnyjaUtoNeve = m.Value.Split(' ')[1];
                return true;
            }
            return false;
        }
        private bool findSzulIdo()
        {
            MatchResult date = RegexEntityMatcher.getBestMatchInInterval(matchesDict["SZULETESIDATUM"], start, end);
            if (date != null)
            {
                MatchCollection m = Regex.Matches(date.Value, "[0-9]+");
                findable.SzulIdo = new DateTime(int.Parse(m[0].Value), int.Parse(m[1].Value), int.Parse(m[2].Value));
                if (findable.SzulIdo > new DateTime(1900, 01, 01) && findable.SzulIdo < new DateTime(3000, 01, 01))
                    return true;
            }
            findable.SzulIdo = null;
            return false;

        }
        private bool findSzulHely()
        {
            MatchResult m = RegexEntityMatcher.getBestMatchInInterval(matchesDict["SZULETESIHELY"], start, end);
            if (m != null)
            {
                findable.SzulHely = m.Value;
                return true;
            }
            return false;
        }
        private bool findCsaladiNev()
        {
            MatchResult m = RegexEntityMatcher.getBestMatchInInterval(matchesDict["CSALADINEV"], start, end);
            if (m != null)
            {
                findable.CsaladiNev = m.Value;
                return true;
            }
            return false;
        }
        private bool findUtoNev()
        {
            MatchResult m = RegexEntityMatcher.getBestMatchInInterval(matchesDict["UTONEV"], start, end);
            if (m != null)
            {
                findable.UtoNev = m.Value;
                return true;
            }
            return false;
        }
        protected override void formatAttribute()
        {
            findable.CsaladiNev = TextAnalyzer.Instance.firstCap(findable.CsaladiNev);
            findable.UtoNev = TextAnalyzer.Instance.firstCap(findable.UtoNev);
            findable.AnyjaCsaladiNeve = TextAnalyzer.Instance.firstCap(findable.AnyjaCsaladiNeve);
            findable.AnyjaUtoNeve = TextAnalyzer.Instance.firstCap(findable.AnyjaUtoNeve);
            findable.SzulHely = TextAnalyzer.Instance.firstCap(findable.SzulHely);
        }

        protected override void validateFindable()
        {
            if (validValues < 6)
                findable.isValid = false;
        }
        protected override List<Szemely> getDatabaseElements()
        {
            return dbHandler.getSzemelyek();
        }
        protected override void insertFindableToDatabase()
        {
            dbHandler.insertSzemelyToDatabase(findable);
        }
    }
}
