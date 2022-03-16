using AdasVetelServer.dbHandler;
using AdasVetelServer.logic.nep;
using AdasVetelServer.model;
using System.Collections.Generic;

namespace AdasVetelServer.logic
{
    class SzervezetFinder : FinderBase<Szervezet>
    {
        public static SzervezetFinder Instance { get; } = new SzervezetFinder();
        private SzervezetFinder() { }
        protected override void fillFindable()
        {
            if (findNev()) validValues++;
            findTipus();
            if (findCegjegy()) validValues += 2;
            if (findAdo()) validValues += 2;
            if (findStat()) validValues++;
            if (findSzamla()) validValues++;
            if (findSzekhely()) validValues++;
        }
        private bool findTipus()
        {
            MatchResult m = RegexEntityMatcher.getBestMatchInInterval(matchesDict["CEGTIPUS"], start, end);
            if (m != null)
            {
                findable.Tipus = m.Value;
                return true;
            }
            return false;
        }
        private bool findSzekhely()
        {
            findable.Szekhely = CimFinder.Instance.findOne(matchesDict, start, end);
            if (findable.Szekhely != null)
            {
                findable.SzekhelyAz = findable.Szekhely.Az;
                return true;
            }
            return false;
        }
        private bool findNev()
        {
            MatchResult m = RegexEntityMatcher.getBestMatchInInterval(matchesDict["CEGNEV"], start, end);
            if (m != null)
            {
                findable.Nev = m.Value;
                return true;
            }
            return false;
        }

        private bool findCegjegy()
        {
            MatchResult m = RegexEntityMatcher.getBestMatchInInterval(matchesDict["CEGJEGYZEKSZAM"], start, end);
            if (m != null)
            {
                findable.Cegjegyzekszam = m.Value.Replace(" ", "");
                return true;
            }
            return false;
        }
        private bool findStat()
        {
            MatchResult m = RegexEntityMatcher.getBestMatchInInterval(matchesDict["STATISZTIKAIAZ"], start, end);
            if (m != null)
            {
                findable.StatAz = m.Value.Replace(" ", "");
                return true;
            }
            return false;
        }
        private bool findAdo()
        {
            MatchResult m = RegexEntityMatcher.getBestMatchInInterval(matchesDict["ADOSZAM"], start, end);
            if (m != null)
            {
                findable.Adoszam = m.Value.Replace(" ", "");
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

        protected override void formatAttribute()
        {
            findable.Nev = TextAnalyzer.Instance.firstCap(findable.Nev);
            findable.Tipus = TextAnalyzer.Instance.firstCap(findable.Tipus);
        }

        protected override void validateFindable()
        {
            if (validValues < 4)
                findable.isValid = false;
        }
        protected override List<Szervezet> getDatabaseElements()
        {
            return dbHandler.getSzervezetek();
        }
        protected override void insertFindableToDatabase()
        {
            dbHandler.insertSzervezetToDatabase(findable);
        }
    }

}

