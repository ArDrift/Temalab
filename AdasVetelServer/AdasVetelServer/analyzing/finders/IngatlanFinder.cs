using AdasVetelServer.dbHandler;
using AdasVetelServer.logger;
using AdasVetelServer.logic.nep;
using AdasVetelServer.model;
using System.Collections.Generic;

namespace AdasVetelServer.logic
{
    class IngatlanFinder : FinderBase<Ingatlan>
    {
        public static IngatlanFinder Instance { get; } = new IngatlanFinder();
        private IngatlanFinder() { }


        protected override void fillFindable()
        {

            if (findTipus()) validValues++;
            if (findTerulet()) validValues++;
            if (findFoldH()) validValues++;
            findBelt();
            if (findSzobak()) validValues++;
            if (findHanyad()) validValues++;
            if (findHelyrajzi()) validValues++;
            if (findCim()) validValues++;


        }
        protected override void validateFindable()
        {
            if (validValues < 3)
                findable.isValid = false;
        }

        protected override void formatAttribute()
        {
            findable.Foldhivatal = TextAnalyzer.Instance.firstCap(findable.Foldhivatal);
            findable.Tipus = TextAnalyzer.Instance.firstCap(findable.Tipus);
        }
        private bool findCim()
        {
            findable.Cim = CimFinder.Instance.findOne(matchesDict, start, end);
            if (findable.Cim != null)
            {
                findable.CimAz = findable.Cim.Az;
                return true;
            }
            return false;
        }
        private bool findSzobak()
        {
            MatchResult m = RegexEntityMatcher.getBestMatchInInterval(matchesDict["SZOBAKSZAMA"], start, end);
            if (m != null)
            {
                findable.SzobakSzama = int.Parse(m.Value);
                return true;
            }
            return false;
        }
        private bool findFoldH()
        {
            MatchResult m = RegexEntityMatcher.getBestMatchInInterval(matchesDict["FOLDHIVATAL"], start, end);
            if (m != null)
            {
                findable.Foldhivatal = m.Value;
                return true;
            }
            return false;

        }
        private bool findBelt()
        {
            MatchResult m = RegexEntityMatcher.getBestMatchInInterval(matchesDict["BELTERULETE"], start, end);
            if (m != null)
            {
                findable.Belterulet = true;
                return true;
            }
            return false;
        }
        private bool findHelyrajzi()
        {
            MatchResult m = RegexEntityMatcher.getBestMatchInInterval(matchesDict["HELYRAJZISZAM"], start, end);
            if (m != null)
            {
                findable.HelyrajziSzam = m.Value;
                return true;
            }
            return false;
        }
        private bool findHanyad()
        {
            MatchResult m = RegexEntityMatcher.getBestMatchInInterval(matchesDict["KOZOSTULAJDON"], start, end);
            if (m != null)
            {
                findable.KozosTulajdoniHanyad = m.Value.Replace(" ", ".");
                return true;
            }
            return false;
        }
        private bool findTerulet()
        {
            MatchResult m = RegexEntityMatcher.getBestMatchInInterval(matchesDict["TERULET"], start, end);
            if (m != null)
            {
                findable.Terulet = m.Value;
                return true;
            }
            return false;
        }

        private bool findTipus()
        {
            MatchResult m = RegexEntityMatcher.getBestMatchInInterval(matchesDict["INGATLANTIPUS"], start, end);
            if (m != null)
            {
                findable.Tipus = m.Value;
                return true;
            }
            return false;
        }
        protected override List<Ingatlan> getDatabaseElements()
        {
            return dbHandler.getIngatlanok();
        }
        protected override void insertFindableToDatabase()
        {
            dbHandler.insertIngatlanToDatabase(findable);
        }
    }
}
