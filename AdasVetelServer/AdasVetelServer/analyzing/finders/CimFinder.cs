using AdasVetelServer.dbHandler;
using AdasVetelServer.logic.nep;
using AdasVetelServer.model;
using System.Collections.Generic;

namespace AdasVetelServer.logic
{
    class CimFinder : FinderBase<Cim>
    {
        public static CimFinder Instance { get; } = new CimFinder();


        private CimFinder() { }
        protected override void fillFindable()
        {
            findOrszag();
            if (findVaros()) validValues++;
            if (findIranyitoszam()) validValues++;
            if (findUtca()) validValues++;
            if (findHazszam()) validValues++;
            if (findEmelet()) validValues++;


        }
        private bool findOrszag()
        {
            MatchResult m = RegexEntityMatcher.getBestMatchInInterval(matchesDict["ORSZAG"], start, end);
            if (m != null)
            {
                findable.Orszag = m.Value;
                return true;
            }
            return false;
        }
        private bool findIranyitoszam()
        {
            MatchResult m = RegexEntityMatcher.getBestMatchInInterval(matchesDict["IRANYITOSZAM"], start, end);
            if (m != null)
            {
                findable.Iranyitoszam = int.Parse(m.Value);
                return true;
            }
            return false;
        }
        private bool findVaros()
        {
            MatchResult m = RegexEntityMatcher.getBestMatchInInterval(matchesDict["VAROS"], start, end);
            if (m != null)
            {
                findable.Varos = m.Value;
                return true;
            }
            return false;
        }
        private bool findUtca()
        {
            MatchResult m = RegexEntityMatcher.getBestMatchInInterval(matchesDict["UTCA"], start, end);
            if (m != null)
            {
                findable.Utca = m.Value;
                return true;
            }
            return false;
        }

        private bool findHazszam()
        {
            MatchResult m = RegexEntityMatcher.getBestMatchInInterval(matchesDict["HAZSZAM"], start, end);
            if (m != null)
            {
                findable.Hazszam = m.Value;
                return true;
            }
            return false;
        }
        private bool findEmelet()
        {
            MatchResult m = RegexEntityMatcher.getBestMatchInInterval(matchesDict["EMELET"], start, end);
            if (m != null)
            {
                findable.Emelet = int.Parse(m.Value);
                return true;
            }
            return false;
        }

        protected override void formatAttribute()
        {
            findable.Hazszam = TextAnalyzer.Instance.firstCap(findable.Hazszam);
            findable.Orszag = TextAnalyzer.Instance.firstCap(findable.Orszag);
            findable.Varos = TextAnalyzer.Instance.firstCap(findable.Varos);
            findable.Utca = TextAnalyzer.Instance.firstCap(findable.Utca);
        }

        protected override void validateFindable()
        {
            if (validValues < 2)
                findable.isValid = false;
        }
        protected override List<Cim> getDatabaseElements() {           
            return dbHandler.getCimek();
        }
        protected override void insertFindableToDatabase() {
            dbHandler.insertCimToDatabase(findable);
        }
    }
}
