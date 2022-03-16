using AdasVetelServer.dbHandler;
using AdasVetelServer.logic.nep;
using AdasVetelServer.model;
using System.Collections.Generic;

namespace AdasVetelServer.logic
{
    class GepjarmuFinder : FinderBase<Gepjarmu>
    {
        public static GepjarmuFinder Instance { get; } = new GepjarmuFinder();
        private GepjarmuFinder() { }
        protected override void fillFindable()
        {
            if (findAlvazszam()) validValues++;
            if (findForgalmiszam()) validValues++;
            findExtra();
            if (findMotorszam()) validValues++;
            if (findRendszam()) validValues++;
            if (findTipus()) validValues++;
            if (findTorzskonyv()) validValues++;
            if (findKilometer()) validValues++;
            if (findGyartasev()) validValues++;
            if (findElsoForgalom()) validValues++;
            findHasznalte();

        }
        private bool findHasznalte()
        {
            MatchResult m = RegexEntityMatcher.getBestMatchInInterval(matchesDict["HASZNALTE"], start, end);
            if (m != null)
            {
                findable.HasznaltE = true;
                return true;
            }
            return false;
        }
        private bool findAlvazszam()
        {
            MatchResult m = RegexEntityMatcher.getBestMatchInInterval(matchesDict["ALVAZSZAM"], start, end);
            if (m != null)
            {
                findable.Alvazszam = m.Value;
                return true;
            }
            return false;
        }
        private bool findForgalmiszam()
        {
            MatchResult m = RegexEntityMatcher.getBestMatchInInterval(matchesDict["FORGALMISZAM"], start, end);
            if (m != null)
            {
                findable.Forgalmiszam = m.Value;
                return true;
            }
            return false;
        }
        private bool findExtra()
        {
            List<MatchResult> ms = RegexEntityMatcher.getMatchesByInterval(matchesDict["EXTRAFELSZ"], start, end);
            if (ms.Count != 0)
            {
                string extras = "";
                foreach (MatchResult item in ms)
                {
                    extras = $"{extras}, {item}";
                }
                findable.ExtraFelsz = extras;
                return true;
            }
            return false;
        }

        private bool findMotorszam()
        {
            MatchResult m = RegexEntityMatcher.getBestMatchInInterval(matchesDict["MOTORSZAM"], start, end);
            if (m != null)
            {
                findable.Motorszam = m.Value;
                return true;
            }
            return false;
        }
        private bool findRendszam()
        {
            MatchResult m = RegexEntityMatcher.getBestMatchInInterval(matchesDict["RENDSZAM"], start, end);
            if (m != null)
            {
                findable.Rendszam = m.Value;
                return true;
            }
            return false;
        }
        private bool findTipus()
        {
            MatchResult m = RegexEntityMatcher.getBestMatchInInterval(matchesDict["JARMUTIPUS"], start, end);
            if (m != null)
            {
                findable.Tipus = m.Value;
                return true;
            }
            return false;
        }
        private bool findTorzskonyv()
        {
            MatchResult m = RegexEntityMatcher.getBestMatchInInterval(matchesDict["TORZSKONYVSZAM"], start, end);
            if (m != null)
            {
                findable.TorzskonyvSzam = m.Value;
                return true;
            }
            return false;
        }
        private bool findKilometer()
        {
            MatchResult m = RegexEntityMatcher.getBestMatchInInterval(matchesDict["KILOMETERSZAM"], start, end);
            if (m != null)
            {
                findable.Kilometerszam = int.Parse(m.Value);
                return true;
            }
            return false;
        }
        private bool findGyartasev()
        {
            MatchResult m = RegexEntityMatcher.getBestMatchInInterval(matchesDict["GYARTASEVE"], start, end);
            if (m != null)
            {
                findable.GyartasEve = int.Parse(m.Value);
                return true;
            }
            return false;
        }
        private bool findElsoForgalom()
        {
            MatchResult m = RegexEntityMatcher.getBestMatchInInterval(matchesDict["ELSOFORGALOM"], start, end);
            if (m != null)
            {
                findable.ElsoForgalombaHely = int.Parse(m.Value);
                return true;
            }
            return false;
        }

        protected override void formatAttribute()
        {
            findable.Tipus = TextAnalyzer.Instance.firstCap(findable.Tipus);
            if (findable.Motorszam != "Ismeretlen")
                findable.Motorszam = findable.Motorszam.ToUpper();
            if (findable.Forgalmiszam != "Ismeretlen")
                findable.Forgalmiszam = findable.Forgalmiszam.ToUpper();
            if (findable.Alvazszam != "Ismeretlen")
                findable.Alvazszam = findable.Alvazszam.ToUpper();
            if (findable.Rendszam != "Ismeretlen")
                findable.Rendszam = findable.Rendszam.ToUpper();
        }

        protected override void validateFindable()
        {

            if (validValues < 4)
                findable.isValid = false;
        }
        protected override List<Gepjarmu> getDatabaseElements()
        {           
            return dbHandler.getGepjarmuvek();
        }
        protected override void insertFindableToDatabase()
        {
            dbHandler.insertGepjarmuToDatabase(findable);
        }
    }

}
