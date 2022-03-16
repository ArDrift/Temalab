using AdasVetelServer.dbHandler;
using AdasVetelServer.logic.nep;
using AdasVetelServer.model;
using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;

namespace AdasVetelServer.logic
{
    class SzerzodesFinder : FinderBase<Szerzodes>
    {


        public string FileName { get; set; } = "Ismeretlen";
        public static SzerzodesFinder Instance { get; } = new SzerzodesFinder();
        private SzerzodesFinder() { }
        private int reszvetelEnd = 0;


        protected override void fillFindable()
        {
            if (end - start < 50)
            {
                findable.isValid = false;
                return;
            }

            findable.FajlNev = FileName;
            findTipus();
            findKelt();
            findJegyzo();

            findResztvevok();
            findTargyak();

        }

        private void findTipus()
        {
            MatchResult m = RegexEntityMatcher.getBestMatchInInterval(matchesDict["SZERZODESTIPUS"], start, start + 500);
            if (m != null)
                findable.Tipus = m.Value + " adásvételi szerződés";
        }
        private void findKelt()
        {
            MatchResult date = RegexEntityMatcher.getBestMatchInInterval(matchesDict["KELTDATUM"], end / 2, end);
            if (date != null)
            {
                MatchCollection m = Regex.Matches(date.Value, "[0-9]+");
                findable.KeltDatum = new DateTime(int.Parse(m[0].Value), int.Parse(m[1].Value), int.Parse(m[2].Value));
            }
        }

        private void findJegyzo()
        {
            MatchResult m = RegexEntityMatcher.getBestMatchInInterval(matchesDict["KOZJEGYZO"], end / 2 , end);
            if (m != null)
                findable.EllenjegyzoIroda = m.Value + " Ügyvédi iroda";
        }

        protected override void formatAttribute()
        {

            findable.Tipus = TextAnalyzer.Instance.firstCap(findable.Tipus);
            findable.EllenjegyzoIroda = TextAnalyzer.Instance.firstCap(findable.EllenjegyzoIroda);

        }

        protected override void validateFindable()
        {
            if (findable.KeltDatum <= new DateTime(1900, 01, 01) || findable.KeltDatum >= new DateTime(3000, 01, 01))
                findable.KeltDatum = null;
        }
        protected override List<Szerzodes> getDatabaseElements()
        {
            return dbHandler.getSzerzodesek();
        }
        protected override void insertFindableToDatabase()
        {
            dbHandler.insertSzerzodesToDatabase(findable);
        }
        private void findTargyak()
        {
            SzerzodesTargyFinder.Instance.Szerzodes = findable;
            SzerzodesTargyFinder.Instance.findOne(matchesDict, reszvetelEnd, end);
        }
        private void findResztvevok()
        {
            List<MatchResult> roleMatches = RegexEntityMatcher.getMatchesByInterval(matchesDict["SZEREP"], start, end);
            MatchResult endMatch = null;
            foreach (var match in roleMatches)
            {
                MatchResult startMatch = endMatch;
                endMatch = match;
                int startIndex = startMatch == null ? 0 : startMatch.BeginIndex;

                ReszvetelFinder.Instance.Szerzodes = findable;
                ReszvetelFinder.Instance.Role = guessRole(startMatch == null ? null : startMatch.Value, endMatch.Value);
                if (ReszvetelFinder.Instance.findOne(matchesDict, startIndex, endMatch.EndIndex) != null) {
                    reszvetelEnd = endMatch.EndIndex;
                }
            }
        }

        private string guessRole(string start, string end)
        {
            int startMap = mapRole(start);
            int endMap = mapRole(end);
            if (startMap == 1 && endMap == 1)
                return "Eladó";
            else if (startMap == 1 && endMap == 0)
                return "Eladó";
            else if (startMap == 1 && endMap == 2)
                return "Eladó";
            else if (startMap == 2 && endMap == 1)
                return "Eladó";
            else
                return "Vevő";
        }

        private int mapRole(string s)
        {
            if (s == null)
                return 2;
            if (s.ToLower().Equals("eladó") || s.ToLower().Equals("eladók"))
                return 1;
            else if (s.ToLower().Equals("vevő") || s.ToLower().Equals("vevők"))
                return 0;
            else
                return 2;
        }
    }
}
