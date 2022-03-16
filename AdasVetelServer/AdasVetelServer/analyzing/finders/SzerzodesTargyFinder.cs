using AdasVetelServer.dbHandler;
using AdasVetelServer.logic.nep;
using AdasVetelServer.model;
using System.Collections.Generic;

namespace AdasVetelServer.logic
{

    class SzerzodesTargyFinder : FinderBase<SzerzodesTargy>
    {
        public static SzerzodesTargyFinder Instance { get; } = new SzerzodesTargyFinder();
        private SzerzodesTargyFinder() { }
        public Szerzodes Szerzodes { get; set; } = null;

        protected override void fillFindable()
        {

            if (findIngatlan()) validValues++;
            else if (findGepjarmu()) validValues++;
            else findable.isValid = false;

            if (findArany()) validValues++;
            if (findteljesites()) validValues++;

            findable.Szerzodes = Szerzodes;
            findable.SzerzodesAz = Szerzodes.Az;
        }
        private bool findIngatlan()
        {
            MatchResult m = RegexEntityMatcher.getBestMatchInInterval(matchesDict["INGATLANSZOVEG"], start, end);
            if (m != null)
            {
                findable.Ingatlan = IngatlanFinder.Instance.findOne(matchesDict, m.BeginIndex, m.EndIndex);
                if (findable.Ingatlan != null)
                {
                    findable.IngatlanAz = findable.Ingatlan.Az;
                    findable.TargyKod = findable.Ingatlan.TargyTipus;
                    return true;
                }
            }
            return false;
        }
        private bool findGepjarmu()
        {
            MatchResult m = RegexEntityMatcher.getBestMatchInInterval(matchesDict["JARMUSZOVEG"], start, end);
            if (m != null)
            {
                findable.Gepjarmu = GepjarmuFinder.Instance.findOne(matchesDict, m.BeginIndex, m.EndIndex);
                if (findable.Gepjarmu != null)
                {
                    findable.GepjarmuAz = findable.Gepjarmu.Az;
                    findable.TargyKod = findable.Gepjarmu.TargyTipus;
                    return true;
                }
            }
            return false;
        }
        private bool findteljesites()
        {
            findable.Teljesites = TeljesitesFinder.Instance.findOne(matchesDict, start, end);
            if (findable.Teljesites != null)
            {
                findable.TeljesitesAz = findable.Teljesites.Az;
                return true;
            }
            return false;
        }
        protected override void formatAttribute()
        {
            findable.TargyKod = TextAnalyzer.Instance.firstCap(findable.TargyKod);
        }
        private bool findArany()
        {
            MatchResult m = RegexEntityMatcher.getBestMatchInInterval(matchesDict["ELADOTULAJDONARANY"], start, end);
            if (m != null)
            {
                findable.EladoTulajdonArany = m.Value.Replace(" ", ".");
                return true;
            }
            return false;
        }
        protected override void validateFindable() { }
        protected override List<SzerzodesTargy> getDatabaseElements()
        {
            return dbHandler.getSzerzodesTargyak();
        }
        protected override void insertFindableToDatabase()
        {
            dbHandler.insertSzerzodesTargyToDatabase(findable);
        }
    }
}
