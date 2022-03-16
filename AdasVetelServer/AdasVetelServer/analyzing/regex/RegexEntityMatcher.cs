using AdasVetelServer.logger;
using AdasVetelServer.logic.regex;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;

namespace AdasVetelServer.logic.nep
{
    class RegexEntityMatcher
    {
        private readonly EntityPattern ep = null;
        private readonly string filePath;
        private readonly string patternSeparator;

        private readonly RegexOptions patternOption = RegexOptions.None;

        public RegexEntityMatcher(string contextFilePath, string patternSeparator = " ")
        {
            this.patternSeparator = patternSeparator;
            filePath = contextFilePath;
            if (!File.Exists(filePath) || Path.GetExtension(filePath) != ".xml")
            {
                ServerLogger.Instance.writeLog("Path doesn't exist");
                return;
            }
            ep = EntityPattern.load(filePath);
            if (ep.PatternCaseInsensitive)
                patternOption = RegexOptions.IgnoreCase;

        }
        public string getEntityPatternTag() {
            return ep.Tag;
        }

        public List<MatchResult> findMatchesByPattern(string plaintext) {
            if (ep == null)
            {
                ServerLogger.Instance.writeLog("Path doesn't exist");
                return new List<MatchResult>();
            }
            //find matches in words to patterns
            List<MatchResult> matches = new List<MatchResult>();

            foreach (string pattern in ep.Patterns)
            {
                foreach (Match match in Regex.Matches(plaintext, pattern, patternOption))
                {
                    MatchResult newMatch = new MatchResult(match.Value, match.Index, match.Index + match.Length - 1, ep.Tag, ep.OnlyPattern ? 1 : 0);
                    if (matches.Find(m => m.valueEquals(newMatch)) == null) { 
                        matches.Add(newMatch);
                    }
                }
            }

            return matches;
        }

        public bool validateMatchesBySimpleContext(Dictionary<string, List<MatchResult>> matchesDict, string plaintext)
        {
            if (ep == null)
            {
                ServerLogger.Instance.writeLog("Path doesn't exist");
                return false;
            }

            if (ep.OnlyPattern) {
                return false;
            }

            bool changed = false;
            List<MatchResult> matchesToValidate = matchesDict[ep.Tag];

            // add weight to remaining matches depending on their context
            List<Context> validSimpleContexts = ep.ContextHolder.getValidSimpleContexts();
            foreach (MatchResult match in matchesToValidate)
            {
                foreach (Context context in validSimpleContexts)
                {
                    if (context.isContextOf(match, matchesDict, plaintext, patternSeparator).isContext && !match.foundContexts.Contains(context)) {
                        match.foundContexts.Add(context);
                        match.Weight++;
                        changed = true;
                    }
                }
            }
            List<Context> inValidSimpleContexts = ep.ContextHolder.getInvalidSimpleContexts();
            // take weight to remaining matches depending on their context
            for (int i = 0; i < matchesToValidate.Count; i++)
            {
                foreach (Context context in inValidSimpleContexts)
                {
                    if (context.isContextOf(matchesToValidate[i], matchesDict, plaintext, patternSeparator).isContext)
                    {                        
                        changed = true;
                        matchesToValidate.RemoveAt(i--);
                        break;
                    }
                }
            }            
            return changed;
        }

        public bool validateMatchesbySpecialContext(Dictionary<string, List<MatchResult>> matchesDict, string plaintext) {
            if (ep == null)
            {
                ServerLogger.Instance.writeLog("Path doesn't exist");
                return false;
            }

            if (ep.OnlyPattern)
            {
                return false;
            }

            bool changed = false;
            List<MatchResult> matchesToValidate = matchesDict[ep.Tag];

            List<Context> validSpecialContexts = ep.ContextHolder.getValidSpecialContexts();
            foreach (MatchResult match in matchesToValidate)
            {                
                foreach (Context context in validSpecialContexts)
                {                       
                    if (context.isContextOf(match, matchesDict, plaintext, patternSeparator).isContext && !match.foundContexts.Contains(context)) {
                        match.foundContexts.Add(context);
                        match.Weight++;
                        changed = true;
                    }                 
                }                                
            }

            List<Context> inValidSpecialContexts = ep.ContextHolder.getInvalidSpecialContexts();
            for (int i = 0; i < matchesToValidate.Count; i++)
            {
                foreach (Context context in inValidSpecialContexts)
                {
                    if (context.isContextOf(matchesToValidate[i], matchesDict, plaintext, patternSeparator).isContext)
                    {
                        changed = true;
                        matchesToValidate.RemoveAt(i--);
                        break;
                    }
                }
            }
            return changed;
        }
        public static List<MatchResult> getMatchesByInterval(List<MatchResult> matches, int start = int.MinValue, int end = int.MaxValue) {
            matches.Sort(delegate (MatchResult m1, MatchResult m2) { return m1.BeginIndex.CompareTo(m2.BeginIndex); });
            return matches.FindAll(m => m.BeginIndex >= start && m.EndIndex <= end);        
        }
        public static MatchResult getBestMatch(List<MatchResult> matches) {
            if (matches.Count == 0) {
                return null;
            }
            matches.Sort(delegate (MatchResult m1, MatchResult m2) { return -1 * m1.Weight.CompareTo(m2.Weight); });
            return matches[0];
        }
        public static MatchResult getBestMatchInInterval(List<MatchResult> matches, int start = int.MinValue, int end = int.MaxValue)
        {
            return getBestMatch(getMatchesByInterval(matches, start, end));
        }
        public static MatchResult getMatchByStartIndex(List<MatchResult> matches, int start)
        {
            return matches.Find(m => m.BeginIndex == start);
        }
    }
}
