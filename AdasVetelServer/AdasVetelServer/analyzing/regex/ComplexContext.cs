using AdasVetelServer.logic.nep;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace AdasVetelServer.logic.regex
{
    public class ComplexContext : Context
    {
        List<Context> innerContexts = new List<Context>();
        public ComplexContext(string pattern, bool isValid, ContextPosition position) : base(pattern, isValid, position)
        {
            foreach (string inner in splittPattern(pattern))
            {
                if (isSpecialContext(inner))
                {
                    innerContexts.Add(new SpecialContext(inner, true, Position));
                }
                else
                {
                    innerContexts.Add(new SimpleContext(inner, true, Position));
                }
            }
        }
        public override (bool isContext, string pattern) isContextOf(MatchResult match, Dictionary<string, List<MatchResult>> inMatchesDict, string inText, string separator)
        {
            MatchResult tempMatch = match;
            switch (Position)
            {
                case ContextPosition.After:
                    {
                        foreach (Context context in innerContexts)
                        {
                            var isContextOf = context.isContextOf(tempMatch, inMatchesDict, inText,separator);
                            if (!isContextOf.isContext) {
                                return (false, "");
                            }
                            string newValue = isContextOf.pattern;
                            tempMatch = new MatchResult(newValue, tempMatch.BeginIndex, tempMatch.BeginIndex + newValue.Length -1, tempMatch.Tag, tempMatch.Weight);
                            separator = "";
                        }
                        break;

                    }
                case ContextPosition.Before:
                    {
                        foreach (Context context in innerContexts.Reverse<Context>())
                        {
                            var isContextOf = context.isContextOf(tempMatch, inMatchesDict, inText, separator);
                            if (!isContextOf.isContext)
                            {
                                return (false, "");
                            }
                            string newValue = isContextOf.pattern;
                            tempMatch = new MatchResult(newValue, tempMatch.EndIndex - (newValue.Length - 1), tempMatch.EndIndex, tempMatch.Tag, tempMatch.Weight);
                            separator = "";
                        }
                        break;
                    }
                default:
                    return (false, "");
            }
            return (true, tempMatch.Value);
        }
        private List<string> splittPattern(string pattern) {
            List<string> patterns = new List<string>();
            var matches = Regex.Matches(pattern, "@{[a-z]+}", RegexOptions.IgnoreCase);
            int startIndex = 0;
            foreach (Match match in matches)
            {
                string frag = "";
                while (startIndex < match.Index) {
                    frag += pattern[startIndex++];
                }
                if (frag != "") {
                    patterns.Add(frag);
                }
                patterns.Add(match.Value);
                startIndex = match.Index + match.Value.Length;
            }
            if (startIndex < pattern.Length) {
                patterns.Add(pattern.Substring(startIndex));
            }
            return patterns;
        }
    }
}
