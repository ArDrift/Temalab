using AdasVetelServer.logic.nep;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace AdasVetelServer.logic.regex
{
    public class SpecialContext : Context
    {
        public SpecialContext(string pattern, bool isValid, ContextPosition position) : base(pattern, isValid, position)
        {
            Pattern = getTagFromSpecialContext(pattern);
        }
        private string getTagFromSpecialContext(string context)
        {
            context = context.TrimStart('@');
            context = context.Trim('{', '}');
            return context;
        }
        public override (bool isContext, string pattern) isContextOf(MatchResult match, Dictionary<string, List<MatchResult>> inMatchesDict, string inText, string separator)
        {            
            List<MatchResult> matchesByTag = inMatchesDict[Pattern];
            string foundPattern = "";
            switch (Position)
            {
                case ContextPosition.After:
                    {
                        matchesByTag = matchesByTag.FindAll(m => m.Weight > 0 && match.EndIndex + separator.Length + 1 == m.BeginIndex);
                        if (matchesByTag.Count == 0)
                        {
                            return (false, "");
                        }
                        foundPattern = match + separator + matchesByTag[0].Value;
                        break;
                       
                    }
                case ContextPosition.Before:
                    {
                        matchesByTag = matchesByTag.FindAll(m => m.Weight > 0 && match.BeginIndex - separator.Length - 1 == m.EndIndex);
                        if (matchesByTag.Count == 0)
                        {
                            return (false, "");
                        }
                        foundPattern = matchesByTag[0].Value + separator + match;
                        break;
                    }
                default:
                    return (false, "");
            }           
            return (true, foundPattern);
        }
    }
}
