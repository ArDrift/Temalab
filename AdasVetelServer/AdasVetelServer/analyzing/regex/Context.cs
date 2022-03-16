using AdasVetelServer.logic.nep;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace AdasVetelServer.logic.regex
{
    public abstract class Context
    {
        public bool IsValid { get; set; }
        public string Pattern { get; set; }
        public ContextPosition Position {get;set;}

        public Context(string pattern, bool isValid, ContextPosition position) {
            Pattern = pattern;
            IsValid = isValid;
            Position = position;
        }

        public abstract (bool isContext, string pattern) isContextOf(MatchResult match, Dictionary<string, List<MatchResult>> inMatchesDict, string inText, string separator);
        public static bool isSpecialContext(string context)
        {
            return Regex.IsMatch(context, "@{[a-z]+}", RegexOptions.IgnoreCase);
        }
    }
}
