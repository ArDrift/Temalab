using AdasVetelServer.logic.nep;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace AdasVetelServer.logic.regex
{
    public class SimpleContext : Context
    {    

        public SimpleContext(string pattern, bool isValid, ContextPosition position) : base(pattern, isValid, position) {
        
        }  

        public override (bool isContext, string pattern) isContextOf(MatchResult match, Dictionary<string, List<MatchResult>> inMatchesDict, string inText, string separator)
        {           
            switch (Position) {
                case ContextPosition.After:
                    { 
                        Match m = Regex.Match(inText.Substring(match.BeginIndex), $"{match.Value}{separator}{Pattern}", RegexOptions.IgnoreCase);
                        return (m.Success && m.Index == 0, m.Value);                   
                    }
                case ContextPosition.Before:
                    {
                        string textFrag = inText.Substring(0, match.EndIndex + 1);
                        MatchCollection ms = Regex.Matches(textFrag, $"{Pattern}{separator}{match.Value}", RegexOptions.IgnoreCase);
                        if (ms.Count > 0) { 
                            return (ms[ms.Count - 1].Index + ms[ms.Count - 1].Length == textFrag.Length, ms[ms.Count-1].Value);                        
                        }
                        return (false, "");
                    }
                default:
                    return (false, "");
            }
        }
    }
}
