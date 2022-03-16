using AdasVetelServer.bert;
using AdasVetelServer.logic.nep;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AdasVetelServer.logic.analyzing
{
    static class MergeRuleHolder
    {       
        public static void executeMergeRules(List<MatchResult> mrs, NerResult nr) {
            foreach (MergeRule mergeRule in mergeRules) {                
                    mergeRule.execute(mrs, nr);
            }
        }    
        
        private static List<MergeRule> mergeRules = new List<MergeRule>() {

            new MergeRule(new[]{"PER", "ORG"},
                delegate(List<MatchResult>mrs, NerResult nr){
                    return nr.certancy == NerResult.Certancy.Certain &&
                    RegexEntityMatcher.getMatchByStartIndex(mrs,nr.start) != null;
                },
                delegate(List<MatchResult>mrs, NerResult nr){RegexEntityMatcher.getMatchByStartIndex(mrs,nr.start).Weight++; }
            )
           
        };
    }
}
