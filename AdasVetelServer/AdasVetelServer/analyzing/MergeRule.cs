using AdasVetelServer.bert;
using AdasVetelServer.logic.nep;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AdasVetelServer.logic.analyzing
{
    class MergeRule
    {
        public delegate void Action(List<MatchResult> mrs, NerResult nr);
        public delegate bool Condition(List<MatchResult> mrs, NerResult nr);

        private string[] nerTags;
        private Action action;
        private Condition condition;

        public MergeRule(string[] nerTags, Condition condition, Action action) {
            this.nerTags = nerTags;
            this.condition = condition;
            this.action = action;
        }

        public void execute(List<MatchResult> mrs, NerResult nr) {
            if (condition(mrs, nr) && nerTags.Contains(nr.entity)) {
                action(mrs, nr);
            }
        }
    }
}
