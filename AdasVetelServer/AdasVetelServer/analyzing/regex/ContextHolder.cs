using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace AdasVetelServer.logic.regex
{
    public enum ContextPosition{ 
        Before,
        After
    };
    public class ContextHolder
    {
        private List<Context> simpleContexts = new List<Context>();
        private List<Context> specialContexts = new List<Context>();

        public ContextHolder(List<string> contextsAfter, List<string> contextsBefore, List<string> invalidsAfter, List<string> invalidsBefore) {
            registerContext(contextsAfter, true, ContextPosition.After);
            registerContext(contextsBefore, true, ContextPosition.Before);
            registerContext(invalidsAfter, false, ContextPosition.After);
            registerContext(invalidsBefore, false, ContextPosition.Before);
        }
        private void registerContext(List<string> contexts, bool isValid, ContextPosition position) {
            foreach (string context in contexts)
            {
                if (Context.isSpecialContext(context))
                {
                    specialContexts.Add(new ComplexContext(context, isValid, position));
                }
                else
                {
                    simpleContexts.Add(new SimpleContext(context, isValid, position));
                }
            }
        }
     
        public List<Context> getInvalidSpecialContexts() {
            return specialContexts.FindAll(c => !c.IsValid);
        }
        public List<Context> getValidSpecialContexts() {
            return specialContexts.FindAll(c => c.IsValid);
        }
        public List<Context> getInvalidSimpleContexts() {
            return simpleContexts.FindAll(c => !c.IsValid);
        }
        public List<Context> getValidSimpleContexts() {
            return simpleContexts.FindAll(c => c.IsValid);
        }
    }


}
