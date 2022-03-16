using AdasVetelServer.logic.nep;

namespace AdasVetelServer.bert
{
    class NerResult
    {
        public enum Certancy { 
            Uncertain,
            Medium,
            Certain
        }
        public Certancy certancy { get; set; }
        public string entity { get; set; }
        public double score { get; set; }
        public string word { get; set; }
        public int start { get; set; }
        public int end { get; set; }      

        public override string ToString()
        {
            return $"{entity}, {score}, {word}, {start}, {end}";
        }
        public MatchResult toMatchResult() {
            return new MatchResult(word, start, end, entity, (int)certancy);
        }

    }
}
