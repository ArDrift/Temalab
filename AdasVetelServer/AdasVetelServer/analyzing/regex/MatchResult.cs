using AdasVetelServer.logic.regex;
using System.Collections.Generic;

namespace AdasVetelServer.logic.nep
{
    public class MatchResult
    {
        public string Value { get; }
        public int BeginIndex { get; }
        public int EndIndex { get; }
        public int Weight { get; set; } = 0;
        public string Tag { get; }
        public List<Context> foundContexts = new List<Context>();

        public MatchResult(string value, int beginIndex, int endIndex, string tag, int weight = 0)
        {
            Value = value;
            BeginIndex = beginIndex;
            EndIndex = endIndex;
            Tag = tag;
            Weight = weight;
        }
        public override string ToString()
        {
            return $"{Value}, {Tag}, {BeginIndex}, {EndIndex}, {Weight}";
        }
        public bool valueEquals(MatchResult otherMatch) {
            return Value == otherMatch.Value &&
                BeginIndex == otherMatch.BeginIndex &&
                EndIndex == otherMatch.EndIndex &&
                Tag == otherMatch.Tag;
        }
    }
}
