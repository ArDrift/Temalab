using AdasVetelServer.bert;
using AdasVetelServer.dbHandler;
using AdasVetelServer.logger;
using AdasVetelServer.logic.analyzing;
using AdasVetelServer.logic.nep;
using AdasVetelServer.model;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Windows.Forms;

namespace AdasVetelServer.logic
{
    public class TextAnalyzer
    {
        public DbHandler dbh { get; set; }
        public static TextAnalyzer Instance { get; } = new TextAnalyzer();
        private TextAnalyzer()
        { }

        private string fileName = "";
        private string text = "";
        private Dictionary<string, List<MatchResult>> matchesDict;
        private List<NerResult> nerResults;
        public string formatData(string data)
        {
            data = data.Replace(".", " ");
            data = data.Replace("-", " ");
            data = data.Replace(",", " ,");
            data = data.Replace(":", " ");
            data = data.Replace("\"", " ");
            data = data.Replace("˝", " ");
            data = data.Replace(";", " ");
            data = data.Replace("(", " ");
            data = data.Replace(")", " ");
            data = data.Replace("[", " ");
            data = data.Replace("]", " ");
            data = data.Replace("§", " ");
            data = data.Replace("\t", " ");
            data = data.Replace("\n", " ");
            Regex regex = new Regex("[ ]{2,}", RegexOptions.None);
            data = regex.Replace(data, " ");
            //data = data.Replace("\n \n", "\n\n");

            return data;
        }
        public void analyzeText(string data, string filename)
        {
            fileName = Path.GetFileName(filename);
            text = formatData(data);

            NerExecuter nerExecuter = new NerExecuter(text);
            nerExecuter.start();


            matchesDict = getAllMatches(text);

            ServerLogger.Instance.writeLog(text);
            ServerLogger.Instance.writeLog();


            
            nerExecuter.waitForExecutionEnd();
            nerResults = nerExecuter.Results;
            mergeNerIntoRegexMatches();

            ServerLogger.Instance.writeLog();
            ServerLogger.Instance.writeLog(nerResults);
            ServerLogger.Instance.writeLog();
            ServerLogger.Instance.writeLog(matchesDict);

            using (dbh = new DbHandler())
            {
               Szerzodes szerzodes = findSzerzodes();
                dbh.submintChanges();
            }            
        }
        private Szerzodes findSzerzodes()
        {
            SzerzodesFinder.Instance.FileName = fileName;
            return SzerzodesFinder.Instance.findOne(matchesDict, 0, text.Length - 1);
        }
       
        public string firstCap(string s)
        {
            // Check for empty string.
            if (string.IsNullOrEmpty(s))
            {
                return string.Empty;
            }
            // Return char and concat substring.
            return char.ToUpper(s[0]) + s.Substring(1);
        }

 
        private Dictionary<string, List<MatchResult>> getAllMatches(string text)
        {

            Dictionary<string, List<MatchResult>> matchesDictionary = new Dictionary<string, List<MatchResult>>();
            List<RegexEntityMatcher> matchers = new List<RegexEntityMatcher>();
            foreach (string filePath in Directory.EnumerateFiles(Application.StartupPath + "\\entityPatterns", "*.xml", SearchOption.AllDirectories))
            {
                RegexEntityMatcher rem = new RegexEntityMatcher(filePath, " ");
                matchers.Add(rem);
                matchesDictionary.Add(rem.getEntityPatternTag(), rem.findMatchesByPattern(text));
                rem.validateMatchesBySimpleContext(matchesDictionary, text);
            }
            bool needToRepeat = true;
            while (needToRepeat)
            {
                needToRepeat = false;
                foreach (RegexEntityMatcher matcher in matchers)
                {
                    if (matcher.validateMatchesbySpecialContext(matchesDictionary, text)) {
                        needToRepeat = true;
                    }
                }
            }
            foreach (var matchesRecord in matchesDictionary)
            {
                matchesRecord.Value.RemoveAll(m => m.Weight == 0);
            }
            return matchesDictionary;
        }
        private void mergeNerIntoRegexMatches() {
            foreach (NerResult nerResult in nerResults) {
                MergeRuleHolder.executeMergeRules(matchesDict["UTONEV"], nerResult);
                MergeRuleHolder.executeMergeRules(matchesDict["ANYJANEVE"], nerResult);
            }                    
        }
    }
}
