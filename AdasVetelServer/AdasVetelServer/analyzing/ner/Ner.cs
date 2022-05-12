using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Onnx;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AdasVetelServer.analyzing.ner
{
   public class Trainer
    {
        public readonly MLContext mlContext;
        public Trainer()
        {
            mlContext = new MLContext();
            
        }

        public ITransformer BuildAndTrain(string bertModelPath, bool useGpu)
        {
            var pipeline = mlContext.Transforms
                            .ApplyOnnxModel(modelFile: bertModelPath,
                                            shapeDictionary: new Dictionary<string, int[]>
                                            {
                                                { "input_ids", new [] { 1, 512 } },
                                                { "attention_mask", new [] { 1, 512 } },
                                                  { "token_type_ids", new [] { 1, 512 } },
                                                  { "logits", new [] { 1, 512, 8 } }
                                                //{ "last_hidden_state", new [] { 1, 512, 1 } },
                                                //{ "1607", new [] { 1, 768 } },
                                            },
                                            inputColumnNames: new[] {"input_ids",
                                                                     "attention_mask",
                                                               "token_type_ids"},
                                            outputColumnNames: new[] { "logits" }, //"last_hidden_state",
                                                              //"1607"},
                                            gpuDeviceId: useGpu ? 0 : (int?)null,
                                            fallbackToCpu: true);
            
            return pipeline.Fit(mlContext.Data.LoadFromEnumerable(new List<BertInput>()));
        }
        
    }

    public class Predictor
    {
        private MLContext _mLContext;
        private PredictionEngine<BertInput, BertPredictions> _predictionEngine;

        public Predictor(ITransformer trainedModel)
        {
            _mLContext = new MLContext();
            _predictionEngine = _mLContext.Model
                                          .CreatePredictionEngine<BertInput, BertPredictions>(trainedModel);
        }

        public BertPredictions Predict(BertInput encodedInput)
        {
            return _predictionEngine.Predict(encodedInput);
        }
    }

    public static class FileReader
    {
        public static List<string> ReadFile(string filename)
        {
            var result = new List<string>();

            using (var reader = new StreamReader(filename))
            {
                string line;

                while ((line = reader.ReadLine()) != null)
                {
                    if (!string.IsNullOrWhiteSpace(line))
                    {
                        result.Add(line);
                    }
                }
            }

            return result;
        }
    }

    public static class SoftmaxEnumerableExtension
    {
        public static IEnumerable<(T Item, float Probability)> Softmax<T>(
                                            this IEnumerable<T> collection,
                                            Func<T, float> scoreSelector)
        {
            var maxScore = collection.Max(scoreSelector);
            var sum = collection.Sum(r => Math.Exp(scoreSelector(r) - maxScore));

            return collection.Select(r => (r, (float)(Math.Exp(scoreSelector(r) - maxScore) / sum)));
        }
    }

    public class Tokens
    {
        public const string Padding = "";
        public const string Unknown = "[UNK]";
        public const string Classification = "[CLS]";
        public const string Separation = "[SEP]";
        public const string Mask = "[MASK]";
    }
    public class TokenizerBase
    {
        public List<string> _vocabulary;
        
        protected readonly Dictionary<string, int> _vocabularyDict;

        public TokenizerBase(string vocabularyFilePath)
        {
            _vocabulary = VocabularyReader.ReadFile(vocabularyFilePath);

            _vocabularyDict = new Dictionary<string, int>();
            for (int i = 0; i < _vocabulary.Count; i++)
                _vocabularyDict[_vocabulary[i]] = i;
        }

        
        public List<List<(long InputIds, long TokenTypeIds, long AttentionMask)>> Encode(int sequenceLength, params string[] texts)
        {
            var tokens = Tokenize(texts);

            var tokensList = new List<List<(string Token, int VocabularyIndex, long SegmentIndex)>>();
            for (int i = 0; i < tokens.Count; i+=512)
            {
                tokensList.Add(tokens.Skip(i).Take(512).ToList());
            }

            var paddingList = new List<List<long>>();
            foreach (var token in tokensList)
            {
                paddingList.Add(Enumerable.Repeat(0L, sequenceLength - token.Count).ToList());
            }

            var tokenIndexes = new List<long>();
            var segmentIndexes = new List<long>();
            var inputMask = new List<long>();
            var output = new List<List<System.Tuple<long, long, long>>>();

            for (int i = 0; i < tokensList.Count; i++)
            {
                tokenIndexes = tokensList[i].Select(token => (long)token.VocabularyIndex).Concat(paddingList[i]).ToList();
                segmentIndexes = tokensList[i].Select(token => token.SegmentIndex).Concat(paddingList[i]).ToList();
                inputMask = tokensList[i].Select(o => 1L).Concat(paddingList[i]).ToList();

                output.Add(tokenIndexes.Zip(segmentIndexes, Tuple.Create)
                .Zip(inputMask, (t, z) => Tuple.Create(t.Item1, t.Item2, z)).ToList());
            }
            /*
            var tokenIndexes = tokens.Select(token => (long)token.VocabularyIndex).Concat(padding).ToArray();
            var segmentIndexes = tokens.Select(token => token.SegmentIndex).Concat(padding).ToArray();
            var inputMask = tokens.Select(o => 1L).Concat(padding).ToArray();*/
            /*
            var output = tokenIndexes.Zip(segmentIndexes, Tuple.Create)
                .Zip(inputMask, (t, z) => Tuple.Create(t.Item1, t.Item2, z));*/

            return output.Select(l => l.Select(x => (InputIds: x.Item1, TokenTypeIds: x.Item2, AttentionMask: x.Item3)).ToList()).ToList();
            
        }

        public string IdToToken(int id)
        {
            return _vocabulary[id];
        }

        public List<string> Untokenize(List<string> tokens)
        {
            var currentToken = string.Empty;
            var untokens = new List<string>();
            tokens.Reverse();

            tokens.ForEach(token =>
            {
                if (token.StartsWith("##"))
                {
                    currentToken = token.Replace("##", "") + currentToken;
                }
                else
                {
                    currentToken = token + currentToken;
                    untokens.Add(currentToken);
                    currentToken = string.Empty;
                }
            });

            untokens.Reverse();

            return untokens;
        }

        public List<(string Token, int VocabularyIndex, long SegmentIndex)> Tokenize(params string[] texts)
        {
            IEnumerable<string> tokens = new string[] { Tokens.Classification };

            foreach (var text in texts)
            {
                tokens = tokens.Concat(TokenizeSentence(text));
                tokens = tokens.Concat(new string[] { Tokens.Separation });
            }

            var tokenAndIndex = tokens
                .SelectMany(TokenizeSubwords)
                .ToList();

            var segmentIndexes = SegmentIndex(tokenAndIndex);

            return tokenAndIndex.Zip(segmentIndexes, (tokenindex, segmentindex)
                                => (tokenindex.Token, tokenindex.VocabularyIndex, segmentindex)).ToList();
        }

        private IEnumerable<long> SegmentIndex(List<(string token, int index)> tokens)
        {
            var segmentIndex = 0;
            var segmentIndexes = new List<long>();

            foreach (var (token, index) in tokens)
            {
                segmentIndexes.Add(segmentIndex);

                if (token == Tokens.Separation)
                {
                    segmentIndex++;
                }
            }

            return segmentIndexes;
        }
        

        private IEnumerable<(string Token, int VocabularyIndex)> TokenizeSubwords(string word)
        {
            if (_vocabularyDict.ContainsKey(word))
            {
                return new (string, int)[] { (word, _vocabularyDict[word]) };
            }

            var tokens = new List<(string, int)>();
            var remaining = word;

            while (!string.IsNullOrEmpty(remaining) && remaining.Length > 2)
            {
                string prefix = null;
                int subwordLength = remaining.Length;
                while (subwordLength >= 2)
                {
                    string subword = remaining.Substring(0, subwordLength);
                    if (!_vocabularyDict.ContainsKey(subword))
                    {
                        subwordLength--;
                        continue;
                    }

                    prefix = subword;
                    break;
                }

                if (prefix == null)
                {
                    tokens.Add((Tokens.Unknown, _vocabularyDict[Tokens.Unknown]));

                    return tokens;
                }

                remaining = remaining.Replace(prefix, "##");

                tokens.Add((prefix, _vocabularyDict[prefix]));
            }

            if (!string.IsNullOrWhiteSpace(word) && !tokens.Any())
            {
                tokens.Add((Tokens.Unknown, _vocabularyDict[Tokens.Unknown]));
            }

            return tokens;
        }

        private IEnumerable<string> TokenizeSentence(string text)
        {
            // remove spaces and split the , . : ; etc..
            return text.Split(new string[] { " ", "   ", "\r\n" }, StringSplitOptions.None)
                .SelectMany(o => StringExtension.SplitAndKeep(o, ".,;:\\/?!#$%()=+-*\"'–_`<>&^@{}[]|~'".ToArray()));
       
        }
    }

    public static class StringExtension
    {
        public static IEnumerable<string> SplitAndKeep(
                              this string inputString, params char[] delimiters)
        {
            int start = 0, index;

            while ((index = inputString.IndexOfAny(delimiters, start)) != -1)
            {
                if (index - start > 0)
                    yield return inputString.Substring(start, index - start);

               // yield return inputString.Substring(index, 1);

                start = index + 1;
            }

            if (start < inputString.Length)
            {
                yield return inputString.Substring(start);
            }
        }
    }

    public class Bert
    {
        private List<string> _vocabulary;

        private readonly TokenizerBase _tokenizer;
        private Predictor _predictor;

        public Bert(string vocabularyFilePath, string bertModelPath)
        {
            
            _tokenizer = new TokenizerBase(vocabularyFilePath);

            var trainer = new Trainer();
            var trainedModel = trainer.BuildAndTrain(bertModelPath, false);
            _predictor = new Predictor(trainedModel);
        }

        public (List<string> tokens, List<float> probabilities) Predict(string context)
        {
            var sentences = context.Split(new char[] {'.','?','!'}, StringSplitOptions.RemoveEmptyEntries);

            var tokens = new List<(string Token, int VocabularyIndex, long SegmentIndex)>();
            /*
            foreach (var sentence in sentences)
            {
                foreach (var item in _tokenizer.Tokenize(sentence))
                {
                    tokens.Add((item.Token, item.VocabularyIndex, item.SegmentIndex));
                }
            }*/
            tokens = _tokenizer.Tokenize(sentences);
            /*var tokens = _tokenizer.Tokenize(context);*/
            var encoded = _tokenizer.Encode(512, context);
            var input = new List<BertInput>();

            foreach (var tupleList in encoded)
            {
                input.Add(new BertInput()
                {
                    InputIds = tupleList.Select(t => t.InputIds).ToArray(),
                    AttentionMask = tupleList.Select(t => t.AttentionMask).ToArray(),
                    TokenTypeIds = tupleList.Select(t => t.TokenTypeIds).ToArray()
                });
            }
            /*
            var input =  new BertInput()
                {
                    InputIds = encoded.Select(t => t.InputIds).ToArray(),
                    AttentionMask = encoded.Select(t => t.AttentionMask).ToArray(),
                    TokenTypeIds = encoded.Select(t => t.TokenTypeIds).ToArray()
                };*/

            var predictionsList = new List<BertPredictions>();
            //var predictions = _predictor.Predict(input);

            foreach (var bertInput in input)
            {
                predictionsList.Add(_predictor.Predict(bertInput));
            }

            var contextStart = tokens.FindIndex(o => o.Token == Tokens.Separation);


            //var lastHiddenStateList = new List<float>();
            //var pollerOutPutList = new List<float>();
            var logitsList = new List<float>();
            foreach (var item in predictionsList)
            {
                // lastHiddenStateList.AddRange(item.LastHiddenState);
                //pollerOutPutList.AddRange(item.PollerOutput);
                logitsList.AddRange(item.Logits);
            }
            /*
            var reduceLast = new List<float>();
            for (int i = 0; i <= lastHiddenStateList.Count-512; i+=512)
            {
                reduceLast.Add(lastHiddenStateList[i]);
            }*/
            var predictions = new BertPredictions()
            {
                //LastHiddenState = reduceLast.ToArray(),
                //PollerOutput = pollerOutPutList.ToArray()
                Logits = logitsList.ToArray()
            };
            var predictionList = new List<(int, int, float)>();
            var separationsList =tokens.FindAll(o => o.Token == Tokens.Separation);


            for (int i = 0; i < separationsList.Count; i++)
            {
                //predictionList.Add(
                  //  GetBestPrediction(predictions, tokens.IndexOf(separationsList[i]), 20, 30));
            }
           // var (startIndex, endIndex, probability) = GetBestPrediction(predictions, contextStart, 20, 30);


            var InputIdsList = new List<long>();
            var AttentionMaskList = new List<long>();
            var TokenTypeIds = new List<long>();
            foreach (var item in input)
            {
                InputIdsList.AddRange(item.InputIds);
                AttentionMaskList.AddRange(item.AttentionMask);
                TokenTypeIds.AddRange(item.TokenTypeIds);
            }
            var inputSum = new BertInput()
            {
                InputIds = InputIdsList.ToArray(),
                AttentionMask = AttentionMaskList.ToArray(),
                TokenTypeIds = TokenTypeIds.ToArray()
            };

            var predictedTokensList = new List<List<(string,float)>>();
            foreach (var item in predictionList)
            {
                predictedTokensList.Add(inputSum.InputIds
                .Skip(item.Item1)
                .Take(item.Item2 + 1 - item.Item1)
                .Select(o =>( _tokenizer._vocabulary[(int)o],item.Item3))
                .ToList());
            }

            /*
            var predictedTokens = inputSum.InputIds
                .Skip( startIndex)
                .Take(endIndex + 1 - startIndex)
                .Select(o => _vocabulary[(int)o])
                .ToList();*/

            var connectedTokensList = new List<List<string>>();
            //var connectedTokens = _tokenizer.Untokenize(predictedTokens);

            foreach (var item in predictedTokensList)
            {
                connectedTokensList.Add(_tokenizer.Untokenize(item.Select(f=>f.Item1).ToList()));
                
            }
            var probabilitesList = new List<float>();
            foreach (var item in predictedTokensList)
            {
                var temp = item.Select(p => p.Item2).ToList();
                foreach (var item2 in temp)
                {
                    probabilitesList.Add(item2);
                }
                
            }
            var ctList = new List<string>();
            foreach (var item in connectedTokensList)
            {
                var temp = item.Select(p=>p);
                foreach (var item2 in temp)
                {
                    ctList.Add(item2);
                }

            }

            return (ctList, probabilitesList);
        }
        /*
        private BertInput BuildInput(List<(string Token, int Index)> tokens)
        {
            var padding = Enumerable.Repeat(0L, 256 - tokens.Count).ToList();

            var tokenIndexes = tokens.Select(token => (long)token.Index).Concat(padding).ToArray();
            
            var inputMask = tokens.Select(o => 1L).Concat(padding).ToArray();

            var typeIds = tokens.Select(token => token.t).ToArray();

            return new BertInput()
            
                InputIds = tokenIndexes,
                AttentionMask = inputMask,
                TokenTypeIds = 

            };
        }*/

        /*
        private (int StartIndex, int EndIndex, float Probability) GetBestPrediction(BertPredictions result, int minIndex, int topN, int maxLength)
        {
            var bestStartLogits = result.LastHiddenState
                .Skip(minIndex)
                .Select((logit, index) => (Logit: logit, Index: index))
                .OrderByDescending(o => o.Logit)
                .Take(topN);

            var bestEndLogits = result.PollerOutput
                .Skip(minIndex)
                .Select((logit, index) => (Logit: logit, Index: index))
                .OrderByDescending(o => o.Logit)
                .Take(topN);

            var bestResultsWithScore = bestStartLogits
                .SelectMany(startLogit =>
                    bestEndLogits
                    .Select(endLogit =>
                        (
                            StartLogit: startLogit.Index,
                            EndLogit: endLogit.Index,
                            Score: startLogit.Logit + endLogit.Logit
                        )
                     )
                )
               //.Where(entry => !(entry.EndLogit < entry.StartLogit || entry.EndLogit - entry.StartLogit > maxLength || entry.StartLogit == 0 && entry.EndLogit == 0 || entry.StartLogit < minIndex))
                .Take(topN);

            var (item, probability) = bestResultsWithScore
                .Softmax(o => o.Score)
                .OrderByDescending(o => o.Probability)
                .FirstOrDefault();

            return (StartIndex: item.StartLogit, EndIndex: item.EndLogit, probability);
        }*/
    }



    public class VocabularyReader
    {
        public static List<string> ReadFile(string filename)
        {
            var result = new List<string>();

            using (var reader = new StreamReader(filename))
            {
                string line;

                while ((line = reader.ReadLine()) != null)
                {
                    if (!string.IsNullOrWhiteSpace(line))
                    {
                        result.Add(line);
                    }
                }
            }

            return result;
        }
    }

    public class BertInput
    {
        [VectorType(1, 512)]
        [ColumnName("input_ids")]
        public long[] InputIds { get; set; }

        [VectorType(1, 512)]
        [ColumnName("attention_mask")]
        public long[] AttentionMask { get; set; }

        [VectorType(1, 512)]
        [ColumnName("token_type_ids")]
        public long[] TokenTypeIds { get; set; }
    }
    public class BertPredictions
    {
        /*
        [VectorType(1, 512, 1)]
        [ColumnName("last_hidden_state")]
        public float[] LastHiddenState { get; set; }

        [VectorType(1, 768)]
        [ColumnName("1607")]
        public float[] PollerOutput { get; set; }*/

        [VectorType(1, 512, 8)]
        [ColumnName("logits")]
        public float[] Logits { get; set; }

    }
}
