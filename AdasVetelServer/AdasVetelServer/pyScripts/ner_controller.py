import sys
import ner
import codecs

def main():
    if len(sys.argv) == 5:
        if sys.argv[1] == "predict":
            myNer = ner.MyNer(sys.argv[2])
            text = sys.argv[3].split()
            textToPredict = ""
            output = "["
            for i, w in enumerate(text):
                textToPredict += w + " "
                if i == len(text) - 1 or i + 1 % 500 == 0:
                    result = myNer.predict(textToPredict.rstrip(' '))
                    textToPredict = ""
                    output += result.strip('[] ') + ','

            output = output.strip(',') + "]"
            with codecs.open(sys.argv[4], 'w', "utf-8") as f:
                f.write(output)


if __name__ == "__main__":
    main()
