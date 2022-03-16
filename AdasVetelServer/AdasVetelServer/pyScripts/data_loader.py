import pandas as pd


def convertToLists(_tokens, _labels, sample_len):
    tokenDoc = []
    labelDoc = []
    tokenList = []
    labelList = []
    for j, t in enumerate(_tokens):
        tokenList.append(_tokens[j].strip())
        labelList.append(_labels[j])
        if t == '.' or t == '?' or t == '!':
            labelDoc.append(labelList)
            tokenDoc.append(tokenList)
            labelList = []
            tokenList = []
            if j >= sample_len:
                break

    return tokenDoc, labelDoc


def loadDataset(source):
    df = pd.read_csv(source, delimiter=',', header=None, names=['token', 'label'], encoding='UTF-8-sig')
    print('Number of training rows: {:,}\n'.format(df.shape[0]))
    print(df.sample(10))
    return df


def loadShapeAndSaveDataset(source, out_path):
    df = pd.read_csv(source, delimiter='\t', header=None, names=['token', 'label'])
    for j, row in df.iterrows():
        df.loc[j, "token"] = (row['token'])[3:]

    print('Number of training rows: {:,}\n'.format(df.shape[0]))

    df.to_csv(out_path)