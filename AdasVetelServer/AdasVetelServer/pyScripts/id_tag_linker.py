id2tag = {
    "0": "O",
    "1": "B-PER",
    "2": "I-PER",
    "3": "B-ORG",
    "4": "I-ORG",
    "5": "B-LOC",
    "6": "I-LOC",
    "7": "B-CAR"
}

tag2Id = {
    "B-CAR": 7,
    "B-LOC": 5,
    "B-ORG": 3,
    "B-PER": 1,
    "I-LOC": 6,
    "I-ORG": 4,
    "I-PER": 2,
    "O": 0
}

label_list = [
    'O',
    'B-PER',
    'I-PER',
    'B-ORG',
    'I-ORG',
    'B-LOC',
    'I-LOC',
    'B-CAR',
]


def convertTags2IdBasic(tagList):
    idLabelList = []
    for t in tagList:
        idLabelList.append(tag2Id[t])
    return idLabelList


def convertTags2Id(tagLists):
    idLabelDoc = []
    idLabelList = []

    for tl in tagLists:
        for t in tl:
            idLabelList.append(tag2Id[t])
        idLabelDoc.append(idLabelList)
        idLabelList = []
    return idLabelDoc
