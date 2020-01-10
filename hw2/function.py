# Created by haiphung106

def LoadingData(fileName):
    with open(fileName, 'rb') as file:
        magic = int.from_bytes(file.readline(4), "big")
        images = int.from_bytes(file.readline(4), "big")
        nrows = int.from_bytes(file.readline(4), "big")
        ncols = int.from_bytes(file.readline(4), "big")

        Number0fImages = []
        for i in range(images):
            tmpRows = []
            for j in range(nrows):
                tmpColumns = []
                for k in range(ncols):
                    pixelValue = int.from_bytes(file.readline(1), "big")
                    # if (pixelValue<128):
                    tmpColumns.append(pixelValue)
                    # else:
                    #     tmpColumns.append(1)
                tmpRows.append(tmpColumns)
            Number0fImages.append(tmpRows)
        return Number0fImages


def LoadingLabel(fileName):
    with open(fileName, 'rb') as file:
        magic = int.from_bytes(file.readline(4), 'big')
        label = int.from_bytes(file.readline(4), "big")
        LabelValue = []
        for i in range(label):
            pixelValue = int.from_bytes(file.readline(1), "big")
            # if (pixelValue<128):
            LabelValue.append(pixelValue)
            # else:
            #     LabelValue.append(1)
        return LabelValue


def factorial(x):
    if x > 2:
        return x * factorial(x-1)
    else:
        return 2

