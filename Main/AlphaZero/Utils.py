def storeWeightsToFile(weights, fileName):
    import pickle
    data = pickle.dumps((weights, ))
    f = open(fileName, 'wb')
    f.write(data)
    f.close()

    print("Stored weights to file")


def getWeightsFromFile(fileName):
    import pickle
    f = open(fileName, 'rb')
    data = pickle.loads(f.read())[0]
    f.close()

    print("Loaded weights from file", len(data))
    return data
