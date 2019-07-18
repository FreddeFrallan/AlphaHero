from Main import Hyperparameters, MachineSpecificSettings
import keras


def _createBatchNormalizedConvBlock(inputNode, filters):
    temp = keras.layers.Conv2D(filters, kernel_size=(4, 4), strides=(1, 1), padding='SAME')(inputNode)
    temp = keras.layers.BatchNormalization()(temp)
    return keras.layers.ReLU()(temp)


def createResidualNetwork(inputShape, filtersPerConv, convPerResidual, amountOfResidualBlocks):
    inputLayer = keras.layers.Input(inputShape, name="InputLayer")
    l2Reg = Hyperparameters.L2_REGULARIZATION
    Conv2D = keras.layers.Conv2D

    # Build Residual Tower
    skipConnection = _createBatchNormalizedConvBlock(inputLayer, filtersPerConv)
    for i in range(amountOfResidualBlocks - 1):
        resBlock = skipConnection
        for j in range(convPerResidual):
            resBlock = Conv2D(filtersPerConv, kernel_size=(4, 4), strides=(1, 1), padding='SAME')(
                resBlock)
            resBlock = keras.layers.BatchNormalization()(resBlock)

            # If it's the last conv Layer in the residual block we wait with adding the ReLU until we have
            # added the skip connection
            if (j + 1 < convPerResidual):
                resBlock = keras.layers.ReLU()(resBlock)

        skipConnection = keras.layers.add([skipConnection, resBlock])
        skipConnection = keras.layers.ReLU()(skipConnection)

    # Evaluation Head
    evalHead = _createBatchNormalizedConvBlock(skipConnection, 32)
    evalHead = keras.layers.Flatten()(evalHead)
    evalHead = keras.layers.Dense(1, activation='sigmoid', name='ValueOut')(evalHead)

    # Policy Head
    policyHead = _createBatchNormalizedConvBlock(skipConnection, 32)
    policyHead = keras.layers.Flatten()(policyHead)
    policyHead = keras.layers.Dense(7, activation='softmax', name='PolicyOut')(policyHead)

    # Create Full Model
    model = keras.Model(inputLayer, outputs=[evalHead, policyHead])
    _compileRezNetModel(model)
    print("Created Rez-Net model")

    return model


def createMultipleGPUModel(model):
    gpuModel = keras.utils.multi_gpu_model(model, gpus=MachineSpecificSettings.AMOUNT_OF_GPUS)
    _compileRezNetModel(gpuModel)
    print("Created Multiple GPU Rez-Net model, using {} GPU's".format(MachineSpecificSettings.AMOUNT_OF_GPUS))
    return gpuModel


def _compileRezNetModel(model):
    # optimizer = keras.optimizers.Adam()
    optimizer = keras.optimizers.SGD(lr=Hyperparameters.LEARNING_RATE, momentum=Hyperparameters.MOMENTUM)
    model.compile(optimizer, ['mse', 'categorical_crossentropy'])
