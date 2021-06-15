import matplotlib.pyplot as plt


def evaluate(model, test_ds):
    predict = model.evaluate(test_ds)
    print("The Test Accuracy of the model is: ", predict * 100)


def plot_traingraph(hist):

    epoch = range(70)
    train_acc = hist.history["accuracy"]
    train_loss = hist.history["loss"]
    validation_acc = hist.history["val_accuracy"]
    validation_loss = hist.history["val_loss"]
    # fig ,ax= plt.subplots(nrows=1,ncols=2)

    # This plot shows the variation of training accuracy and the validation accuracy with epochs
    fig = plt.figure(figsize=(7, 5))
    plt.plot(epoch, train_acc)
    plt.plot(epoch, validation_acc)
    plt.xlabel("EPOCHS")
    plt.ylabel("ACCURACY")
    plt.title("Variation of Accuracy")
    plt.legend(["train_acc", "validation_acc"])
    plt.grid()

    # This plot shows the variation of training loss and the validation loss with epochs
    fig = plt.figure(figsize=(7, 5))
    plt.plot(epoch, train_loss)
    plt.plot(epoch, validation_loss)
    plt.xlabel("EPOCHS")
    plt.ylabel("LOSS")
    plt.title("Variation of Loss")
    plt.legend(["train_loss", "validation_loss"])
    plt.grid()
    fig.tight_layout(pad=0.3)
