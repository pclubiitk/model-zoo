import matplotlib.pyplot as plt


def plot_epoch(train_costs, val_costs, train_accuracy, val_accuracy):

    epochs = range(1, len(train_costs) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    fig.set_figheight(5)
    fig.set_figwidth(15)

    ax1.plot(epochs, train_costs, label='Training Loss')
    ax1.plot(epochs, val_costs, label='Validation Loss')
    ax1.set(xlabel='Epochs', ylabel='Loss')
    ax1.legend()

    ax2.plot(epochs, train_accuracy, label='Training Accuracy')
    ax2.plot(epochs, val_accuracy, label='Validation Accuracy')
    ax2.set(xlabel='Epochs', ylabel='Accuracy')
    ax2.legend()

    fig.savefig('epochs.png')
    plt.show()

    return
