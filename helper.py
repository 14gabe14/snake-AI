
import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores, last_20):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, color="darkgray")
    plt.plot(mean_scores)
    plt.plot(range(0, len(last_20)*20, 20), last_20)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(round(mean_scores[-1], 4)))
    plt.text(len(last_20)*20-20, last_20[-1], str(round(last_20[-1], 4)))
    plt.show()

def plot_loss(losses):
    plt.plot(losses)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.show()