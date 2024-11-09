# Snake AI

This project implements a reinforcement learning agent to play the classic Snake game using a Convolutional Neural Network (CNN).

## Project Structure
- `model.py`: Contains the implementation of the linear-based agent.
- `CNNSnake.py`: Contains the implementation of the CNN-based agent.
- `snake.py`: Contains implementation of the Snake game.
- `helper.py`: Contains helper functions for plotting training progress.
- `test_linear.ipynb`: Jupyter notebook for testing the linear model.
- `test_cnn.ipynb`: Jupyter notebook for testing the CNN model.

## Usage

### Training the Agent

To train the agent:
```python
from agent import train

train()
```

### Testing the Agent
You can test the trained agent using the provided Jupyter notebooks:

`test_linear.ipynb`: For testing the linear model.
`test_cnn.ipynb`: For testing the CNN model.
Logging and Checkpoints
Training logs and checkpoints are saved in the checkpoints/ directory. Each training session creates a new subdirectory with a timestamp.

### Visualization
The helper.py script contains functions to plot the training progress. The plots include:
- Scores over time
- Mean scores
- Last 20 scores