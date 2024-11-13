import torch
import numpy as np
from snake import SnakeGame, Direction, Point
from model import PPOModel, PPOTrainer
from helper import plot, plot_loss

MAX_EPISODES = 1000
LR = 0.0005

class Agent:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gamma = 0.99
        self.model = PPOModel(11, 256, 3).to(self.device)
        self.trainer = PPOTrainer(self.model, lr=LR, gamma=self.gamma)
        # Trajectory storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []


    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)


    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        logits, value = self.model(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    def remember(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def compute_returns_and_advantages(self, next_value):
        returns = []
        advantages = []
        gae = 0
        values = self.values + [next_value]
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + self.gamma * values[step + 1] * (1 - self.dones[step]) - values[step]
            gae = delta + self.gamma * gae * (1 - self.dones[step])
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
        return returns, advantages

    def train(self, next_value):
        returns, advantages = self.compute_returns_and_advantages(next_value)
        # Convert to tensors
        states = torch.tensor(self.states, dtype=torch.float).to(self.device)
        actions = torch.tensor(self.actions).to(self.device)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float).to(self.device)
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # Update policy
        self.trainer.train_step(states, actions, old_log_probs, returns, advantages)
        # Clear trajectory
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

def train(n_games=MAX_EPISODES):
    agent = Agent()
    plot_scores = []
    plot_mean_scores = []
    last_20_mean_scores = []
    last_20_scores = 0
    total_score = 0
    record = 0

    for i in range(n_games):
        game = SnakeGame()
        state = agent.get_state(game)
        done = False

        while not done:
            action, log_prob, value = agent.get_action(state)
            final_move = [0, 0, 0]
            final_move[action] = 1
            reward, done, score = game.play_step(final_move)
            next_state = agent.get_state(game)
            agent.remember(state, action, reward, done, log_prob, value)
            state = next_state

        # Get value of the final state
        next_state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(agent.device)
        _, next_value = agent.model(next_state_tensor)
        next_value = next_value.item()
        agent.train(next_value)

        # Logging
        if score > record:
            record = score
            agent.model.save()

        plot_scores.append(score)
        mean_score = np.mean(plot_scores[-100:])
        plot_mean_scores.append(mean_score)
        last_20_scores += score

        if (i + 1) % 20 == 0:
            last_20_mean_scores.append(last_20_scores / 20)
            plot(plot_scores, plot_mean_scores, last_20_mean_scores)
            plot_loss(agent.trainer.losses)
            print(f'Game {i+1}, Score: {score}, Record: {record}')
            last_20_scores = 0

