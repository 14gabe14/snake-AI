import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

#from DQN import DQN, ReplayMemory, EpsilonGreedyStrategy

DEVICE = 'cpu'

pygame.init()

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')

#Colours 
RED = (200, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

BLOCKSIZE = 20
SPEED = 20


class SnakeGame:

    def __init__(self, w=640, h=480): #32 by 24 blocks
        self.w = w
        self.h = h

        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake Game')

        self.reset()

    def reset(self):
        #game state
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        
        self.snake = [self.head,
                      Point(self.head.x-BLOCKSIZE, self.head.y),
                      Point(self.head.x-(2*BLOCKSIZE), self.head.y)]

        self.score = 0
        self.food = None
        self.place_food()
        self.frame_iteration = 0
        

    def place_food(self):
        x = random.randint(0, (self.w-BLOCKSIZE)//BLOCKSIZE)*BLOCKSIZE
        y = random.randint(0, (self.h-BLOCKSIZE)//BLOCKSIZE)*BLOCKSIZE

        self.food = Point(x, y)
        if self.food in self.snake:
            self.place_food()


    def play_step(self, action: list):
        self.frame_iteration += 1

        #MOVEMENT FROM AI-------------
        #update snake's position according to action

        self._move(action)
        self.snake.insert(0, self.head)
        

        #Check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
            

        #Place new food
        if self.head == self.food:
            self.score += 1
            reward = 10
            self.place_food()
        else:
            self.snake.pop()

        
        #Update ui and clock
        self.update_ui()

        # 6. return game over and score
        #return torch.tensor([game_over, self.score], device = DEVICE)
        return reward, game_over, self.score

    #Collision Detection
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCKSIZE or pt.x < 0 or pt.y > self.h - BLOCKSIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False


    #Interface
    def update_ui(self):
        #Background is black
        self.display.fill(BLACK)

        #Snake is made up of blue rectangles 
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE, pygame.Rect(pt.x, pt.y, BLOCKSIZE, BLOCKSIZE))
            

        #Food is made up of red rectangles
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCKSIZE, BLOCKSIZE))

        pygame.display.flip()


    def _move(self, action):

        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        print(3)
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCKSIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCKSIZE
        elif self.direction == Direction.DOWN:
            y += BLOCKSIZE
        elif self.direction == Direction.UP:
            y -= BLOCKSIZE
        
        self.head = Point(x, y)



if __name__ == '__main__':
    game = SnakeGame()
    clock = pygame.time.Clock()

    while True:
        reward, game_over, score = game.play_step([1, 0, 0])
        reward, game_over, score = game.play_step([0, 1, 0])

        clock.tick(SPEED)

        if game_over == True:
            break


    print('Final Score', score)

    pygame.quit()