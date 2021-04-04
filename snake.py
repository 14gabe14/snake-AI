import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
from PIL import Image

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
LIGHT_BLUE = (0, 255, 255)
BLACK = (0, 0, 0)

BLOCKSIZE = 20
SPEED = 100


class SnakeGame:

    def __init__(self, w=480, h=480): #24 by 24 blocks
        self.w = w
        self.h = h

        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake Game')
        self.clock = pygame.time.Clock()

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
        self.over = False
        self.flag = 0
        self.dif = self.difference()
        

    def difference(self):
        dx = abs(self.head.x - self.food.x)/BLOCKSIZE/24
        dy = abs(self.head.y - self.food.y)/BLOCKSIZE/24

        return (dx+dy)/2

    def place_food(self):
        x = random.randint(0, (self.w-BLOCKSIZE)//BLOCKSIZE)*BLOCKSIZE
        y = random.randint(0, (self.h-BLOCKSIZE)//BLOCKSIZE)*BLOCKSIZE

        self.food = Point(x, y)
        if self.food in self.snake:
            self.place_food()


    def play_step(self, action: list):
        self.frame_iteration += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        #MOVEMENT FROM AI-------------
        #update snake's position according to action

        self._move(action)
        self.snake.insert(0, self.head)
        

        #Check if game over
        reward = 0
        
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            self.over = True
            reward = -10
            return reward, self.over, self.score
            

        #Place new food
        if self.head == self.food:
            self.score += 1
            reward = 10
            self.place_food()
        else:
            self.snake.pop()
            #reward = self.dif - self.difference()
            #self.dif = self.difference()
            if self.frame_iteration > 50*len(self.snake):
                reward -= 1
            
        #Update ui and clock
        self.update_ui()
        self.clock.tick(SPEED)
        
        # 6. return game over and score
        #return torch.tensor([game_over, self.score], device = DEVICE)
        return reward, self.over, self.score

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

    def toArray(self):
        if(self.over):
            print("GAME OVER")
            return np.zeros(shape=(1, 3, int(self.h/BLOCKSIZE), int(self.w/BLOCKSIZE)), dtype=np.uint8)

        array_image = np.zeros(shape=(int(self.h/BLOCKSIZE), int(self.w/BLOCKSIZE), 3), dtype=np.uint8)
        
        for pt in self.snake:
            array_image[int(pt.y/BLOCKSIZE), int(pt.x/BLOCKSIZE)] = BLUE

        array_image[int(self.head.y/BLOCKSIZE), int(self.head.x/BLOCKSIZE)] = LIGHT_BLUE

        array_image[int(self.food.y/BLOCKSIZE), int(self.food.x/BLOCKSIZE)] = RED

        array_image = np.moveaxis(array_image, 0, 2)

        array_image = np.reshape(array_image, (1, 3, int(self.h/BLOCKSIZE), int(self.h/BLOCKSIZE)))
        
        #print(array_image.shape)

        return array_image


if __name__ == '__main__':
    game = SnakeGame()
    clock = pygame.time.Clock()

    game.reset()

    while True:
        reward, game_over, score = game.play_step([1, 0, 0])
        reward, game_over, score = game.play_step([0, 1, 0])

        if game_over == True:
            break


    print('Final Score', score)
    


