import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
from PIL import Image

import torch
from torch import nn
from torchvision import transforms as T


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

    def __init__(self, w=240, h=240): #24 by 24 blocks
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
        self.action = [0, 0, 0]
        self.frameStack = np.zeros(shape = (4, int(self.h/BLOCKSIZE)+2, int(self.w/BLOCKSIZE)+2), dtype=np.uint8)
        return self.toTensor(self.frameStack)
       

    def difference(self):
        dx = abs(self.head.x - self.food.x)/BLOCKSIZE/16
        dy = abs(self.head.y - self.food.y)/BLOCKSIZE/16

        return (dx+dy)/2

    def place_food(self):
        x = random.randint(0, (self.w-BLOCKSIZE)//BLOCKSIZE)*BLOCKSIZE
        y = random.randint(0, (self.h-BLOCKSIZE)//BLOCKSIZE)*BLOCKSIZE

        self.food = Point(x, y)
        if self.food in self.snake:
            self.place_food()


    def play_step(self, action:list):

        self.frame_iteration += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        #MOVEMENT FROM AI-------------
        #update snake's position according to action

        self._move(action)
        self.snake.insert(0, self.head)
        
        self.reward = 0
        
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            self.over = True
            self.reward = -10
            return self.reward, self.over, self.score
        
        #if self.difference() < 2:
            #self.reward = 1

        #Place new food
        if self.head == self.food:
            self.score += 1
            self.reward = 10
            self.place_food()
            self.frame_iteration = 0
        else:
            self.snake.pop()
            #reward = self.dif - self.difference()
            #self.dif = self.difference()
            #if self.frame_iteration > 50*len(self.snake):
                #self.reward -= 1

        #if np.array_equal(self.action, action) and not np.array_equal(action, [1, 0, 0]):
            #self.flag += 1

        #if self.flag > 4:
            #self.reward -= 5

        self.action = action
            
        #Update ui and clock
        self.update_ui()
        self.clock.tick(SPEED)
        self.updateStack()
        # 6. return game over and score
        #return torch.tensor([game_over, self.score], device = DEVICE)
        return self.reward, self.over, self.score

    def play(self, action_idx):

        action = [0]*3
        action[action_idx] = 1

        self.frame_iteration += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self._move(action)
        self.snake.insert(0, self.head)
        
        self.reward = 0
        
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            self.over = True
            self.reward = -10
            return self.toTensor(self.frameStack), self.reward, self.over, self.score
        
        #Place new food
        if self.head == self.food:
            self.score += 1
            self.reward = 10
            self.place_food()
            self.frame_iteration = 0
        else:
            self.snake.pop()


        self.action = action
            
        self.update_ui()
        self.clock.tick(SPEED)
        self.updateStack()

        return self.toTensor(self.frameStack), self.reward, self.over, self.score


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

    def updateStack(self):
        
        fy = int(self.food.y/BLOCKSIZE)
        fx = int(self.food.x/BLOCKSIZE)

        bh = int(self.h/BLOCKSIZE)
        bw = int(self.w/BLOCKSIZE)

        if(self.over):
            print("GAME OVER")
        else:
            #print("Hello")
            array_image = self.getFrame()

            for pt in self.snake:
                array_image[0, int(pt.y/BLOCKSIZE)+1, int(pt.x/BLOCKSIZE)+1] = 128
            
            array_image[0, int(fy)+1, int(fx)+1] = 255
            
            self.frameStack = np.append(self.frameStack, array_image, axis=0)

            np.set_printoptions(threshold=np.inf)
            
            self.frameStack = np.delete(self.frameStack, 0, 0)

    def getFrame(self):
        bh = int(self.h/BLOCKSIZE)
        bw = int(self.w/BLOCKSIZE)

        array_image = np.zeros(shape=(1, bh+2, bw+2), dtype=np.uint8)

        array_image[0, 0, :] = 64
        array_image[0, -1, :] = 64
        array_image[0, :, 0] = 64
        array_image[0, :, -1] = 64

        return array_image

    def toTensor(self, observation):

        observation = torch.tensor(observation.copy(), dtype=torch.float)
        transforms = T.Normalize(0, 255)
        observation = transforms(observation).squeeze(0)
        
        return observation



if __name__ == '__main__':
    game = SnakeGame()
    clock = pygame.time.Clock()

    game.reset()

    

    #_, reward, game_over, score = game.play_step(2)
    _, reward, game_over, score = game.play(0)
    #_, reward, game_over, score = game.play_step(2)
    _, reward, game_over, score = game.play(0)
    #_, reward, game_over, score = game.play_step(2)
    _, reward, game_over, score = game.play(0)
    #_, reward, game_over, score = game.play_step(2)
    #_, reward, game_over, score = game.play_step(0)
    fs, reward, game_over, score = game.play(0)
    np.set_printoptions(threshold=np.inf)
    torch.set_printoptions(threshold=10_000)
    print(fs)

    fs, reward, game_over, score = game.play(0)
    print(fs)

    fs, reward, game_over, score = game.play_step(0)
    print(fs)


    #print('Final Score', score)
    


