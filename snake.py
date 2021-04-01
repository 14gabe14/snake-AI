import pygame
import random
from enum import Enum
from collections import namedtuple

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

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h

        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption ('Snake Game')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        #game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, Point(self.head.x-BLOCKSIZE, self.head.y), Point(self.head.x-(2*BLOCKSIZE), self.head.y)]

        self.score = 0
        self.food = None
        self.place_food()
        

    def place_food(self):
        x = random.randint(0, (self.w-BLOCKSIZE)//BLOCKSIZE)*BLOCKSIZE
        y = random.randint(0, (self.h-BLOCKSIZE)//BLOCKSIZE)*BLOCKSIZE

        self.food = Point(x, y)
        if self.food in self.snake:
            self.place_food()


    def play_step(self):
        #MOVEMENT FROM AI-------------


        #Check if game over
        game_over = False
        if self.is_collision():
            game_over = True
            return game_over, self.score
            
        #Place new food
        if self.head == self.food:
            self.score += 1
            self.place_food()

        
        #Update ui and clock
        self.update_ui()
        self.clock.tick(SPEED)

        # 6. return game over and score
        return game_over, self.score

    #Collision Detection
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head

        #if snake hits a boundary
        if pt.x > self.w - BLOCKSIZE or pt.x < 0 or pt.y > self.h - BLOCKSIZE or pt.y < 0:
            return True

        #if snake hits itself
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


if __name__ == '__main__':
    game = SnakeGame()

    while True:
        game_over, score = game.play_step()

        if game_over == True:
            break

    print('Final Score', score)

    pygame.quit()
