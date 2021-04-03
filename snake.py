import pygame
import random
from enum import Enum
from collections import namedtuple

#from DQN import DQN, ReplayMemory, EpsilonGreedyStrategy

DEVICE = 'cpu'

pygame.init()

direction = {"left": 1, "right":2, "up":3, "down":4}


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
        pygame.display.set_caption ('Snake Game')

        self.reset()

    def reset(self):
        #game state
        self.direction = direction["right"]
        
        
        self.snake = [Point(self.w/2-(2*BLOCKSIZE), self.h/2), 
                      Point(self.w/2-BLOCKSIZE, self.h/2), 
                      Point(self.w/2, self.h/2)] #HEAD
        
        self.snakeLength = len(self.snake)
        self.score = 0
        self.food = None
        self.place_food()
        

    def place_food(self):
        x = random.randint(0, (self.w-BLOCKSIZE)//BLOCKSIZE)*BLOCKSIZE
        y = random.randint(0, (self.h-BLOCKSIZE)//BLOCKSIZE)*BLOCKSIZE

        self.food = Point(x, y)
        if self.food in self.snake:
            self.place_food()


    def play_step(self, action):
        #MOVEMENT FROM AI-------------
        #update snake's position according to action
        for point in self.snake:
            print(point.x, point.y)
            
        print(action)
        
        x1 = self.snake[-1].x
        y1 = self.snake[-1].y
        
        x1_change = 0       
        y1_change = 0
        
        #OPTIONAL IMPLEMENT NO BACKWARDS MOVEMENT
        if action == 1: #LEFT
            x1_change = -BLOCKSIZE
            y1_change = 0
            print("LEFT")
        elif action == 2: #RIGHT
            x1_change = BLOCKSIZE
            y1_change = 0
            print("RIGHT")
        elif action == 3: #UP
            y1_change = -BLOCKSIZE
            x1_change = 0
            print("UP")
        elif action == 4: #DOWN
            y1_change = BLOCKSIZE
            x1_change = 0
            print("DOWN")
            
        x1 += x1_change
        y1 += y1_change
        
        self.snake.append(Point(x1, y1))
        #delete tail
        if len(self.snake) > self.snakeLength:
            del self.snake[0]
        
        #Check if game over
        game_over = False
        if self.is_collision():
            print("collision")
            game_over = True
            #return torch.tensor([game_over, self.score], device = DEVICE)
            return game_over, self.score
            
        #Place new food
        if self.snake[-1] == self.food:
            self.score += 100
            self.place_food()
            self.snakeLenght += 1

        
        #Update ui and clock
        self.update_ui()

        # 6. return game over and score
        #return torch.tensor([game_over, self.score], device = DEVICE)
        return game_over, self.score

    #Collision Detection
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.snake[-1]

        #if snake hits a boundary
        if pt.x > self.w - BLOCKSIZE or pt.x < 0 or pt.y > self.h - BLOCKSIZE or pt.y < 0:
            print("boundary hit")
            return True
        
        for point in self.snake:
            print(point.x, point.y)
            
        #if snake hits itself
        if pt in self.snake[:-1]:
            
            print("hits itself")
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
    clock = pygame.time.Clock()

    while True:
        game_over, score = game.play_step(random.randint(1,4))

        clock.tick(SPEED)
        print("TICK")

        if game_over == True:
            break

    print('Final Score', score)

    pygame.quit()

