import pygame
import random
from settings import *


class Player:
    def __init__(self, img):
        self.img = img
        self.x = WIDTH / 4
        self.y = HEIGHT / 2
        self.vel = 0
        self.can_jump = True

    def update(self, screen, tubes):

        # update position and velocity
        self.y += self.vel
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE] and self.can_jump:
            self.can_jump = False
            self.vel = -JUMP_VEL
        else:
            self.can_jump = True
        self.vel += GRAVITY

        # check collision
        if self.y + PLAYER_HEIGHT > HEIGHT:
            self.vel = 0
            self.y = HEIGHT - PLAYER_HEIGHT
        elif self.y < 0:
            self.vel = 0
            self.y = 0

        # check with tubes
        for tube in tubes:
            if self.x + PLAYER_WIDTH > tube.x and self.x < tube.x + TUBE_WIDTH:
                if self.y < tube.y + TUBE_HEIGHT and self.y + PLAYER_HEIGHT > tube.y:
                    # collision - break or something
                    
                    print(random.randint(1, 50))
                    break
                nexty = self.y + self.vel
                if nexty < tube.y + TUBE_HEIGHT and nexty + PLAYER_HEIGHT > tube.y:
                    if self.vel > 0:
                        self.vel = tube.y - (self.y + PLAYER_HEIGHT)
                    elif self.vel < 0:
                        self.vel =  (tube.y + TUBE_HEIGHT) - self.y

        # draw
        screen.blit(self.img, (self.x, self.y))


class Tube:
    def __init__(self, img, height):
        self.img = img
        self.x = WIDTH
        self.y = height

    def update(self, screen):
        self.x -= SPEED
        screen.blit(self.img, (self.x, self.y))


class TubeController:
    def __init__(self):
        self.tubes = []
        self.bottomImg = pygame.transform.scale(pygame.image.load(TUBE_BOTTOM_IMG), (TUBE_WIDTH, TUBE_HEIGHT))
        self.topImg = pygame.transform.scale(pygame.image.load(TUBE_TOP_IMG), (TUBE_WIDTH, TUBE_HEIGHT))
        self.remaining = DISTANCE_BETWEEN_TUBES

    def update(self, screen):
        i = 0
        while i < len(self.tubes):
            if self.tubes[i].x < -TUBE_WIDTH:
                del self.tubes[i]
                i -= 1
            self.tubes[i].update(screen)
            i += 1
            

        self.remaining -= SPEED
        if self.remaining < 0:
            self.create_pair()

    def create_pair(self):
        self.remaining = DISTANCE_BETWEEN_TUBES
        height = random.randint(min(GAP_SIZE, HEIGHT - GAP_SIZE), max(GAP_SIZE, HEIGHT - GAP_SIZE))
        # bottom
        self.tubes.append(Tube(self.bottomImg, HEIGHT - height))
        # top
        self.tubes.append(Tube(self.bottomImg, HEIGHT - height - GAP_SIZE - TUBE_HEIGHT))