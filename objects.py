import pygame
import random
import numpy as np
from settings import *

class Player:
    def __init__(self, img):
        self.img = img
        self.x = WIDTH / 4
        self.y = HEIGHT / 2
        self.vel = 0
        self.can_jump = True
        self.alive = True

    def update(self, screen, tubes, score_value=[0], ai=None, tube_controller=None):
        # hitbox
        if SHOW_HITBOX:
            pygame.draw.rect(screen, (0, 0, 0), pygame.Rect(self.x, self.y, PLAYER_WIDTH, PLAYER_HEIGHT))

        # update position and velocity
        self.y += self.vel

        if ai is None:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE]:
                if self.can_jump:
                    self.can_jump = False
                    self.vel = -JUMP_VEL
            else:
                self.can_jump = True
        else:
            top_y, bottom_y, tube_x = tube_controller.get_inputs(self.x)

            prediction = ai.predict(np.array([[self.y, self.vel, top_y, bottom_y, tube_x]]), verbose=False)[0][0]
            if prediction > 0.9:
                if self.can_jump:
                    self.can_jump = False
                    self.vel = -JUMP_VEL
            else:
                self.can_jump = True
        
        # gravity
        self.vel += GRAVITY
        if self.vel > MAX_FALL_SPEED:
            self.vel = MAX_FALL_SPEED

        # check if hit ground
        lose = self.y + PLAYER_HEIGHT > HEIGHT or self.y < 0

        # check if hit tubes
        for tube in tubes:
            if self.x + PLAYER_WIDTH > tube.x and self.x < tube.x + TUBE_WIDTH:
                if self.y < tube.y + TUBE_HEIGHT and self.y + PLAYER_HEIGHT > tube.y:
                    lose = True
                    break

            elif tube.addPoint and self.x > tube.x + TUBE_WIDTH:
                tube.addPoint = False
                score_value[0] += 1
        # draw
        screen.blit(self.img, (self.x, self.y))
        if lose:
            self.alive = False
        return not lose


class Tube:
    def __init__(self, img, height, addPoint):
        self.img = img
        self.x = WIDTH
        self.y = height
        self.addPoint = addPoint

    def update(self, screen):
        self.x -= SPEED
        screen.blit(self.img, (self.x, self.y))


class TubeController:
    def __init__(self):
        self.tubes = []
        self.bottomImg = pygame.transform.scale(pygame.image.load(TUBE_BOTTOM_IMG), (TUBE_WIDTH, TUBE_HEIGHT))
        self.topImg = pygame.transform.scale(pygame.image.load(TUBE_TOP_IMG), (TUBE_WIDTH, TUBE_HEIGHT))
        self.remaining = DISTANCE_BETWEEN_TUBES
        self.create_pair()

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
        self.tubes.append(Tube(self.bottomImg, HEIGHT - height, False))
        # top
        self.tubes.append(Tube(self.bottomImg, HEIGHT - height - GAP_SIZE - TUBE_HEIGHT, True))
   
    # returns top_y, bottom_y, x_distance
    def get_inputs(self, player_x):
        for i in range(len(self.tubes)):
            bottom_tube = self.tubes[i]
            if player_x < bottom_tube.x + TUBE_WIDTH:
                top_tube = self.tubes[i + 1]
                # Return y distance to top tube first
                return top_tube.y, bottom_tube.y, bottom_tube.x - player_x
        print('uh oh :(((((')
        raise Exception('something is so wrong :(')
        