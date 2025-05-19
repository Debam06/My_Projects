import pygame
import random
import math
import sys

# Initialize Pygame
pygame.init()

# Window dimensions and setup
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Missile Defender - Enhanced")

# Define colors
BLACK   = (  0,   0,   0)
WHITE   = (255, 255, 255)
GREEN   = (  0, 255,   0)
RED     = (255,   0,   0)
BLUE    = (  0,   0, 255)
GRAY    = (150, 150, 150)
ORANGE  = (255, 165,   0)

# Set game clock and FPS
clock = pygame.time.Clock()
FPS = 60

# Fonts for text
font_small = pygame.font.SysFont("arial", 24)
font_large = pygame.font.SysFont("arial", 48)

# ------------------------- Classes ------------------------- #

class MissileLauncher:
    def __init__(self):
        self.width = 80
        self.height = 20
        self.x = WIDTH // 2 - self.width // 2
        self.y = HEIGHT - self.height - 10
        self.color = GREEN
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)

    def update(self, mouse_x):
        self.x = mouse_x - self.width // 2
        if self.x < 0:
            self.x = 0
        if self.x + self.width > WIDTH:
            self.x = WIDTH - self.width
        self.rect.x = self.x

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)

class Missile:
    def __init__(self, x, y):
        self.width = 5
        self.height = 10
        self.x = x
        self.y = y
        self.speed = 10
        self.color = RED
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)

    def update(self):
        self.y -= self.speed
        self.rect.y = self.y

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)

class Drone:
    def __init__(self):
        self.radius = 20
        # Start at a random horizontal position just above the window.
        self.x = random.randint(self.radius, WIDTH - self.radius)
        self.y = -self.radius
        self.speed_y = random.uniform(1.0, 3.0)
        # Movement pattern: straight, zigzag, or curve.
        self.pattern = random.choice(["straight", "zigzag", "curve"])
        if self.pattern == "straight":
            self.speed_x = 0
        elif self.pattern == "zigzag":
            self.speed_x = random.choice([-3, 3])
            self.change_time = random.randint(20, 40)
            self.counter = 0
        elif self.pattern == "curve":
            self.amplitude = random.randint(20, 50)
            self.frequency = random.uniform(0.05, 0.1)
            self.origin_x = self.x
        # For enhanced variability, drones are either "normal" or "homer".
        # 85% chance to be normal, 15% chance to be homing.
        r = random.random()
        if r < 0.85:
            self.drone_type = "normal"
        else:
            self.drone_type = "homer"
        # All drones have only one hit of health.
        self.health = 1
        # Determine drone shape: circle or square.
        self.shape = random.choice(["circle", "square"])
        if self.shape == "square":
            self.color = ORANGE
        else:
            self.color = BLUE

    def update(self, difficulty, launcher):
        # Increase vertical speed based on difficulty.
        self.y += self.speed_y * difficulty

        # Update horizontal movement based on the chosen pattern.
        if self.pattern == "zigzag":
            self.x += self.speed_x
            self.counter += 1
            if self.counter >= self.change_time:
                self.speed_x = -self.speed_x
                self.counter = 0
        elif self.pattern == "curve":
            self.x = self.origin_x + self.amplitude * math.sin(self.frequency * self.y)
        
        # Homing behavior: adjust x coordinate slightly toward the missile launcher.
        if self.drone_type == "homer":
            center_launcher = launcher.x + launcher.width / 2
            if self.x < center_launcher:
                self.x += 1
            elif self.x > center_launcher:
                self.x -= 1

        # Keep within horizontal boundaries.
        if self.x < self.radius:
            self.x = self.radius
        if self.x > WIDTH - self.radius:
            self.x = WIDTH - self.radius

    def draw(self, surface):
        if self.shape == "circle":
            pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), self.radius)
        else:
            rect = pygame.Rect(self.x - self.radius, self.y - self.radius, self.radius * 2, self.radius * 2)
            pygame.draw.rect(surface, self.color, rect)

    def get_rect(self):
        # Return a bounding rectangle for collision detection.
        return pygame.Rect(self.x - self.radius, self.y - self.radius,
                           self.radius * 2, self.radius * 2)

class Explosion:
    def __init__(self, x, y, max_radius=100, duration=20):
        self.x = x
        self.y = y
        self.max_radius = max_radius
        self.duration = duration
        self.current_frame = 0

    def update(self):
        self.current_frame += 1

    def draw(self, surface):
        # Animate an expanding circle for the explosion.
        r = int(self.max_radius * (self.current_frame / self.duration))
        if r < 1:
            r = 1
        pygame.draw.circle(surface, RED, (self.x, self.y), r, 2)

    def is_finished(self):
        return self.current_frame >= self.duration

# ------------------------- Helper Functions ------------------------- #

def draw_reset_button(surface):
    button_rect = pygame.Rect(10, 10, 100, 40)
    pygame.draw.rect(surface, GRAY, button_rect)
    pygame.draw.rect(surface, WHITE, button_rect, 2)
    text = font_small.render("Reset", True, WHITE)
    text_rect = text.get_rect(center=button_rect.center)
    surface.blit(text, text_rect)
    return button_rect

def reset_game():
    global missiles, drones, explosions, score, game_over, drone_spawn_timer, big_missile_available, big_missile_threshold
    missiles = []
    drones = []
    explosions = []
    score = 0
    game_over = False
    drone_spawn_timer = 0
    big_missile_available = False
    big_missile_threshold = 10

# ------------------------- Global Variables ------------------------- #

reset_game()
high_score = 0
big_missile_available = False  # Big missile becomes available after reaching the threshold.
big_missile_threshold = 10     # Next availability threshold after reaching this score.

# Create the missile launcher instance.
launcher = MissileLauncher()

# ------------------------- Main Game Loop ------------------------- #

running = True
while running:
    clock.tick(FPS)
    screen.fill(BLACK)

    # Event loop
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            reset_rect = pygame.Rect(10, 10, 100, 40)
            if reset_rect.collidepoint(mouse_x, mouse_y):
                reset_game()
            elif not game_over:
                # Left click: fire a normal missile.
                if event.button == 1:
                    missile_x = launcher.x + launcher.width // 2 - 2
                    missile_y = launcher.y
                    missiles.append(Missile(missile_x, missile_y))
                # Right click: fire the BIG missile if available.
                elif event.button == 3:
                    if big_missile_available:
                        explosion_radius = int(100 * (1 + score / 100.0))
                        explosion = Explosion(mouse_x, mouse_y, max_radius=explosion_radius, duration=20)
                        explosions.append(explosion)
                        # Destroy every drone within the explosion radius.
                        drones_hit = []
                        for drone in drones:
                            dist = math.hypot(drone.x - mouse_x, drone.y - mouse_y)
                            if dist <= explosion.max_radius:
                                drones_hit.append(drone)
                        for drone in drones_hit:
                            drones.remove(drone)
                            score += 1
                        big_missile_available = False
                        big_missile_threshold = score + 10

    # Update game objects
    mouse_x, _ = pygame.mouse.get_pos()
    launcher.update(mouse_x)

    # Difficulty scaling: increase factor based on score.
    difficulty = 1 + score / 100.0
    # Adjust drone spawn threshold as score increases.
    spawn_threshold = max(10, 40 - score // 2)

    if not game_over:
        # Update normal missiles.
        for missile in missiles[:]:
            missile.update()
            if missile.y < -missile.height:
                missiles.remove(missile)

        # Spawn new drone based on the dynamic spawn threshold.
        drone_spawn_timer += 1
        if drone_spawn_timer >= spawn_threshold:
            drones.append(Drone())
            drone_spawn_timer = 0

        # Update drones.
        for drone in drones[:]:
            drone.update(difficulty, launcher)
            # If a drone collides with the launcher, game over.
            if drone.get_rect().colliderect(launcher.rect):
                game_over = True
                if score > high_score:
                    high_score = score
                break
            # Instead of ending the game, deduct 2 points when a drone reaches the bottom.
            if drone.y + drone.radius >= HEIGHT:
                score -= 2
                drones.remove(drone)

        # Check collisions between normal missiles and drones.
        for missile in missiles[:]:
            for drone in drones[:]:
                if missile.rect.colliderect(drone.get_rect()):
                    drones.remove(drone)
                    score += 1
                    try:
                        missiles.remove(missile)
                    except ValueError:
                        pass
                    break

    # Update explosion animations (for the big missile effect).
    for explosion in explosions[:]:
        explosion.update()
        if explosion.is_finished():
            explosions.remove(explosion)

    # Check for big missile availability.
    if score >= big_missile_threshold and not big_missile_available:
        big_missile_available = True

    # Draw everything.
    launcher.draw(screen)
    for missile in missiles:
        missile.draw(screen)
    for drone in drones:
        drone.draw(screen)
    for explosion in explosions:
        explosion.draw(screen)

    # Draw score and high score.
    score_text = font_small.render(f"Score: {score}", True, WHITE)
    screen.blit(score_text, (WIDTH - 130, 10))
    high_score_text = font_small.render(f"High: {high_score}", True, WHITE)
    screen.blit(high_score_text, (WIDTH - 130, 40))

    # Draw reset button.
    reset_button = draw_reset_button(screen)

    # Display prompt if the big missile is available.
    if big_missile_available and not game_over:
        big_text = font_small.render("BIG MISSILE READY: Right-click to launch!", True, RED)
        big_rect = big_text.get_rect(center=(WIDTH // 2, 60))
        screen.blit(big_text, big_rect)

    # If game over, display a message.
    if game_over:
        over_text = font_large.render("GAME OVER", True, RED)
        over_rect = over_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        screen.blit(over_text, over_rect)

    pygame.display.flip()

pygame.quit()
sys.exit()