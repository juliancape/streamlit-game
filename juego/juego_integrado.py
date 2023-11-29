import pygame
import pygame.freetype
import sys
import random
import os

# Inicialización de Pygame
pygame.init()

# Configuración de la pantalla
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Titanic Survivor")

def obt_carpeta():
    # Obtiene la ruta completa del script actual
    ruta_script = os.path.realpath(__file__)
    # Obtiene la carpeta del script
    carpeta_script = os.path.dirname(ruta_script)
    return carpeta_script + "/img_juego"

# Carga la imagen de fondo
#background_image = pygame.image.load("/home/daniel11/Documentos/DisenioCreativo/entregafinal/juego/oceano3.jpg").convert()
background_image = pygame.image.load(obt_carpeta() + "/oceano.jpg").convert()
background_image = pygame.transform.scale(background_image, (screen_width, screen_height))
# Cargar imagen de Game Over
#game_over_image = pygame.image.load("/home/daniel11/Documentos/DisenioCreativo/entregafinal/juego/Game_Over.png").convert_alpha()
game_over_image = pygame.image.load(obt_carpeta() + "/Game_Over.png").convert_alpha()
game_over_image = pygame.transform.scale(game_over_image, (screen_width-200, screen_height-200))

# Variables
# Variable para controlar el estado del juego (iniciar o no)
game_started = False
game_over = False
#start_time = pygame.time.get_ticks() # Tiempo de inicio del juego
start_time = 0
elapsed_time = 0
FPS = 100
x = 0
y = 0
screen.blit(background_image, (x, y))
clock = pygame.time.Clock() # Reloj para controlar la velocidad del juego
puntaje = 0

# Clase para la nave del jugador
class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        #original_image = pygame.image.load("/home/daniel11/Documentos/DisenioCreativo/entregafinal/juego/barco3.png").convert_alpha()
        original_image = pygame.image.load(obt_carpeta() + "/barco.png").convert_alpha()
        new_width, new_height = 65, 65
        self.image = pygame.transform.scale(original_image, (new_width, new_height))
        self.rect = self.image.get_rect()
        self.rect.centerx = screen_width // 2
        self.rect.bottom = screen_height - 10
        self.speed = 5

    def update(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] and self.rect.left > 0:
            self.rect.x -= self.speed
        if keys[pygame.K_RIGHT] and self.rect.right < screen_width:
            self.rect.x += self.speed


# Clase para los enemigos
class Enemy(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        #original_image = pygame.image.load("/home/daniel11/Documentos/DisenioCreativo/entregafinal/juego/iceberg3.png").convert_alpha()
        original_image = pygame.image.load(obt_carpeta() + "/iceberg.png").convert_alpha()
        tam = random.randint(20, 50)
        new_width, new_height = tam, tam
        self.image = pygame.transform.scale(original_image, (new_width, new_height))
        self.rect = self.image.get_rect()
        self.rect.x = random.randrange(screen_width - self.rect.width)
        self.rect.y = random.randrange(-50, -10)
        self.speed = random.randint(2, 4)

    def update(self):
        self.rect.y += self.speed
        if self.rect.y > screen_height:
            self.rect.y = random.randrange(-50, -10)
            self.rect.x = random.randrange(screen_width - self.rect.width)
   
   
# Funcion para el movimiento del fondo         
def movimiento_background(y):
    # Movimiento del fondo
    y_relativa = y % background_image.get_rect().height
    screen.blit(background_image, (x, y_relativa - background_image.get_rect().height))
    if y_relativa < screen_height:
        screen.blit(background_image, (0, y_relativa))
    y += 3
    return y


# Funcion para definir los eventos 
def eventos():
    global game_started, start_time, puntaje
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            print(puntaje)
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                # Iniciar el juego al presionar Enter
                game_started = True
                start_time = pygame.time.get_ticks() # Tiempo de inicio del juego


# Funcion para establecer las colisiones
def colisiones(player, enemies):
    global elapsed_time, game_over
    hits = pygame.sprite.spritecollide(player, enemies, False)
    if hits:
        end_time = pygame.time.get_ticks()
        elapsed_time = (end_time - start_time) // 1000  # Convierte a segundos
        game_over = True
    #return False

def calcular_puntaje(tiempo):
    # El valor por segundo va a ser de 2 puntos, para que el que llegue a 60 seg gane 120 puntos
    valorXseg = 2
    return tiempo * valorXseg

def mostrar_tiempo_playing():
    font = pygame.font.Font(None, 36)
    # Dibujar el tiempo en la esquina superior derecha
    current_time = pygame.time.get_ticks()
    current_elapsed_time = (current_time - start_time) // 1000  # Convierte a segundos
    text = font.render(f"Tiempo: {current_elapsed_time} segundos", True, (255, 255, 255))
    text_rect = text.get_rect(topright=(screen_width - 10, 10))
    screen.blit(text, text_rect)

def aumentar_dificultad():
    global FPS
    #, start_time
    current_time = pygame.time.get_ticks()
    tiempo = (current_time - start_time) // 1000
    if tiempo == 10:
        FPS = 130
    if tiempo == 20:
        FPS = 150
    if tiempo == 30:
        FPS = 170    

# Funcion para establecer la pantalla de Game Over
def pantalla_game_over():
    # Obtener las dimensiones de la imagen de Game Over
    game_over_rect = game_over_image.get_rect()
    # Calcular las coordenadas para centrar la imagen de Game Over
    x = (screen_width - game_over_rect.width) // 2
    y = (screen_height - game_over_rect.height) // 2
    # Mostrar la imagen de Game Over en el centro de la pantalla
    screen.blit(game_over_image, (x, y))
    
    # Mostrar el tiempo de juego
    font = pygame.font.Font(None, 36)
    # Mostrar el puntaje
    global puntaje
    puntaje = calcular_puntaje(elapsed_time)
    textPuntaje = font.render(f"{puntaje} puntos", True, (255, 255, 255))
    #text_rect_puntaje = textPuntaje.get_rect(center=(screen_width // 2, y + game_over_rect.height + 45))
    text_rect_puntaje = textPuntaje.get_rect(topright=(screen_width - 10, 40))
    screen.blit(textPuntaje, text_rect_puntaje)

def pantalla_start():
    # Mostrar pantalla de espera
    font = pygame.font.Font(None, 106)
    screen.fill((0, 0, 0))
    text = font.render("TITANIC SURVIVOR", True, (255, 255, 255))
    text_rect = text.get_rect(center=(screen_width // 2, screen_height // 2))
    screen.blit(text, text_rect)
    
    font = pygame.font.Font(None, 16)
    text = font.render("Presiona Enter para comenzar", True, (255, 255, 255))
    text_rect = text.get_rect(center=( screen_width // 2, (screen_height+100) // 2))
    screen.blit(text, text_rect)
    
    pygame.display.flip()

#def actualizar_puntaje():
#    global puntaje    
#    nombre_archivo = os.path.abspath(os.getcwd()) + "/puntaje.txt"
#    with open(nombre_archivo, 'r') as archivo:
#        contenido = archivo.read()
#    numero_actual = int(contenido)
#    nuevo_numero = numero_actual + int(puntaje)
#    with open(nombre_archivo, 'w') as archivo:
#        archivo.write(str(nuevo_numero))

def main():
    global screen_width, screen_height, screen, background_image, background_image, game_over_image, game_started, game_over, start_time, elapsed_time, FPS, x, y, clock, puntaje 
    
    # Grupos de sprites
    all_sprites = pygame.sprite.Group()
    enemies = pygame.sprite.Group()


    # Crear jugador y enemigos
    player = Player()
    all_sprites.add(player)
    for _ in range(10):
        enemy = Enemy()
        all_sprites.add(enemy)
        enemies.add(enemy)


    # Bucle principal del juego
    while True:
        # Manejo de eventos
        eventos()
        
        if game_started:
            if not game_over:
                aumentar_dificultad()
                # Movimiento del fondo
                y = movimiento_background(y)
                # Control de FPS
                clock.tick(FPS)
                # Actualizar
                all_sprites.update()
                # Colisiones
                colisiones(player, enemies)
                # Mostrar tiempo
                mostrar_tiempo_playing()
                # Dibujar
                all_sprites.draw(screen)
            else:
                pantalla_game_over()
                pygame.display.flip()

                # Manejo de eventos en la pantalla de Game Over
                for event in pygame.event.get():
                    #if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                    if event.type == pygame.QUIT:
                        #pygame.quit()
                        #sys.exit()
                        print(puntaje)
                        pygame.quit()
                        sys.exit()
                        #return puntaje
                        #actualizar_puntaje()
                        #print(puntaje)
                        
        else:
            pantalla_start()
        # Actualizar la pantalla
        pygame.display.flip()
        

if __name__ == "__main__":
    main()