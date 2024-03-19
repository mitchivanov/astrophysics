import numpy as np
import matplotlib.pyplot as plt
import pygame

# Параметры моделирования
G = 10  # Гравитационная постоянная
n = 3  # Количество тел
dt = 1  # Шаг интегрирования

# Параметры графического окна
width = 1000  # Ширина окна
height = 1000 # Высота окна
fps = 30  # Частота кадров в секунду

# Параметры математической системы координат
xmin, xmax, ymin, ymax = -50, 50, -50, 50

# Параметры визуализациии цвета отображения и размер
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
sizes = [5, 5, 5]


# Функции преобразования координат
def XtoX(x):
    result = (x - xmin) / (xmax - xmin) * width
    if 0 <= result <= width:
        return round(result)
    else:
        raise ValueError("Некорректное значение XtoX")


def YtoY(y):
    result = height - (y - ymin) / (ymax - ymin) * height
    if 0 <= result <= height:
        return round(result)
    else:
        raise ValueError("Некорректное значение YtoY")


# Функция начальных условий и пораметров тел
def init_values():
    M = np.array([1, 1.67054 * 10 ** -8, 2.30634 * 10 ** -9])

    X = np.array([0, 5, 12.508])
    Y = np.array([0, 0, 0])

    VX = np.array([0, 0, 0])
    VY = np.array([0, 1.4, 1])

    return X, Y, VX, VY, M


# Функция расчёта ускорения
def acceleration(X, Y, M):
    AX = np.zeros(n)
    AY = np.zeros(n)
    for i in range(n):
        ax = 0
        ay = 0
        for j in range(n):
            if (i != j):
                dx = X[i] - X[j]
                dy = Y[i] - Y[j]
                r = np.sqrt(dx ** 2 + dy ** 2)
                ax = ax + M[j] * dx / r ** 3
                ay = ay + M[j] * dy / r ** 3
        AX[i] = -G * ax
        AY[i] = -G * ay
    return AX, AY


# Алгоритм Верле
def verlet(X, Y, VX, VY, AX, AY, M, dt):
    X = X + VX * dt + AX * dt ** 2 / 2
    Y = Y + VY * dt + AY * dt ** 2 / 2
    AX1, AY1 = acceleration(X, Y, M)
    VX = VX + (AX + AX1) / 2 * dt
    VY = VY + (AY + AY1) / 2 * dt
    AX, AY = AX1, AY1
    return X, Y, VX, VY, AX, AY


# Алгоритм Эйлера-Кромера
def euler(X, Y, VX, VY, M, dt):
    AX, AY = acceleration(X, Y, M)
    VX = VX + AX * dt
    VY = VY + AY * dt
    X = X + VX * dt
    Y = Y + VY * dt
    return X, Y, VX, VY


# Создаем и заполняем массивы координат, скоростей и масс; вычисляем ускороения
X, Y, VX, VY, M = init_values()
AX, AY = acceleration(X, Y, M)

# Инициализация Pygame
pygame.init()
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Gravity Simulation")
clock = pygame.time.Clock()

# Создаем списки для хранения траекторий
trajectory_X = [[] for _ in range(n)]
trajectory_Y = [[] for _ in range(n)]

# Запускаем основной цикл симуляции
running = True
while running:

    # Расчет и обновление координат тел
    X, Y, VX, VY = euler(X, Y, VX, VY, M, dt)
    # X, Y, VX, VY, AX, AY = verlet(X, Y, VX, VY, AX, AY, M, dt)

    # Добавляем текущие координаты в траектории
    for i in range(n):
        trajectory_X[i].append(X[i])
        trajectory_Y[i].append(Y[i])

    clock.tick(fps)  # Ограничение частоты кадров

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    screen.fill((255, 255, 255))  # Очистка экрана

    # Отображение тел
    for i in range(n):
        pygame.draw.circle(screen, colors[i], (int(XtoX(X[i])), int(YtoY(Y[i]))), sizes[i])

    # Отображение следа движения
    for i in range(n):
        for j in range(len(trajectory_X[i]) - 1):
            pygame.draw.line(screen, colors[i], (int(XtoX(trajectory_X[i][j])), int(YtoY(trajectory_Y[i][j]))),
                             (int(XtoX(trajectory_X[i][j + 1])), int(YtoY(trajectory_Y[i][j + 1]))))

    pygame.display.flip()  # Обновление экрана

pygame.quit()  # Завершение Pygame

# Отображение графика траекторий с помощью Matplotlib
for i in range(n):
    plt.plot(trajectory_X[i], trajectory_Y[i], label=f'Тело {i + 1}', )

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Траектории движения тел')
plt.legend()
plt.grid(True)
plt.show()
