import subprocess
import time
import pygetwindow as gw
from PIL import Image
from torchvision import transforms
import pyautogui

def start():
    # Путь к файлу игры и DOSBox
    dosbox_path = "C:\\Program Files (x86)\\DOSBox-0.74-3\\DOSBox.exe"
    game_path = "C:\\dendi_game\\lode\\LR.COM"

    # Запуск DOSBox с игрой
    subprocess.Popen([dosbox_path, game_path])

    # Подождать, пока игра загрузится
    time.sleep(3)

    # Получаем окно игры (например, по имени окна)
    window = gw.getWindowsWithTitle('DOSBox')[0]

    # Перемещаем окно в координаты (x, y)
    window.moveTo(0, 0)

    # Пример: отправляем стрелку вправо, чтобы двигать героя
    pyautogui.press('right')

    # Захватить экран (проверка состояния)
    screenshot = pyautogui.screenshot()
    screenshot.save('screenshot.png')
    time.sleep(2)


    # Указание области захвата (left, top, width, height)
    screenshot = pyautogui.screenshot(region=(0, 0, 645, 410))
    screenshot.save('game_screenshot.png')

start()




# Функция для захвата изображения
def capture_game_screen(region=(0, 0, 645, 410)):
    # region - координаты области захвата (x, y, ширина, высота)
    screenshot = pyautogui.screenshot(region=region)
    return screenshot

# Функция для преобразования изображения в тензор с двумя каналами
def image_to_tensor(image):
    # Преобразуем изображение в RGB
    image_rgb = image.convert("RGB")

    # Получаем отдельные каналы
    r, g, b = image_rgb.split()

    # Оставляем только красный и зеленый каналы
    image_two_channels = Image.merge("RGB", (r, g, g))  # Используем два канала: R и G

    # Преобразуем изображение в тензор
    transform = transforms.Compose([
        transforms.ToTensor(),  # Преобразуем изображение в тензор
    ])

    image_tensor = transform(image_two_channels)

    return image_tensor[1::]

image = capture_game_screen()
image_tensor = image_to_tensor(image)

# Проверим результат
print(image_tensor.shape)



# Параметры для игры
actions = ['no_action', 'left', 'right', 'dig',"up","down"]
n_actions = len(actions)



