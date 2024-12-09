from PIL import Image


def crop_and_show(image_path, left, top, right, bottom):
    """
    Вырезает часть изображения и отображает её.

    :param image_path: Путь к файлу изображения.
    :param left: Координата левого края вырезаемой области.
    :param top: Координата верхнего края вырезаемой области.
    :param right: Координата правого края вырезаемой области.
    :param bottom: Координата нижнего края вырезаемой области.
    """
    # Открываем изображение
    image = Image.open(image_path)

    # Вырезаем часть
    cropped_image = image.crop((left, top, right, bottom))

    # Показываем вырезанную часть
    cropped_image.show()


# Пример использования
score= crop_and_show("game_screenshot.png", 310, 395, 420, 410)  #
