# Задачи классификации

# Логистическая регрессия / Бинарная классификация
# Функция потерь для бинарной классификации, принимает логиты (без активации сигмоиды)
loss_fn_bce = nn.BCEWithLogitsLoss()
# Пример: для задачи предсказания "да" или "нет".

# Кросс-энтропия
# Функция потерь для многоклассовой классификации, ожидает логиты
loss_fn_ce = nn.CrossEntropyLoss()
# Пример: для задачи классификации изображений (кот, собака, машина и т.д.).

# Мультизадачная классификация
# Функция потерь для многоклассовой классификации с несколькими правильными классами
loss_fn_bce_multi = nn.BCELoss()
# Пример: задача, где объект может принадлежать сразу нескольким категориям.

# Задачи регрессии

# Среднеквадратичная ошибка (MSE)
# Используется для регрессионных задач, минимизирует квадрат разницы между предсказанием и истинным значением
loss_fn_mse = nn.MSELoss()
# Пример: предсказание цены дома на основе его характеристик.

# Средняя абсолютная ошибка (MAE)
# Минимизирует абсолютное значение разницы между предсказанием и истинным значением
loss_fn_mae = nn.L1Loss()
# Пример: предсказание температуры.

# Задачи ранжирования

# Гингфовая функция потерь (Margin Ranking Loss)
# Используется для задач ранжирования, чтобы гарантировать, что один объект имеет больший ранг, чем другой
loss_fn_margin = nn.MarginRankingLoss()
# Пример: задача ранжирования поисковых результатов.

# Триплетная функция потерь (Triplet Margin Loss)
# Максимизирует расстояние между разными классами и минимизирует расстояние между одним классом
loss_fn_triplet = nn.TripletMarginLoss()
# Пример: задачи с эмбеддингами, например, распознавание лиц.

# Задачи сегментации

# Дайсовая функция потерь (Dice Loss)
# Часто используется для задач сегментации изображений, особенно в медицине
# Реализуется через комбинацию других функций потерь
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)  # Применяем сигмоиду к входным данным
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice

loss_fn_dice = DiceLoss()
# Пример: задачи сегментации, такие как выделение органов на медицинских изображениях.

# Задачи обучения с подкреплением

# Huber Loss
# Используется, чтобы комбинировать чувствительность MAE и устойчивость MSE
loss_fn_huber = nn.SmoothL1Loss()
# Пример: обучение агента в задаче управления роботом.

# Заключение
# Выбор функции потерь зависит от конкретной задачи и данных. Убедитесь, что входные данные и метки соответствуют требованиям функции потерь!