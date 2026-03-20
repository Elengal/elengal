# Структура проекта

```
elengal/
├── elengal_v1.py           # Основная модель
├── train_elengal.py        # Скрипт обучения
│
├── examples/
│   └── simple_example.py   # Пример использования
│
├── tests/
│   └── test_elengal.py     # Тесты (если есть)
│
├── README.md               # Документация
├── CONTRIBUTING.md         # Правила контрибуции
├── CODE_OF_CONDUCT.md      # Кодекс поведения
├── COMMERCIAL_LICENSE.txt  # Коммерческая лицензия
├── LICENSE                 # AGPL v3
├── requirements.txt        # Зависимости
└── CITATION.cff            # Для цитирования
```

## Основные файлы

### elengal_v1.py

Содержит все классы модели:

| Класс | Описание |
|-------|----------|
| `ElengalDevice` | Автоопределение устройства (CUDA/MPS/CPU) |
| `ElengalConfig` | Конфигурация модели |
| `ElengalMath` | Математика: q-экспонента, тета-функции, Похгаммер |
| `ElengalTokenState` | Физическое состояние токена |
| `ElengalField` | Фундаментальное поле (гравитация, магнетизм, время) |
| `ElengalAttention` | q-внимание на основе q-экспоненты |
| `ElengalCellularFFN` | FFN с gate от генома и энергии |
| `ElengalEvolutionLayer` | Слой эволюции токенов |
| `ElengalV1` | Полная модель |

### train_elengal.py

Скрипт для обучения на тексте:
- Токенизация по словам
- Обучение с Tsallis-энтропией
- Семантический анализ
- Сохранение модели

## Потоки данных

```
Input Tokens
     │
     ▼
┌─────────────┐
│  Embedding  │
└─────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│         ElengalEvolutionLayer × N        │
│  ┌─────────────────────────────────────┐ │
│  │         ElengalField                │ │
│  │  (рождение токенов в поле)          │ │
│  └─────────────────────────────────────┘ │
│  ┌─────────────────────────────────────┐ │
│  │         ElengalAttention            │ │
│  │  (q-экспонента вместо softmax)      │ │
│  └─────────────────────────────────────┘ │
│  ┌─────────────────────────────────────┐ │
│  │         ElengalCellularFFN          │ │
│  │  (gate от генома и энергии)         │ │
│  └─────────────────────────────────────┘ │
│  ┌─────────────────────────────────────┐ │
│  │      Field.interact()               │ │
│  │  (эволюция физических свойств)      │ │
│  └─────────────────────────────────────┘ │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────┐
│   Output    │
│  (logits)   │
└─────────────┘
```

## Физические свойства токена

Каждый токен в Elengal — это объект с физическими свойствами:

```python
state = ElengalTokenState(batch_size, seq_len, config, device)

# Основные свойства
state.mass            # Масса (семантическая значимость)
state.energy          # Энергия (активность)
state.phase           # Фаза (позиция в пространстве)
state.magnetic_moment # Магнитный момент (полярность)
state.genome          # Геном (ДНК эволюции)
state.q_param         # q-параметр (избирательность)

# Производные
state.phase_velocity  # Скорость изменения фазы
state.gravity_well    # Гравитационный потенциал
state.entropy         # Энтропия
state.lifetime        # Время жизни
```

## Добавление новых полей

```python
class MyCustomField(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Ваши параметры
        
    def interact(self, state, context):
        # Ваша физика
        state.my_property = ...
        return state

# Интеграция в ElengalEvolutionLayer
class ElengalEvolutionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.field = ElengalField(config)
        self.my_field = MyCustomField(config)  # Добавили
        ...
```

## Зависимости

| Библиотека | Версия | Назначение |
|------------|--------|------------|
| torch | ≥2.0.0 | Основной фреймворк |
| numpy | ≥1.24.0 | Вычисления |
| matplotlib | ≥3.7.0 | Визуализация (опционально) |
| scikit-learn | ≥1.2.0 | PCA анализ (опционально) |
