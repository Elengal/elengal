"""
================================================================
  ELENGAL v1.0
  Токены как физические сущности в пространстве смыслов
================================================================

"Токен решает, чем стать — мы только танцуем вокруг"

Автор: Семушкин Александр Геннадьевич / Alexander Gennadyevich Semushkin
Email: Konter88@mail.ru
GitHub: https://github.com/Elengal

Философия:
  - Мы НЕ создаём токены с параметрами
  - Мы создаём ПОЛЕ с условиями
  - Токены появляются, взаимодействуют, СТАНОВЯТСЯ

5 ФУНДАМЕНТАЛЬНЫХ СИЛ:
  1. ГРАВИТАЦИЯ — притяжение смыслов
  2. МАССА — инерция, устойчивость
  3. ВРЕМЯ — стрела развития
  4. МАГНЕТИЗМ — полярность отношений
  5. ГЕНОМ — врождённая способность эволюционировать

ФОРМУЛЫ МИРОЗДАНИЯ:
  - q-экспонента (внимание)
  - Поле Эйзенштейна (комплексное пространство)
  - Pochhammer symbol (квантовая память)
  - Тета-функции Якоби (позиционное кодирование)

АВТОМАТИЧЕСКОЕ ОПРЕДЕЛЕНИЕ УСТРОЙСТВА:
  - CUDA (NVIDIA GPU) — автоматически включает Mixed Precision
  - MPS (Apple Silicon) — оптимизации для M1/M2/M3
  - CPU — универсальный fallback
================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import platform
import sys
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, Literal


# ============================================================================
# 0. АВТОМАТИЧЕСКОЕ ОПРЕДЕЛЕНИЕ УСТРОЙСТВА
# ============================================================================

class ElengalDevice:
    """
    Автоматическое определение и настройка устройства.
    
    Поддерживает:
    - CUDA (NVIDIA GPU) — с Mixed Precision
    - MPS (Apple Silicon M1/M2/M3)
    - CPU — универсальный fallback
    
    Использование:
        device = ElengalDevice.detect()
        device.print_info()
        model = model.to(device.torch_device)
    """
    
    def __init__(
        self,
        device_type: Literal["cuda", "mps", "cpu"],
        torch_device: torch.device,
        name: str,
        memory_gb: Optional[float] = None,
        supports_amp: bool = False,
        supports_bfloat16: bool = False,
        recommended_dtype: str = "float32",
    ):
        self.device_type = device_type
        self.torch_device = torch_device
        self.name = name
        self.memory_gb = memory_gb
        self.supports_amp = supports_amp
        self.supports_bfloat16 = supports_bfloat16
        self.recommended_dtype = recommended_dtype
    
    @classmethod
    def detect(cls) -> "ElengalDevice":
        """Автоматическое определение лучшего доступного устройства."""
        
        # 1. Проверяем CUDA (NVIDIA GPU)
        if torch.cuda.is_available():
            device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            # Проверяем поддержку bfloat16 (Ampere+)
            supports_bf16 = torch.cuda.is_bf16_supported()
            
            return cls(
                device_type="cuda",
                torch_device=device,
                name=gpu_name,
                memory_gb=gpu_memory,
                supports_amp=True,
                supports_bfloat16=supports_bf16,
                recommended_dtype="bfloat16" if supports_bf16 else "float16",
            )
        
        # 2. Проверяем MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            
            # Определяем модель Mac
            mac_name = "Apple Silicon"
            try:
                import subprocess
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    mac_name = result.stdout.strip()
            except:
                pass
            
            return cls(
                device_type="mps",
                torch_device=device,
                name=mac_name,
                memory_gb=None,  # Unified memory
                supports_amp=False,
                supports_bfloat16=True,  # M1+ поддерживает
                recommended_dtype="float32",
            )
        
        # 3. Fallback на CPU
        device = torch.device('cpu')
        cpu_name = platform.processor() or "Unknown CPU"
        
        # Определяем количество ядер
        try:
            import os
            cpu_cores = os.cpu_count() or 1
            cpu_name = f"{cpu_name} ({cpu_cores} cores)"
        except:
            pass
        
        return cls(
            device_type="cpu",
            torch_device=device,
            name=cpu_name,
            memory_gb=None,
            supports_amp=False,
            supports_bfloat16=False,
            recommended_dtype="float32",
        )
    
    def print_info(self):
        """Вывод информации об устройстве."""
        icons = {
            "cuda": "🎮",
            "mps": "🍎", 
            "cpu": "💻"
        }
        icon = icons.get(self.device_type, "🔧")
        
        print(f"\n{icon} ELENGAL — Устройство определено автоматически:")
        print(f"   Тип: {self.device_type.upper()}")
        print(f"   Устройство: {self.name}")
        
        if self.memory_gb:
            print(f"   Память: {self.memory_gb:.1f} GB")
        
        print(f"   Рекомендуемый dtype: {self.recommended_dtype}")
        
        if self.supports_amp:
            print(f"   Mixed Precision: ✅ доступна")
        if self.supports_bfloat16:
            print(f"   BFloat16: ✅ поддерживается")
        
        print()
    
    def get_optimal_dtype(self, requested_dtype: Optional[str] = None) -> str:
        """
        Возвращает оптимальный dtype для данного устройства.
        
        Если requested_dtype указан и поддерживается — возвращает его.
        Иначе возвращает рекомендуемый.
        """
        if requested_dtype:
            # Проверяем поддержку
            if requested_dtype == "bfloat16" and not self.supports_bfloat16:
                return self.recommended_dtype
            if requested_dtype == "float16" and self.device_type == "cpu":
                return "float32"  # CPU не любит float16
            return requested_dtype
        return self.recommended_dtype


def get_device() -> ElengalDevice:
    """Удобная функция для получения устройства."""
    return ElengalDevice.detect()


# ============================================================================
# 1. КОНФИГУРАЦИЯ — ПРОСТАЯ ДЛЯ ЛЮДЕЙ, ГЛУБОКАЯ ВНУТРИ
# ============================================================================

@dataclass
class ElengalConfig:
    """
    Конфигурация модели Elengal.
    
    Простые параметры для людей.
    Но каждый влияет на физику внутри.
    
    dim может быть: 4, 8, 16, 32, 64, 128
    dtype может быть: float16, bfloat16, float32, float64
    device может быть: "auto", "cuda", "mps", "cpu"
    """
    # === РАЗМЕРНОСТЬ ПРОСТРАНСТВА ===
    dim: int = 64
    
    # === ТОЧНОСТЬ ВЫЧИСЛЕНИЙ ===
    dtype: str = "auto"  # "auto" = автоматически по устройству
    
    # === УСТРОЙСТВО ===
    device: str = "auto"  # "auto" = автоматически определить
    
    # === ФИЗИЧЕСКИЕ КОНСТАНТЫ (РЕАЛЬНО влияют на поведение) ===
    
    # Гравитация: сила притяжения смыслов
    # Высокая = токены сильнее притягиваются к "тяжёлым" смыслам
    gravity_constant: float = 1.0
    
    # Магнетизм: полярность взаимодействия
    # Высокая = сильнее притяжение/отталкивание по смыслу
    magnetic_constant: float = 0.5
    
    # Скорость времени: темп эволюции
    # Высокая = быстрая эволюция токенов
    time_speed: float = 1.0
    
    # Затухание энергии: как быстро токены "остывают"
    # Ближе к 1 = медленнее затухание
    energy_decay: float = 0.99
    
    # Скорость мутации генома
    # Высокая = быстрее эволюция, но меньше стабильность
    genome_mutation_rate: float = 0.01
    
    # === РАЗМЕРЫ ВНУТРЕННИХ ПРОСТРАНСТВ ===
    genome_dim: int = 32       # Размер генома (ДНК токена)
    phase_dim: int = 16        # Размерность фазового пространства
    
    # === ПАРАМЕТРЫ q-ИСЧИСЛЕНИЯ ===
    q_base: float = 0.5        # Базовое q для q-экспоненты
    q_adaptive: bool = True    # Адаптивное q (меняется в процессе)
    
    # === ПАРАМЕТРЫ ПОЛЯ ЭЙЗЕНШТЕЙНА ===
    eisenstein_order: int = 3  # Порядок поля (кубические корни)
    
    # === АРХИТЕКТУРА ===
    n_layers: int = 6          # Количество слоёв эволюции
    n_heads: int = 4           # Количество "голов" внимания
    
    # === РЕГУЛЯРИЗАЦИЯ ===
    dropout: float = 0.1
    entropy_bonus: float = 0.01
    
    # === ОПТИМИЗАЦИЯ ===
    use_amp: bool = True       # Использовать Mixed Precision (если доступно)
    
    # Внутренние поля (вычисляются автоматически)
    _elengal_device: Optional[ElengalDevice] = field(default=None, repr=False)
    _resolved_dtype: Optional[str] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Автоматическое определение устройства и dtype после создания."""
        if self._elengal_device is None:
            self._elengal_device = self._resolve_device()
        if self._resolved_dtype is None:
            self._resolved_dtype = self._resolve_dtype()
    
    def _resolve_device(self) -> ElengalDevice:
        """Определение устройства."""
        if self.device == "auto":
            return ElengalDevice.detect()
        else:
            # Ручное указание устройства
            torch_device = torch.device(self.device)
            return ElengalDevice(
                device_type=self.device,
                torch_device=torch_device,
                name=f"Manual: {self.device}",
                supports_amp=(self.device == "cuda"),
                supports_bfloat16=(self.device in ["cuda", "mps"]),
                recommended_dtype="float32",
            )
    
    def _resolve_dtype(self) -> str:
        """Определение оптимального dtype."""
        if self.dtype == "auto":
            return self._elengal_device.get_optimal_dtype()
        return self._elengal_device.get_optimal_dtype(self.dtype)
    
    def get_dtype(self) -> torch.dtype:
        """Получить torch dtype."""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "float64": torch.float64,
        }
        return dtype_map[self._resolved_dtype]
    
    def get_device(self) -> torch.device:
        """Получить torch device."""
        return self._elengal_device.torch_device
    
    def get_elengal_device(self) -> ElengalDevice:
        """Получить ElengalDevice."""
        return self._elengal_device
    
    def should_use_amp(self) -> bool:
        """Нужно ли использовать Mixed Precision."""
        return self.use_amp and self._elengal_device.supports_amp
    
    def print_device_info(self):
        """Вывести информацию об устройстве."""
        self._elengal_device.print_info()
        print(f"   Используемый dtype: {self._resolved_dtype}")
        if self.should_use_amp():
            print(f"   AMP включён: ✅")
            
# ============================================================================
# 2. МАТЕМАТИКА МИРОЗДАНИЯ — ФОРМУЛЫ ELENGAL
# ============================================================================

class ElengalMath:
    """
    Математика Elengal.
    
    Эти формулы описывают фундаментальные свойства Вселенной.
    Мы применяем их к смыслам.
    """
    
    @staticmethod
    def pochhammer_symbol(a: torch.Tensor, q: torch.Tensor, n: int) -> torch.Tensor:
        """
        Символ Похгаммера (a; q)_n = П(1 - a*q^k), k=0..n-1
        
        Это "квантовая память" — формулы для q-рядов.
        При q → 1 превращается в обычную математику.
        
        Общий случай — используется в других q-рядах.
        """
        result = torch.ones_like(a)
        for k in range(n):
            result = result * (1 - a * (q ** k))
        return result
    
    @staticmethod
    def q_pochhammer(q: torch.Tensor, n: int) -> torch.Tensor:
        """
        q-символ Похгаммера (q; q)_n = П_{k=1}^{n} (1 - q^k)
        
        Специальный случай для q-экспоненты.
        Отличается от общего pochhammer_symbol: k начинается с 1, а не с 0.
        
        При q=0.5:
          (q; q)_0 = 1
          (q; q)_1 = 0.5
          (q; q)_2 = 0.375
          (q; q)_3 = 0.328
        
        При q → 1: (q; q)_n → n! (факториал)
        """
        if n == 0:
            return torch.ones_like(q)
        
        # Векторизованная версия — быстрее и численно стабильнее
        k = torch.arange(1, n + 1, device=q.device, dtype=q.dtype)
        factors = 1 - q.unsqueeze(-1) ** k
        
        return factors.prod(dim=-1)
    
    @staticmethod
    def q_number(n: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        q-число: [n]_q = (1 - q^n) / (1 - q)
        
        Это "квантовое" обобщение обычного числа.
        При q → 1: [n]_q → n
        """
        return (1 - q ** n) / (1 - q + 1e-8)
    
    @staticmethod
    def q_exponential(x: torch.Tensor, q: torch.Tensor, terms: int = 10) -> torch.Tensor:
        """
        Классическая q-экспонента: exp_q(x) = Σ_{n=0}^{∞} x^n / (q; q)_n
        
        Это НЕ Tsallis-экспонента! Это настоящая q-экспонента из q-анализа.
        
        Особенности:
        - При q → 1: exp_q(x) → exp(x) (обычная экспонента)
        - При q < 1: ряд растёт быстрее — "суперэкспоненциальное" поведение
        - Это создаёт уникальную нелинейную динамику, невозможную для обычных трансформеров
        
        Используется в ElengalAttention для создания уникального механизма внимания.
        """
        result = torch.zeros_like(x)
        for n in range(terms):
            poch = ElengalMath.q_pochhammer(q, n)
            term = (x ** n) / (poch + 1e-8)
            result = result + term
        return result
    
    @staticmethod
    def tsallis_entropy(probs: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        Tsallis q-энтропия: S_q = (1 - Σ p_i^q) / (q - 1)
        
        Физический смысл:
        - При q=1: обычная энтропия Шеннона (мера неопределённости)
        - При q<1: поощряет концентрацию (токен "решает" стать специфичным)
        - При q>1: поощряет разнообразие (токен "решает" быть гибким)
        
        В Elengal используется для регуляризации внимания:
        - Токены с разным геномом = разная энтропийная динамика
        """
        q_val = q.item() if isinstance(q, torch.Tensor) and q.numel() == 1 else q
        
        if abs(q_val - 1.0) < 1e-6:
            # Энтропия Шеннона (предел при q→1)
            return -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        
        # Tsallis энтропия
        return (1 - (probs ** q_val).sum(dim=-1)) / (q_val - 1)
    
    @staticmethod
    def genome_to_q(genome: torch.Tensor) -> torch.Tensor:
        """
        Преобразует геном в q-параметр.
        
        Философия: геном = ДНК токена, определяет его "квантовый характер".
        Каждый токен решает свою q через геном:
        - Гены → q ∈ [0.3, 0.99]
        - q < 0.5: горячий, концентрированный, мало связей
        - q > 0.9: холодный, размытый, много связей
        
        Это Вариант 2: адаптивное q из генома.
        """
        # Сжимаем геном до скаляра и преобразуем в [0.3, 0.99]
        genome_signal = genome.mean(dim=-1, keepdim=True)  # [B, T, 1]
        # sigmoid → [0, 1], затем масштабируем → [0.3, 0.99]
        q = torch.sigmoid(genome_signal) * 0.69 + 0.3
        return q
    
    @staticmethod
    def q_to_temperature(q: torch.Tensor) -> torch.Tensor:
        """
        Преобразует q в температуру для масштабирования scores.
        
        Физика:
        - q < 1 → температура низкая → внимание острое
        - q → 1 → температура высокая → внимание размытое
        
        Формула: T = 1 / (1 - q + ε)
        
        Это Вариант 4: температура из q.
        """
        q_val = q.item() if isinstance(q, torch.Tensor) and q.numel() == 1 else q
        return 1.0 / (1.0 - q_val + 0.1)  # +0.1 для стабильности
    
    @staticmethod
    def q_to_sparsity(q: torch.Tensor, seq_len: int) -> int:
        """
        Преобразует q в количество сохраняемых связей (top-k).
        
        Физика:
        - q < 0.5: мало связей (избирательное внимание)
        - q > 0.9: много связей (внимание на всё)
        
        Это Вариант 5: спарсификация через q.
        """
        q_val = q.item() if isinstance(q, torch.Tensor) and q.numel() == 1 else q
        
        # q=0.3 → 10% связей, q=0.99 → 100% связей
        ratio = min(1.0, (q_val - 0.3) / 0.69)  # [0, 1]
        k = max(1, int(seq_len * (0.1 + 0.9 * ratio)))
        return k
    
    @staticmethod
    def eisenstein_phase(x: torch.Tensor, order: int = 3) -> torch.Tensor:
        """
        Поле Эйзенштейна: ω = e^(2πi/order)
        
        Создаёт "кристаллическую" структуру пространства.
        Для order=3 это кубические корни из 1 (равносторонний треугольник).
        """
        angle = 2 * math.pi / order
        x_angle = x * angle
        return torch.cos(x_angle)  # Возвращаем действительную часть
    
    @staticmethod
    def jacobi_theta(z: torch.Tensor, tau: torch.Tensor, terms: int = 10) -> torch.Tensor:
        """
        Тета-функция Якоби: θ(z, τ) = Σ (-1)^n q^(n(n-1)) sin((2n-1)πz)
        
        Используется для позиционного кодирования.
        τ определяет "температуру" решётки.
        """
        if tau.numel() == 1:
            q_angle = math.pi * tau.item()
        else:
            q_angle = math.pi * tau.mean().item()
        
        result = torch.zeros_like(z)
        
        for n in range(1, terms + 1):
            k = n * (n - 1)
            q_pow_real = math.cos(k * q_angle)
            sin_term = torch.sin((2 * n - 1) * math.pi * z)
            term = ((-1) ** n) * q_pow_real * sin_term
            result = result + term
        
        return 2 * result


# ============================================================================
# 3. СОСТОЯНИЕ ТОКЕНА — ФИЗИЧЕСКАЯ СУЩНОСТЬ
# ============================================================================

class ElengalTokenState:
    """
    Состояние токена Elengal как физической сущности.
    
    Токен — это НЕ вектор.
    Токен — это совокупность физических свойств.
    
    Каждое свойство имеет физический смысл:
    - Масса = инерция смысла
    - Энергия = активность
    - Фаза = позиция в семантическом пространстве
    - Магнетизм = полярность
    - Время = развитие
    - Геном = ДНК эволюции
    
    Токен рождается "голым" и ОБРЕТАЕТ свойства через взаимодействие.
    """
    
    def __init__(self, batch_size: int, seq_len: int, config: ElengalConfig, device: torch.device):
        self.config = config
        self.device = device
        self.B = batch_size
        self.T = seq_len
        self.dim = config.dim
        self.dtype = config.get_dtype()
        
        # === 1. МАССА (инерция смысла) ===
        # Начальная масса с небольшим шумом — каждый токен уникален!
        self.mass = torch.ones(self.B, self.T, 1, device=device, dtype=self.dtype) * 0.1
        self.mass = self.mass + torch.randn_like(self.mass) * 0.02  # Небольшой шум
        self.mass = torch.clamp(self.mass, min=0.01, max=1.0)  # Начальный диапазон [0.01, 1.0]
        self.mass_center = torch.zeros(self.B, self.T, self.dim, device=device, dtype=self.dtype)
        
        # === 2. ЭНЕРГИЯ (активность) ===
        # Начальная энергия с шумом
        self.energy = torch.rand(self.B, self.T, 1, device=device, dtype=self.dtype) * 0.1
        self.energy_flow = torch.zeros(self.B, self.T, self.dim, device=device, dtype=self.dtype)
        
        # === 3. ФАЗА (позиция в семантическом пространстве) ===
        # Начальная фаза — случайная для каждого токена
        self.phase = torch.randn(self.B, self.T, config.phase_dim, device=device, dtype=self.dtype) * 0.1
        self.phase_velocity = torch.zeros(self.B, self.T, config.phase_dim, device=device, dtype=self.dtype)
        
        # === 4. МАГНЕТИЗМ (полярность) ===
        # Начальный магнитный момент с шумом
        self.magnetic_moment = torch.randn(self.B, self.T, 1, device=device, dtype=self.dtype) * 0.1
        self.magnetic_direction = torch.zeros(self.B, self.T, self.dim, device=device, dtype=self.dtype)
        
        # === 5. ВРЕМЯ (развитие) ===
        self.time_position = torch.zeros(self.B, self.T, 1, device=device, dtype=self.dtype)
        self.time_velocity = torch.ones(self.B, self.T, 1, device=device, dtype=self.dtype)
        
        # === 6. ГЕНОМ (врождённая эволюция) ===
        self.genome = torch.zeros(self.B, self.T, config.genome_dim, device=device, dtype=self.dtype)
        self.genome_expression = torch.zeros(self.B, self.T, 1, device=device, dtype=self.dtype)
        
        # === ДОПОЛНИТЕЛЬНЫЕ ФИЗИЧЕСКИЕ СВОЙСТВА ===
        self.gravity_well = torch.zeros(self.B, self.T, 1, device=device, dtype=self.dtype)
        self.entropy = torch.zeros(self.B, self.T, 1, device=device, dtype=self.dtype)
        self.spin = torch.zeros(self.B, self.T, 1, device=device, dtype=self.dtype)
        self.charge = torch.zeros(self.B, self.T, 1, device=device, dtype=self.dtype)
        self.lifetime = torch.ones(self.B, self.T, 1, device=device, dtype=self.dtype)
        self.q_param = torch.ones(self.B, self.T, 1, device=device, dtype=self.dtype) * config.q_base
    
    def to_vector(self) -> torch.Tensor:
        """Конвертация в вектор для совместимости."""
        components = [
            self.mass_center,
            self.energy_flow * self.energy,
            self.phase[:, :, :self.dim] if self.phase.shape[-1] >= self.dim 
                else F.pad(self.phase, (0, self.dim - self.phase.shape[-1])),
            self.magnetic_direction * self.magnetic_moment,
            self.genome[:, :, :self.dim] if self.genome.shape[-1] >= self.dim 
                else F.pad(self.genome, (0, self.dim - self.genome.shape[-1])),
        ]
        vector = sum(components) / len(components)
        return vector
    
    def get_physical_summary(self) -> Dict[str, float]:
        """Получить сводку физических параметров (для отладки)."""
        return {
            "mass_mean": self.mass.mean().item(),
            "mass_std": self.mass.std().item(),  # Разнообразие массы!
            "energy_mean": self.energy.mean().item(),
            "energy_std": self.energy.std().item(),
            "magnetic_mean": self.magnetic_moment.mean().item(),
            "entropy_mean": self.entropy.mean().item(),
            "time_mean": self.time_position.mean().item(),
            "genome_norm": self.genome.norm().item(),
            "q_param_mean": self.q_param.mean().item(),
            "q_param_std": self.q_param.std().item(),  # Разнообразие q!
        }


# ============================================================================
# 4. ФУНДАМЕНТАЛЬНОЕ ПОЛЕ — ПРОСТРАНСТВО, ГДЕ ЖИВУТ ТОКЕНЫ
# ============================================================================

class ElengalField(nn.Module):
    """
    Фундаментальное поле Elengal.
    
    Мы НЕ создаём токены.
    Мы создаём УСЛОВИЯ, в которых токены появляются и живут.
    
    Поле имеет 5 фундаментальных составляющих:
    1. Гравитационный ландшафт
    2. Магнитную структуру
    3. Временной поток
    4. Энергетические источники
    5. Геномный потенциал
    
    Токен "чувствует" поле и через взаимодействие ОБРЕТАЕТ свойства.
    """
    
    def __init__(self, config: ElengalConfig):
        super().__init__()
        self.config = config
        self.dim = config.dim
        
        # === 1. ГРАВИТАЦИОННОЕ ПОЛЕ ===
        self.gravity_landscape = nn.Parameter(
            torch.randn(config.dim, dtype=config.get_dtype()) * 0.1
        )
        self.gravity_masses = nn.Parameter(
            torch.randn(config.dim, dtype=config.get_dtype()) * 0.1
        )
        
        # === 2. МАГНИТНОЕ ПОЛЕ ===
        self.magnetic_field = nn.Parameter(
            torch.randn(config.dim, dtype=config.get_dtype()) * 0.1
        )
        self.magnetic_domains = nn.Parameter(
            torch.randn(config.dim, dtype=config.get_dtype()) * 0.1
        )
        
        # === 3. ВРЕМЕННОЕ ПОЛЕ ===
        self.time_arrow = nn.Parameter(
            torch.ones(config.dim, dtype=config.get_dtype())
        )
        self.time_flow = nn.Parameter(
            torch.randn(config.dim, dtype=config.get_dtype()) * 0.1
        )
        
        # === 4. ЭНЕРГЕТИЧЕСКОЕ ПОЛЕ ===
        self.energy_sources = nn.Parameter(
            torch.randn(config.dim, dtype=config.get_dtype()) * 0.1
        )
        self.energy_sinks = nn.Parameter(
            torch.randn(config.dim, dtype=config.get_dtype()) * 0.1
        )
        
        # === 5. ГЕНОМНОЕ ПОЛЕ ===
        self.genome_potential = nn.Parameter(
            torch.randn(config.genome_dim, dtype=config.get_dtype()) * 0.1
        )
        self.ancestral_genes = nn.Parameter(
            torch.randn(config.genome_dim, dtype=config.get_dtype()) * 0.1
        )
        
        # === 6. ТЁМНАЯ МАТЕРИЯ/ЭНЕРГИЯ ===
        self.dark_matter = nn.Parameter(
            torch.randn(config.dim, dtype=config.get_dtype()) * 0.01
        )
        self.dark_energy = nn.Parameter(
            torch.randn(config.dim, dtype=config.get_dtype()) * 0.01
        )
        
        # === 7. ПОЛЕ ЭЙЗЕНШТЕЙНА ===
        self.eisenstein_field = nn.Parameter(
            torch.randn(config.dim, dtype=config.get_dtype()) * 0.1
        )
        
        # === 8. q-ПОЛЕ ===
        self.q_field = nn.Parameter(
            torch.ones(config.dim, dtype=config.get_dtype()) * config.q_base
        )
    
    def birth(self, batch_size: int, seq_len: int, device: torch.device) -> ElengalTokenState:
        """
        Рождение токенов в поле.
        
        Токены появляются "голыми" — без предустановленных свойств.
        Они ОБРЕТУТ свойства через взаимодействие с полем.
        """
        state = ElengalTokenState(batch_size, seq_len, self.config, device)
        
        with torch.no_grad():
            # Начальная фаза — случайное возмущение поля
            initial_phase = torch.randn(
                batch_size, seq_len, self.config.phase_dim, 
                device=device, dtype=self.config.get_dtype()
            ) * 0.1
            state.phase = initial_phase
            
            # Начальная масса — минимальная
            state.mass = torch.ones(batch_size, seq_len, 1, device=device, dtype=self.config.get_dtype()) * 0.1
            
            # Начальная энергия — от поля
            phase_for_energy = state.phase[:, :, :min(self.config.phase_dim, self.dim)]
            energy_sources_reduced = self.energy_sources[:phase_for_energy.shape[-1]]
            energy_inflow = (phase_for_energy * energy_sources_reduced).sum(dim=-1, keepdim=True)
            state.energy = torch.abs(energy_inflow) * 0.1
            
            # Начальный геном — "ДНК" из поля
            genome_noise = torch.randn(
                batch_size, seq_len, self.config.genome_dim,
                device=device, dtype=self.config.get_dtype()
            )
            state.genome = genome_noise * 0.1 + self.genome_potential * 0.1
            
            # Начальный магнитный момент
            phase_for_magnetic = state.phase[:, :, :min(self.config.phase_dim, self.dim)]
            magnetic_field_reduced = self.magnetic_field[:phase_for_magnetic.shape[-1]]
            magnetic_interaction = (phase_for_magnetic * magnetic_field_reduced).sum(dim=-1, keepdim=True)
            state.magnetic_moment = torch.tanh(magnetic_interaction)
            
            # Начальное время — позиция в контексте
            positions = torch.arange(seq_len, device=device, dtype=self.config.get_dtype())
            state.time_position = positions.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1)
            
            # Начальный q-параметр
            state.q_param = torch.ones(batch_size, seq_len, 1, device=device, dtype=self.config.get_dtype()) * self.config.q_base
        
        return state
        
    def compute_gravity_force(self, state: ElengalTokenState) -> torch.Tensor:
        """
        Гравитационная сила между токенами.
        F_gravity = G × m₁ × m₂ / r²
        
        Токены с большей массой притягивают сильнее.
        """
        phase_i = state.phase.unsqueeze(2)  # [B, T, 1, phase_dim]
        phase_j = state.phase.unsqueeze(1)  # [B, 1, T, phase_dim]
        r = torch.norm(phase_i - phase_j, dim=-1) + 1e-6  # [B, T, T]
        
        m_i = state.mass.view(state.B, state.T, 1)
        m_j = state.mass.view(state.B, 1, state.T)
        
        G = self.config.gravity_constant
        gravity = G * m_i * m_j / (r ** 2 + 1e-6)
        
        # Каузальная маска
        mask = torch.tril(torch.ones(state.T, state.T, device=state.device))
        gravity = gravity * mask
        
        return gravity
    
    def compute_magnetic_force(self, state: ElengalTokenState) -> torch.Tensor:
        """
        Магнитная сила между токенами.
        U = -μ₁ × μ₂
        
        Одноимённые полюса отталкиваются, разноимённые притягиваются.
        """
        mu = state.magnetic_moment.view(state.B, state.T)
        
        mu_i = mu.unsqueeze(2)  # [B, T, 1]
        mu_j = mu.unsqueeze(1)  # [B, 1, T]
        
        magnetic = -mu_i * mu_j * self.config.magnetic_constant
        interaction = torch.exp(-magnetic)
        
        # Каузальная маска
        mask = torch.tril(torch.ones(state.T, state.T, device=state.device))
        interaction = interaction * mask
        
        return interaction
    
    def compute_time_flow(self, state: ElengalTokenState) -> torch.Tensor:
        """Течение времени для токенов."""
        phase_slice = state.phase[:, :, :min(state.phase.shape[-1], self.dim)]
        time_arrow_reduced = self.time_arrow[:phase_slice.shape[-1]]
        time_shift = (phase_slice * time_arrow_reduced).sum(dim=-1, keepdim=True)
        flow = time_shift * state.time_velocity * self.config.time_speed
        return flow
    
    def compute_energy_exchange(self, state: ElengalTokenState) -> torch.Tensor:
        """Обмен энергией с полем."""
        phase_slice = state.phase[:, :, :min(state.phase.shape[-1], self.dim)]
        energy_sources_reduced = self.energy_sources[:phase_slice.shape[-1]]
        inflow = (phase_slice * energy_sources_reduced).sum(dim=-1, keepdim=True)
        decay = state.energy * (1 - self.config.energy_decay)
        delta = inflow * 0.1 - decay * 0.1
        return delta
    
    def evolve_genome(self, state: ElengalTokenState, gradient: torch.Tensor) -> torch.Tensor:
        """
        Эволюция генома через q-числа.
        genome_new = genome + α × [||x||]_q × pressure
        """
        grad_norm = gradient.norm(dim=-1, keepdim=True)
        q_grad = ElengalMath.q_number(grad_norm, state.q_param)
        
        if state.phase.shape[-1] >= self.config.genome_dim:
            pressure = (state.phase[:, :, :self.config.genome_dim] * self.genome_potential).sum(
                dim=-1, keepdim=True
            )
        else:
            pressure = torch.zeros(state.B, state.T, 1, device=state.device, dtype=self.config.get_dtype())
        
        mutation = pressure * self.config.genome_mutation_rate * q_grad
        return mutation
    
    def interact(self, state: ElengalTokenState, context: torch.Tensor) -> ElengalTokenState:
        """
        Взаимодействие токена с полем.
        
        Через это взаимодействие токен ОБРЕТАЕТ свойства.
        "Токен решает, чем стать — мы только танцуем вокруг"
        
        ФИЗИКА МАССЫ:
        ─────────────────────────────────────────────────────
        Масса = мера "семантической значимости" токена.
        
        ДИНАМИЧЕСКОЕ РАВНОВЕСИЕ:
        - Масса может расти И убывать
        - Есть "источники" массы и "стоки"
        - Равновесие зависит от активности токена
        
        Источники (масса растёт):
        → Популярность (на токен смотрят другие)
        → Генетический потенциал
        
        Стоки (масса убывает):
        → Изоляция (на токен никто не смотрит)
        → Энтропийное рассеяние
        
        Модуляторы:
        → q-параметр (избирательность)
        → Энергия (активность ускоряет обмен)
        ─────────────────────────────────────────────────────
        """
        # === Обновление массы (ИНДИВИДУАЛЬНО для каждого токена) ===
        
        # 1. Гравитация: вычисляем "популярность" токена
        gravity_force = self.compute_gravity_force(state)
        # popularity: сколько токенов "смотрят" на этот токен
        # [B, T, T] → суммируем по dim=-2 (кто смотрит)
        popularity = gravity_force.sum(dim=-2, keepdim=True).transpose(-1, -2)  # [B, T, 1]
        
        # Нормализуем популярность относительно среднего
        popularity_normalized = popularity / (popularity.mean() + 1e-6)
        
        # 2. Источник массы: популярность (> 1 = популярный, растёт)
        mass_source = torch.relu(popularity_normalized - 1.0) * 0.01
        
        # 3. Сток массы: изоляция (< 1 = изолированный, убывает)
        mass_sink = torch.relu(1.0 - popularity_normalized) * 0.005
        
        # 4. Геном: потенциал (может быть + или -)
        genome_effect = state.genome.mean(dim=-1, keepdim=True) * 0.002  # [B, T, 1]
        
        # 5. q-параметр: модуляция (низкое q = концентрация массы)
        q_modulation = (0.65 - state.q_param) * 0.005  # q < 0.65 → рост, q > 0.65 → убывание
        
        # 6. Энтропийное рассеяние: масса "испаряется" со временем
        # Но активные токены сопротивляются рассеянию
        entropy_decay = 0.002 * state.mass / (state.energy + 0.1)
        
        # Итоговое изменение массы
        mass_delta = (
            mass_source           # Рост от популярности
            - mass_sink           # Убыль от изоляции
            + genome_effect       # Генетический потенциал
            + q_modulation        # q-модуляция
            - entropy_decay       # Естественное рассеяние
        )
        
        # Применяем
        state.mass = state.mass + mass_delta
        state.mass = torch.clamp(state.mass, min=0.01, max=10.0)
        
        # === Обновление энергии (ИНДИВИДУАЛЬНО) ===
        energy_delta = self.compute_energy_exchange(state)
        state.energy = state.energy + energy_delta
        state.energy = torch.clamp(state.energy, min=0.0, max=10.0)
        
        # === Обновление фазы ===
        time_flow = self.compute_time_flow(state)
        state.phase = state.phase + state.phase_velocity * time_flow * 0.1
        
        # Обновление фазовой скорости (от контекста и энергии)
        phase_dim = state.phase.shape[-1]
        context_for_phase = context[:, :, :phase_dim] if context.shape[-1] >= phase_dim else F.pad(context, (0, phase_dim - context.shape[-1]))
        
        # Ускорение от контекста + энергия
        acceleration = context_for_phase * 0.01 + state.energy * 0.001
        state.phase_velocity = state.phase_velocity + acceleration
        
        # Затухание (фрикция в фазовом пространстве)
        state.phase_velocity = state.phase_velocity * 0.99
        
        # === Обновление магнитного момента (ИНДИВИДУАЛЬНО) ===
        magnetic_force = self.compute_magnetic_force(state)
        # Каждый токен получает своё изменение
        magnetic_delta = magnetic_force.sum(dim=-1, keepdim=True) * 0.01  # [B, T, 1]
        state.magnetic_moment = state.magnetic_moment + magnetic_delta
        state.magnetic_moment = torch.tanh(state.magnetic_moment)
        
        # === Обновление времени ===
        state.time_position = state.time_position + state.time_velocity * self.config.time_speed * 0.01
        
        # === Эволюция генома ===
        genome_mutation = self.evolve_genome(state, context)
        state.genome = state.genome + genome_mutation * 0.01
        
        # === Обновление энтропии ===
        state.entropy = state.entropy + torch.abs(state.phase_velocity).mean(dim=-1, keepdim=True) * 0.01
        
        # Тёмное влияние
        phase_slice = state.phase[:, :, :min(state.phase.shape[-1], self.dim)]
        dark_matter_reduced = self.dark_matter[:phase_slice.shape[-1]]
        dark_effect = (phase_slice * dark_matter_reduced).sum(dim=-1, keepdim=True) * 0.001
        state.phase = state.phase + dark_effect * 0.1
        
        # Адаптация q-параметра
        state.q_param = torch.clamp(
            state.q_param + state.genome_expression * 0.01,
            min=0.1, max=0.99
        )
        
        return state


# ============================================================================
# 5. q-ВНИМАНИЕ ELENGAL
# ============================================================================

class ElengalAttention(nn.Module):
    """
    Внимание Elengal на основе q-экспоненты.
    
    Особенности:
    - q-экспонента вместо softmax
    - Фазовый поворот Q и K (волновая интерференция)
    - Влияние массы и магнетизма
    """
    
    def __init__(self, config: ElengalConfig):
        super().__init__()
        self.config = config
        
        self.q_proj = nn.Linear(config.dim, config.dim)
        self.k_proj = nn.Linear(config.dim, config.dim)
        self.v_proj = nn.Linear(config.dim, config.dim)
        self.out_proj = nn.Linear(config.dim, config.dim)
        
        # Проекция фазы для поворота
        self.phase_proj = nn.Linear(config.phase_dim, config.dim)
        
        self.adaptive_q = nn.Parameter(torch.ones(config.n_heads) * config.q_base)
        self.dropout = nn.Dropout(config.dropout)
    
    def apply_phase_rotation(self, x: torch.Tensor, phase_angles: torch.Tensor) -> torch.Tensor:
        """
        Применяет фазовый поворот к тензору (Q или K).
        
        Это создаёт волновую интерференцию:
        - Токены с близкой фазой → усиливаются
        - Токены с разной фазой → ослабляются
        
        Аналогично Rotary Position Embeddings, но фаза из состояния токена.
        """
        # phase_angles: [B, T, dim]
        # Разбиваем на пары для вращения
        x_rot = x.reshape(*x.shape[:-1], -1, 2)  # [B, T, dim//2, 2]
        
        cos_theta = torch.cos(phase_angles[..., ::2])  # [B, T, dim//2]
        sin_theta = torch.sin(phase_angles[..., ::2])
        
        # Вращение: [x0*cos - x1*sin, x0*sin + x1*cos]
        x0, x1 = x_rot[..., 0], x_rot[..., 1]
        rotated = torch.stack([
            x0 * cos_theta - x1 * sin_theta,
            x0 * sin_theta + x1 * cos_theta
        ], dim=-1)
        
        return rotated.reshape(*x.shape)
    
    def forward(self, x: torch.Tensor, state: ElengalTokenState) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Единый организм q-внимания.
        
        Цепочка преобразований:
        1. Геном → q (Вариант 2)
        2. q → температура (Вариант 4)
        3. scores / температура → q-экспонента
        4. q → top_k спарсификация (Вариант 5)
        5. Возвращает q для Tsallis энтропии (Вариант 3)
        
        Returns:
            output: [B, T, dim]
            attention_weights: [B, T, T]
            q_avg: скаляр для tsallis_entropy в loss
        """
        B, T, dim = x.shape
        
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # === 1. ГЕНОМ → q (Вариант 2) ===
        # Токен решает свою q через геном
        q_from_genome = ElengalMath.genome_to_q(state.genome)
        q_avg = q_from_genome.mean()  # Усредняем для всей последовательности
        
        # Обновляем q_param в состоянии (для физики)
        state.q_param = q_from_genome
        
        # === 2. q → ТЕМПЕРАТУРА (Вариант 4) ===
        temperature = ElengalMath.q_to_temperature(q_avg)
        
        # === ФАЗОВЫЙ ПОВОРОТ ===
        phase_angles = self.phase_proj(state.phase)
        Q_rotated = self.apply_phase_rotation(Q, phase_angles)
        K_rotated = self.apply_phase_rotation(K, phase_angles)
        
        # Скоры с температурой
        scores = torch.matmul(Q_rotated, K_rotated.transpose(-2, -1)) / math.sqrt(dim)
        scores = scores / temperature  # Масштабируем температурой
        
        # === q-ЭКСПОНЕНТА ===
        attention = ElengalMath.q_exponential(scores, q_avg, terms=8)
        
        # === ВЛИЯНИЕ ФИЗИЧЕСКИХ СВОЙСТВ ===
        gravity_boost = self.config.gravity_constant * state.mass.view(B, 1, T)
        attention = attention + gravity_boost * 0.1
        
        magnetic_mod = 1.0 + state.magnetic_moment.view(B, T, 1) * self.config.magnetic_constant
        attention = attention * magnetic_mod
        
        # === 3. СПАРСИФИКАЦИЯ (Вариант 5) ===
        # q определяет сколько связей оставить
        top_k = ElengalMath.q_to_sparsity(q_avg, T)
        
        # === КАУЗАЛЬНАЯ МАСКА ===
        mask = torch.tril(torch.ones(T, T, device=x.device))
        attention = attention.masked_fill(mask == 0, float('-inf'))
        
        # Top-k маска (только для ненулевых позиций)
        if top_k < T:
            # Находим top-k для каждой позиции
            top_k_mask = torch.zeros_like(attention)
            for t in range(T):
                valid_positions = t + 1  # Каузально доступные
                k_for_t = min(top_k, valid_positions)
                if k_for_t > 0:
                    _, top_indices = attention[:, t, :valid_positions].topk(k_for_t, dim=-1)
                    top_k_mask[:, t, :valid_positions].scatter_(-1, top_indices, 1.0)
            attention = attention.masked_fill(top_k_mask == 0, float('-inf'))
        
        attention_weights = F.softmax(attention, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        output = self.out_proj(output)
        
        # Возвращаем q_avg для Tsallis энтропии в loss (Вариант 3)
        return output, attention_weights, q_avg


# ============================================================================
# 6. КЛЕТОЧНАЯ FFN ELENGAL
# ============================================================================

class ElengalCellularFFN(nn.Module):
    """
    Клеточная FFN Elengal — трансформация через "деление".
    
    Gate от генома и энергии управляет активацией.
    Каждый токен — свой pattern трансформации.
    """
    
    def __init__(self, config: ElengalConfig):
        super().__init__()
        self.config = config
        
        self.up_proj = nn.Linear(config.dim, config.dim * 4)
        self.down_proj = nn.Linear(config.dim * 4, config.dim)
        
        self.gate_proj = nn.Linear(config.genome_dim, config.dim * 4)
        self.energy_gate = nn.Linear(1, config.dim * 4)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor, state: ElengalTokenState) -> torch.Tensor:
        up = self.up_proj(x)
        
        gate = torch.sigmoid(self.gate_proj(state.genome))
        energy_gate = torch.sigmoid(self.energy_gate(state.energy))
        
        combined_gate = gate * energy_gate
        
        up = F.gelu(up) * combined_gate
        output = self.down_proj(up)
        output = self.dropout(output)
        
        return output


# ============================================================================
# 7. СЛОЙ ЭВОЛЮЦИИ ELENGAL
# ============================================================================

class ElengalEvolutionLayer(nn.Module):
    """
    Слой эволюции токенов Elengal.
    
    Токены взаимодействуют с полем и друг с другом.
    Через это взаимодействие они эволюционируют.
    """
    
    def __init__(self, config: ElengalConfig):
        super().__init__()
        self.config = config
        
        self.field = ElengalField(config)
        self.attention = ElengalAttention(config)
        self.ffn = ElengalCellularFFN(config)
        
        self.norm1 = nn.LayerNorm(config.dim)
        self.norm2 = nn.LayerNorm(config.dim)
        self.state_proj = nn.Linear(config.dim, config.dim)
    
    def forward(self, x: torch.Tensor, state: ElengalTokenState) -> Tuple[torch.Tensor, ElengalTokenState, torch.Tensor]:
        """
        Эволюция токена через поле.
        
        Returns:
            x: трансформированный тензор
            state: обновлённое состояние
            q_avg: q-параметр для Tsallis энтропии
        """
        attn_out, _, q_avg = self.attention(self.norm1(x), state)
        x = x + attn_out
        
        ffn_out = self.ffn(self.norm2(x), state)
        x = x + ffn_out
        
        context = self.state_proj(x)
        state = self.field.interact(state, context)
        
        return x, state, q_avg
        
# ============================================================================
# 8. ELENGAL v1 — ПОЛНАЯ МОДЕЛЬ
# ============================================================================

class ElengalV1(nn.Module):
    """
    Elengal v1 — модель на основе физического поля смыслов.
    
    Ключевые отличия от обычного трансформера:
    - Токены — это не векторы, а физические сущности
    - Внимание — это не умножение матриц, а гравитация/магнетизм
    - Позиция — это не индекс, а фаза в поле
    - Обновление — это эволюция, не трансформация
    
    Автор: Семушкин Александр Геннадьевич / Alexander Gennadyevich Semushkin
    """
    
    def __init__(
        self,
        vocab_size: int,
        config: Optional[ElengalConfig] = None,
    ):
        super().__init__()
        
        self.config = config or ElengalConfig()
        self.vocab_size = vocab_size
        
        # Точка входа токенов в поле
        self.entry_point = nn.Embedding(vocab_size, self.config.dim)
        
        # Позиционное кодирование через тета-функции Якоби
        self.pos_tau = nn.Parameter(torch.ones(1) * 0.5)
        
        # Эволюционные слои
        self.layers = nn.ModuleList([
            ElengalEvolutionLayer(self.config)
            for _ in range(self.config.n_layers)
        ])
        
        # Выход — "горизонт событий"
        self.event_horizon = nn.Linear(self.config.dim, vocab_size)
        
        self.entropy_bonus = self.config.entropy_bonus
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ElengalTokenState, torch.Tensor, torch.Tensor]:
        """
        Прямой проход через Elengal.
        
        Returns:
            logits: [B, T, vocab_size]
            state: финальное физическое состояние токенов
            attention_weights: веса внимания для Tsallis энтропии
            q_avg: средний q-параметр для Tsallis энтропии
        """
        B, T = x.shape
        device = x.device
        
        # Рождение токенов в поле
        state = self.layers[0].field.birth(B, T, device)
        
        # Точка входа + позиционное кодирование
        entry = self.entry_point(x)
        
        positions = torch.arange(T, device=device, dtype=torch.float32)
        pos_encoding = ElengalMath.jacobi_theta(
            positions / T, 
            self.pos_tau.expand(T), 
            terms=5
        ).unsqueeze(0).unsqueeze(-1).expand(B, -1, self.config.dim)
        
        x = entry + pos_encoding.to(entry.dtype)
        
        # Обновляем фазу токенов
        phase_update_dim = min(state.phase.shape[-1], x.shape[-1])
        state.phase[:, :, :phase_update_dim] = state.phase[:, :, :phase_update_dim] + x[:, :, :phase_update_dim] * 0.1
        
        # Эволюция через слои — собираем q_avg и attention_weights
        q_avg_total = 0.0
        attention_weights_total = None
        
        for layer in self.layers:
            x, state, q_avg = layer(x, state)
            q_avg_total = q_avg_total + q_avg
        
        # Усредняем q по слоям
        q_avg_total = q_avg_total / len(self.layers)
        
        # Горизонт событий
        logits = self.event_horizon(x)
        
        return logits, state, attention_weights_total, q_avg_total
    
    def compute_loss(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor,
        state: ElengalTokenState,
        attention_weights: torch.Tensor = None,
        q_avg: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Вычисление loss с Tsallis q-энтропией.
        
        Единый организм:
        - Cross-entropy: основная задача
        - Tsallis entropy (Вариант 3): регуляризация через q
        - Энтропия состояния: связь с физикой токена
        """
        # Основной loss
        loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            targets.view(-1),
            reduction='mean'
        )
        
        # Tsallis q-энтропия от внимания (Вариант 3)
        if attention_weights is not None and q_avg is not None:
            tsallis_s = ElengalMath.tsallis_entropy(attention_weights, q_avg)
            # q < 1: tsallis_s положительная → поощряем концентрацию
            # q > 1: tsallis_s отрицательная → поощряем разнообразие
            loss = loss - self.entropy_bonus * tsallis_s
        
        # Энтропия состояния (связь с физикой)
        state_entropy = state.entropy.mean()
        loss = loss - self.entropy_bonus * 0.1 * state_entropy
        
        return loss
    
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ) -> Tuple[torch.Tensor, ElengalTokenState]:
        """Генерация текста."""
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits, state, _, _ = self.forward(prompt)
                logits = logits[:, -1, :] / temperature
                
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, float('-inf'))
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                prompt = torch.cat([prompt, next_token], dim=1)
        
        _, final_state, _, _ = self.forward(prompt)
        return prompt, final_state


# ============================================================================
# 9. УДОБНЫЕ ФУНКЦИИ
# ============================================================================

def create_elengal(
    vocab_size: int,
    dim: int = 64,
    n_layers: int = 6,
    device: str = "auto",
    dtype: str = "auto",
    **kwargs
) -> Tuple[ElengalV1, ElengalConfig]:
    """
    Удобное создание модели Elengal с автоматической настройкой.
    """
    config = ElengalConfig(
        dim=dim,
        n_layers=n_layers,
        device=device,
        dtype=dtype,
        **kwargs
    )
    
    model = ElengalV1(vocab_size, config)
    model = model.to(config.get_device())
    
    return model, config


# ============================================================================
# 10. ТЕСТИРОВАНИЕ
# ============================================================================

def test_elengal_math():
    """Тест математики Elengal."""
    
    print("\n" + "=" * 60)
    print("  ELENGAL MATH — ТЕСТ")
    print("=" * 60)
    
    x = torch.tensor([0.5, 1.0, 2.0])
    q = torch.tensor(0.5)
    
    print(f"\nq-экспонента (q={q.item()}):")
    q_exp = ElengalMath.q_exponential(x, q, terms=10)
    regular_exp = torch.exp(x)
    
    for i, xi in enumerate(x):
        print(f"   x={xi.item():.1f}: exp_q={q_exp[i].item():.4f}, exp={regular_exp[i].item():.4f}")
    
    print(f"\nq-числа:")
    for n in range(1, 6):
        q_n = ElengalMath.q_number(torch.tensor(float(n)), q)
        print(f"   [{n}]_q = {q_n.item():.4f} (обычное: {n})")
    
    print("\nOK - Тест математики завершён!")


def test_elengal_field():
    """Тест поля Elengal."""
    
    print("\n" + "=" * 60)
    print("  ELENGAL v1.0 — ТЕСТ ПОЛЯ")
    print("  Автор: Семушкин Александр Геннадьевич / Alexander Gennadyevich Semushkin")
    print("=" * 60)
    
    config = ElengalConfig(
        dim=64,
        dtype="float32",
        gravity_constant=1.0,
        magnetic_constant=0.5,
        genome_dim=32,
    )
    
    config.print_device_info()
    
    field = ElengalField(config)
    
    print(f"\nКонфигурация:")
    print(f"   Размерность: {config.dim}")
    print(f"   Геном: {config.genome_dim}")
    print(f"   Гравитация: {config.gravity_constant}")
    print(f"   Магнетизм: {config.magnetic_constant}")
    
    B, T = 2, 16
    device = config.get_device()
    
    print(f"\nРождение токенов (B={B}, T={T})...")
    state = field.birth(B, T, device)
    
    print(f"\nНачальные параметры:")
    summary = state.get_physical_summary()
    for key, value in summary.items():
        print(f"   {key}: {value:.4f}")
    
    print(f"\nЭволюция токенов...")
    context = torch.randn(B, T, config.dim, device=device, dtype=config.get_dtype())
    for i in range(10):
        state = field.interact(state, context)
        if i % 3 == 0:
            summary = state.get_physical_summary()
            print(f"   Шаг {i}: mass={summary['mass_mean']:.4f}, energy={summary['energy_mean']:.4f}")
    
    print("\nOK - Тест поля завершён!")


def test_elengal_model():
    """Тест полной модели Elengal."""
    
    print("\n" + "=" * 60)
    print("  ELENGAL v1.0 — ТЕСТ МОДЕЛИ")
    print("  Автор: Семушкин Александр Геннадьевич / Alexander Gennadyevich Semushkin")
    print("=" * 60)
    
    model, config = create_elengal(
        vocab_size=100,
        dim=32,
        genome_dim=16,
        n_layers=4,
        n_heads=2,
    )
    
    config.print_device_info()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nПараметры модели: {total_params:,}")
    
    device = config.get_device()
    x = torch.randint(0, 100, (2, 16), device=device)
    
    print(f"\nПрямой проход...")
    logits, state, attn, q_avg = model(x)
    
    print(f"   Вход: {x.shape}")
    print(f"   Выход: {logits.shape}")
    
    summary = state.get_physical_summary()
    print(f"\nФизическое состояние после прохода:")
    for key, value in summary.items():
        print(f"   {key}: {value:.4f}")
    
    print(f"\nТест генерации...")
    prompt = torch.randint(0, 100, (1, 5), device=device)
    generated, final_state = model.generate(prompt, max_new_tokens=10)
    print(f"   Prompt: {prompt.shape}")
    print(f"   Generated: {generated.shape}")
    
    print("\nOK - Модель работает!")


def test_elengal_configs():
    """Тест разных конфигураций."""
    
    print("\n" + "=" * 60)
    print("  ELENGAL — ТЕСТ КОНФИГУРАЦИЙ")
    print("=" * 60)
    
    configs = [
        ("tiny", {"dim": 4, "genome_dim": 4, "n_layers": 2}),
        ("small", {"dim": 8, "genome_dim": 8, "n_layers": 3}),
        ("medium", {"dim": 16, "genome_dim": 16, "n_layers": 4}),
        ("large", {"dim": 32, "genome_dim": 32, "n_layers": 6}),
    ]
    
    vocab_size = 50
    
    for name, params in configs:
        print(f"\nКонфигурация: {name}")
        
        model, config = create_elengal(vocab_size=vocab_size, **params)
        
        print(f"   dim={config.dim}, dtype={config._resolved_dtype}")
        
        device = config.get_device()
        x = torch.randint(0, vocab_size, (1, 8), device=device)
        
        try:
            logits, state, attn, q_avg = model(x)
            print(f"   OK! Выход: {logits.shape}")
            print(f"   Физика: mass={state.mass.mean():.4f}, energy={state.energy.mean():.4f}")
        except Exception as e:
            print(f"   Ошибка: {e}")


if __name__ == "__main__":
    print("\n")
    print("=" * 60)
    print("  ELENGAL v1.0")
    print("  Токены как физические сущности")
    print("  Автор: Семушкин Александр Геннадьевич / Alexander Gennadyevich Semushkin")
    print("=" * 60)
    print("\n")
    
    test_elengal_math()
    test_elengal_field()
    test_elengal_model()
    test_elengal_configs()
    
    print("\n")
    print("=" * 60)
    print("  ELENGAL v1.0 — Все тесты завершены!")
    print("  Автор: Семушкин Александр Геннадьевич / Alexander Gennadyevich Semushkin")
    print("=" * 60)
    print("\n")