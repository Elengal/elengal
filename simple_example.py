"""
Простой пример использования Elengal.

Демонстрирует:
- Создание модели
- Прямой проход
- Анализ физического состояния токенов
"""

import torch
from Elengal_v1 import ElengalV1, ElengalConfig, ElengalMath

def main():
    print("\n" + "=" * 60)
    print("  ELENGAL — Пример использования")
    print("=" * 60)
    
    # =========================================================================
    # 1. Создаём конфигурацию
    # =========================================================================
    
    config = ElengalConfig(
        dim=64,                    # Размерность пространства
        n_layers=4,                # Слои эволюции
        n_heads=4,                 # Головы внимания
        q_base=0.5,                # Базовое q
        gravity_constant=1.0,      # Сила гравитации
        magnetic_constant=0.5,     # Сила магнетизма
        genome_dim=32,             # Размер генома
        phase_dim=16,              # Размерность фазового пространства
    )
    
    print(f"\n Конфигурация создана:")
    print(f"   Размерность: {config.dim}")
    print(f"   Слои: {config.n_layers}")
    print(f"   q_base: {config.q_base}")
    
    # =========================================================================
    # 2. Создаём модель
    # =========================================================================
    
    vocab_size = 1000  # Размер словаря
    model = ElengalV1(vocab_size=vocab_size, config=config)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n Параметры модели: {total_params:,}")
    
    # =========================================================================
    # 3. Прямой проход
    # =========================================================================
    
    # Входная последовательность
    batch_size = 2
    seq_len = 16
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"\n Вход: {x.shape}")
    
    # Forward
    logits, state, attention_weights, q_avg = model(x)
    
    print(f" Выход (logits): {logits.shape}")
    
    # =========================================================================
    # 4. Анализ физического состояния
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("  ФИЗИЧЕСКОЕ СОСТОЯНИЕ ТОКЕНОВ")
    print("=" * 60)
    
    summary = state.get_physical_summary()
    
    print(f"\n Масса (семантическая значимость):")
    print(f"   Mean: {summary['mass_mean']:.4f}")
    print(f"   Std:  {summary['mass_std']:.4f}")
    
    print(f"\n Энергия (активность):")
    print(f"   Mean: {summary['energy_mean']:.4f}")
    print(f"   Std:  {summary['energy_std']:.4f}")
    
    print(f"\n q-параметр (избирательность):")
    print(f"   Mean: {summary['q_param_mean']:.4f}")
    print(f"   Std:  {summary['q_param_std']:.4f}")
    
    print(f"\n Магнетизм (полярность):")
    print(f"   Mean: {summary['magnetic_mean']:.4f}")
    
    # =========================================================================
    # 5. Генерация
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("  ГЕНЕРАЦИЯ ТЕКСТА")
    print("=" * 60)
    
    model.eval()
    prompt = torch.randint(0, vocab_size, (1, 5))
    
    print(f"\n Prompt shape: {prompt.shape}")
    
    with torch.no_grad():
        generated, final_state = model.generate(prompt, max_new_tokens=20, temperature=0.8)
    
    print(f" Generated shape: {generated.shape}")
    
    # =========================================================================
    # 6. Тест q-математики
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("  q-МАТЕМАТИКА")
    print("=" * 60)
    
    x_val = torch.tensor([0.5, 1.0, 2.0])
    q_val = torch.tensor(0.5)
    
    q_exp = ElengalMath.q_exponential(x_val, q_val, terms=10)
    regular_exp = torch.exp(x_val)
    
    print(f"\n q-экспонента (q={q_val.item()}):")
    for i, xi in enumerate(x_val):
        print(f"   x={xi.item():.1f}: exp_q={q_exp[i].item():.4f}, exp={regular_exp[i].item():.4f}")
    
    print("\n OK — Пример завершён успешно!")


if __name__ == "__main__":
    main()
