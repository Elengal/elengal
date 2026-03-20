"""
Обучение Elengal на стихах Лермонтова и Блока
Словесная токенизация (не BPE!)
"""

import torch
import torch.nn.functional as F
from Elengal_v1 import ElengalV1, ElengalConfig, ElengalMath

# ============================================================================
# СТИХИ
# ============================================================================

POEMS = """
У врат обители святой
Стоял просящий подаянья
Бедняк иссохший чуть живой
От глада жажды и страданья
Куска лишь хлеба он просил
И взор являл живую муку
И кто-то камень положил
В его протянутую руку
Так я молил твоей любви
С слезами горькими с тоскою
Так чувства лучшие мои
Обмануты навек тобою

Когда под заступом холодным
Скрипел песок и яркий снег
Во мне печальном и свободном
Еще смирялся человек
Пусть эта смерть была понятна
В душе под песни панихид
Уж проступали злые пятна
Незабываемых обид
Уже с угрозою сжималась
Доселе добрая рука
Уж подымалась и металась
В душе отравленной тоска
Я подавлю глухую злобу
Тоску забвению предам
Святому маленькому гробу
Молиться буду по ночам
Но быть коленопреклоненным
Тебя благодарить скорбя
Нет Над младенцем над блаженным
Скорбеть я буду без Тебя
"""

# ============================================================================
# СЛОВЕСНАЯ ТОКЕНИЗАЦИЯ
# ============================================================================

class WordTokenizer:
    """Простая словесная токенизация."""
    
    def __init__(self, text: str):
        # Нормализация: нижний регистр, разделение по словам
        words = text.lower().split()
        
        # Уникальные слова
        self.vocab = ['<PAD>', '<UNK>', '<BOS>', '<EOS>'] + sorted(set(words))
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2word = {i: w for i, w in enumerate(self.vocab)}
        
        print(f"\n📚 Словарь: {len(self.vocab)} слов")
        print(f"   Примеры: {self.vocab[4:14]}")
    
    def encode(self, text: str) -> list:
        """Текст → индексы."""
        words = text.lower().split()
        return [self.word2idx.get(w, 1) for w in words]  # 1 = <UNK>
    
    def decode(self, indices: list) -> str:
        """Индексы → текст."""
        return ' '.join(self.idx2word.get(i, '<UNK>') for i in indices)
    
    def __len__(self):
        return len(self.vocab)


# ============================================================================
# ДАТАСЕТ
# ============================================================================

class PoemDataset:
    """Простой датасет для стихов."""
    
    def __init__(self, text: str, tokenizer: WordTokenizer, seq_len: int = 16):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        # Токенизируем весь текст
        tokens = tokenizer.encode(text)
        
        # Разбиваем на последовательности
        self.sequences = []
        for i in range(0, len(tokens) - seq_len, seq_len // 2):
            seq = tokens[i:i + seq_len + 1]  # +1 для target
            if len(seq) == seq_len + 1:
                self.sequences.append(seq)
        
        print(f"\n📝 Последовательностей: {len(self.sequences)}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return torch.tensor(seq[:-1]), torch.tensor(seq[1:])


# ============================================================================
# ТЕСТ Q-ЭКСПОНЕНТЫ ПЕРЕД ОБУЧЕНИЕМ
# ============================================================================

def test_q_exp():
    """Проверка q-экспоненты."""
    print("\n" + "=" * 60)
    print("  ТЕСТ Q-ЭКСПОНЕНТЫ")
    print("=" * 60)
    
    x = torch.tensor([0.5, 1.0, 2.0])
    q = torch.tensor(0.5)
    
    q_exp = ElengalMath.q_exponential(x, q, terms=10)
    regular_exp = torch.exp(x)
    
    print(f"\nq={q.item()}:")
    for i, xi in enumerate(x):
        print(f"   x={xi.item():.1f}: exp_q={q_exp[i].item():.4f}, exp={regular_exp[i].item():.4f}")
    
    # Тест с разными q
    print(f"\nПри разных q (x=1.0):")
    for q_val in [0.3, 0.5, 0.7, 0.9, 0.95, 0.99]:
        q_exp = ElengalMath.q_exponential(torch.tensor([1.0]), torch.tensor(q_val), terms=10)
        print(f"   q={q_val:.2f}: exp_q(1)={q_exp[0].item():.4f}")


# ============================================================================
# ОБУЧЕНИЕ
# ============================================================================

def train():
    """Обучение модели."""
    
    print("\n" + "=" * 60)
    print("  ELENGAL — ОБУЧЕНИЕ НА СТИХАХ")
    print("=" * 60)
    
    # Токенизация
    tokenizer = WordTokenizer(POEMS)
    dataset = PoemDataset(POEMS, tokenizer, seq_len=16)
    
    # Конфигурация
    config = ElengalConfig(
        dim=32,
        dtype="float32",
        genome_dim=16,
        phase_dim=8,
        n_layers=4,
        n_heads=2,
        q_base=0.5,
        gravity_constant=1.0,
        magnetic_constant=0.5,
        dropout=0.1,
    )
    
    print(f"\n⚙️ Конфигурация:")
    print(f"   dim={config.dim}, layers={config.n_layers}")
    print(f"   q_base={config.q_base}")
    
    # Модель
    model = ElengalV1(vocab_size=len(tokenizer), config=config)
    
    # Считаем параметры
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Параметры: {total_params:,}")
    
    # Оптимизатор
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    # Обучение
    print(f"\n🚀 Начинаем обучение...")
    
    model.train()
    losses = []
    
    for epoch in range(50):
        total_loss = 0.0
        num_batches = 0
        q_avg_track = 0.0  # Отслеживание q
        
        for i in range(len(dataset)):
            x, y = dataset[i]
            x = x.unsqueeze(0)  # [1, seq_len]
            y = y.unsqueeze(0)
            
            # Forward — теперь 4 значения
            logits, state, attention_weights, q_avg = model(x)
            q_avg_track = q_avg.item() if isinstance(q_avg, torch.Tensor) else q_avg
            
            # Loss с Tsallis энтропией
            loss = model.compute_loss(logits, y, state, attention_weights, q_avg)
            
            # Проверка на NaN
            if torch.isnan(loss):
                print(f"\n⚠️ NaN на эпохе {epoch+1}, батч {i}!")
                print(f"   logits min/max: {logits.min().item():.4f} / {logits.max().item():.4f}")
                print(f"   q_avg: {q_avg_track:.4f}")
                break
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"   Эпоха {epoch+1:2d}: loss = {avg_loss:.4f}")
            
            # Физическое состояние + разнообразие
            summary = state.get_physical_summary()
            pv_mean = state.phase_velocity.mean().item()
            print(f"      mass={summary['mass_mean']:.4f}±{summary['mass_std']:.4f}, "
                  f"q={summary['q_param_mean']:.4f}±{summary['q_param_std']:.4f}, "
                  f"phase_vel={pv_mean:.6f}")
    
    print(f"\n✅ Обучение завершено!")
    print(f"   Начальный loss: {losses[0]:.4f}")
    print(f"   Конечный loss: {losses[-1]:.4f}")
    
    # === СОХРАНЕНИЕ МОДЕЛИ ===
    print(f"\n💾 Сохранение модели...")
    
    import os
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)
    
    # Сохраняем состояние модели
    model_path = os.path.join(save_dir, "elengal_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'dim': config.dim,
            'vocab_size': len(tokenizer),
            'n_layers': config.n_layers,
            'genome_dim': config.genome_dim,
            'phase_dim': config.phase_dim,
            'q_base': config.q_base,
        },
        'vocab': tokenizer.vocab,
        'losses': losses,
        'final_loss': losses[-1],
    }, model_path)
    print(f"   ✅ Модель сохранена: {model_path}")
    
    # Сохраняем словарь отдельно
    vocab_path = os.path.join(save_dir, "vocab.txt")
    with open(vocab_path, 'w', encoding='utf-8') as f:
        for word in tokenizer.vocab:
            f.write(word + '\n')
    print(f"   ✅ Словарь сохранён: {vocab_path}")
    
    # Генерация
    print(f"\n🎨 Генерация текста:")
    model.eval()
    
    # Старт с нескольких слов
    start_words = "у врат обители"
    start_tokens = tokenizer.encode(start_words)
    prompt = torch.tensor([start_tokens])
    
    with torch.no_grad():
        generated, final_state = model.generate(prompt, max_new_tokens=15, temperature=0.8)
    
    generated_text = tokenizer.decode(generated[0].tolist())
    print(f"   '{start_words}' →")
    print(f"   '{generated_text}'")
    
    # Финальное состояние
    summary = final_state.get_physical_summary()
    print(f"\n📊 Финальное физическое состояние:")
    for key, value in summary.items():
        print(f"   {key}: {value:.4f}")
    
    # Дополнительно: фазовая динамика
    print(f"\n🌊 Фазовая динамика:")
    print(f"   phase_velocity mean: {final_state.phase_velocity.mean().item():.6f}")
    print(f"   phase_velocity std: {final_state.phase_velocity.std().item():.6f}")
    print(f"   phase mean: {final_state.phase.mean().item():.6f}")
    print(f"   phase std: {final_state.phase.std().item():.6f}")
    
    # === СЕМАНТИЧЕСКИЙ АНАЛИЗ ===
    print(f"\n" + "=" * 60)
    print(f"  🔬 СЕМАНТИЧЕСКИЙ АНАЛИЗ")
    print("=" * 60)
    
    analyze_semantics(model, tokenizer, POEMS)
    
    return model, tokenizer, losses


# ============================================================================
# СЕМАНТИЧЕСКИЙ АНАЛИЗ
# ============================================================================

def analyze_semantics(model: ElengalV1, tokenizer: WordTokenizer, text: str):
    """
    Анализирует семантические свойства слов после обучения.
    
    Исследует:
    1. Масса слов — какие слова "тяжелые" (знаменательные), какие "лёгкие" (служебные)
    2. q по словам — избирательность внимания
    3. Фазовые траектории — кластеризация слов в фазовом пространстве
    """
    model.eval()
    
    # Токенизируем весь текст
    words = text.lower().split()
    word_to_indices = {}
    for i, word in enumerate(words):
        if word not in word_to_indices:
            word_to_indices[word] = []
        word_to_indices[word].append(i)
    
    # Создаём длинную последовательность для анализа
    all_tokens = tokenizer.encode(text)
    
    # Запускаем через модель
    with torch.no_grad():
        x = torch.tensor([all_tokens])
        logits, state, _, _ = model(x)
    
    # === 1. МАССА СЛОВ ===
    print("\n📊 1. МАССА СЛОВ (семантическая значимость)")
    print("-" * 50)
    
    word_mass = {}
    for word, indices in word_to_indices.items():
        # Средняя масса для всех вхождений слова
        masses = [state.mass[0, idx, 0].item() for idx in indices if idx < state.mass.shape[1]]
        if masses:
            word_mass[word] = sum(masses) / len(masses)
    
    # Сортируем по массе
    sorted_mass = sorted(word_mass.items(), key=lambda x: x[1], reverse=True)
    
    print("\n🔴 ТЯЖЁЛЫЕ слова (высокая значимость):")
    for word, mass in sorted_mass[:10]:
        print(f"   {word:<20} mass = {mass:.4f}")
    
    print("\n🟡 СРЕДНИЕ слова:")
    mid_start = len(sorted_mass) // 3
    for word, mass in sorted_mass[mid_start:mid_start+5]:
        print(f"   {word:<20} mass = {mass:.4f}")
    
    print("\n🟢 ЛЁГКИЕ слова (служебные):")
    for word, mass in sorted_mass[-10:]:
        print(f"   {word:<20} mass = {mass:.4f}")
    
    # === 2. q ПО СЛОВАМ ===
    print("\n" + "-" * 50)
    print("🎯 2. q-ПАРАМЕТР СЛОВ (избирательность внимания)")
    print("-" * 50)
    
    word_q = {}
    for word, indices in word_to_indices.items():
        q_values = [state.q_param[0, idx, 0].item() for idx in indices if idx < state.q_param.shape[1]]
        if q_values:
            word_q[word] = sum(q_values) / len(q_values)
    
    sorted_q = sorted(word_q.items(), key=lambda x: x[1])
    
    print("\n🎯 ИЗБИРАТЕЛЬНЫЕ слова (q < 0.6, концентрированное внимание):")
    selective_words = [(w, q) for w, q in sorted_q if q < 0.6]
    for word, q in selective_words[:10]:
        print(f"   {word:<20} q = {q:.4f}")
    
    print("\n⚖️ СБАЛАНСИРОВАННЫЕ слова (0.6 < q < 0.7):")
    balanced_words = [(w, q) for w, q in sorted_q if 0.6 <= q <= 0.7]
    for word, q in balanced_words[:8]:
        print(f"   {word:<20} q = {q:.4f}")
    
    print("\n🌐 РАЗМЫТЫЕ слова (q > 0.7, широкое внимание):")
    diffuse_words = [(w, q) for w, q in sorted_q if q > 0.7]
    for word, q in diffuse_words[:10]:
        print(f"   {word:<20} q = {q:.4f}")
    
    # === 3. ФАЗОВЫЕ ТРАЕКТОРИИ ===
    print("\n" + "-" * 50)
    print("🌀 3. ФАЗОВЫЕ ТРАЕКТОРИИ")
    print("-" * 50)
    
    # Собираем фазы для каждого слова
    word_phases = {}
    for word, indices in word_to_indices.items():
        phases = [state.phase[0, idx, :].numpy() for idx in indices if idx < state.phase.shape[1]]
        if phases:
            word_phases[word] = phases
    
    # PCA для визуализации
    try:
        from sklearn.decomposition import PCA
        import numpy as np
        
        # Собираем все фазы
        all_phases = []
        all_labels = []
        all_positions = []
        
        for i, (word, phases) in enumerate(word_phases.items()):
            for phase in phases:
                all_phases.append(phase)
                all_labels.append(word)
                all_positions.append(i)
        
        if len(all_phases) > 2:
            all_phases = np.array(all_phases)
            
            # PCA до 2D
            pca = PCA(n_components=2)
            phases_2d = pca.fit_transform(all_phases)
            
            print(f"\n   PCA объяснённая дисперсия: {pca.explained_variance_ratio_.sum()*100:.1f}%")
            
            # Группируем по словам для анализа
            word_positions_2d = {}
            for i, (label, pos_2d) in enumerate(zip(all_labels, phases_2d)):
                if label not in word_positions_2d:
                    word_positions_2d[label] = []
                word_positions_2d[label].append(pos_2d)
            
            # Находим слова с наибольшим разбросом (интересные)
            word_spread = {}
            for word, positions in word_positions_2d.items():
                positions = np.array(positions)
                spread = np.std(positions[:, 0]) + np.std(positions[:, 1])
                word_spread[word] = spread
            
            sorted_spread = sorted(word_spread.items(), key=lambda x: x[1], reverse=True)
            
            print("\n   📍 Слова с наибольшим фазовым разбросом (многозначные?):")
            for word, spread in sorted_spread[:8]:
                print(f"      {word:<20} spread = {spread:.4f}")
            
            print("\n   📍 Слова с наименьшим фазовым разбросом (стабильные):")
            for word, spread in sorted_spread[-5:]:
                print(f"      {word:<20} spread = {spread:.4f}")
            
            # Сохраняем данные для визуализации
            print("\n   💾 Сохраняю данные для визуализации...")
            save_visualization_data(word_phases, word_mass, word_q, phases_2d, all_labels)
            
    except ImportError:
        print("   ⚠️ sklearn не установлен — PCA пропущен")
        print("   Установите: pip install scikit-learn")
    
    # === ИТОГОВАЯ СТАТИСТИКА ===
    print("\n" + "=" * 50)
    print(" 📈 ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 50)
    
    masses = list(word_mass.values())
    q_values = list(word_q.values())
    
    print(f"\n   Масса: min={min(masses):.4f}, max={max(masses):.4f}, mean={sum(masses)/len(masses):.4f}")
    print(f"   q:     min={min(q_values):.4f}, max={max(q_values):.4f}, mean={sum(q_values)/len(q_values):.4f}")
    
    # Корреляция масса-q
    if len(word_mass) > 1:
        mass_list = [word_mass[w] for w in word_q.keys()]
        q_list = [word_q[w] for w in word_q.keys()]
        
        mean_mass = sum(mass_list) / len(mass_list)
        mean_q = sum(q_list) / len(q_list)
        
        cov = sum((m - mean_mass) * (q - mean_q) for m, q in zip(mass_list, q_list))
        std_mass = (sum((m - mean_mass)**2 for m in mass_list) ** 0.5)
        std_q = (sum((q - mean_q)**2 for q in q_list) ** 0.5)
        
        correlation = cov / (std_mass * std_q) if std_mass > 0 and std_q > 0 else 0
        
        print(f"\n   Корреляция масса ↔ q: {correlation:.4f}")
        if correlation < -0.3:
            print("      → Тяжёлые слова более избирательны (низкое q)")
        elif correlation > 0.3:
            print("      → Тяжёлые слова более размазаны (высокое q)")
        else:
            print("      → Связи между массой и q не обнаружено")


def save_visualization_data(word_phases, word_mass, word_q, phases_2d, all_labels):
    """Сохраняет данные для внешней визуализации."""
    import json
    import os
    
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)
    
    # Подготовка данных
    viz_data = {
        "words": {},
        "phases_2d": phases_2d.tolist(),
        "labels": all_labels
    }
    
    for word in word_phases.keys():
        viz_data["words"][word] = {
            "mass": word_mass.get(word, 0),
            "q": word_q.get(word, 0),
            "phase_samples": len(word_phases[word])
        }
    
    viz_path = os.path.join(save_dir, "semantic_analysis.json")
    with open(viz_path, 'w', encoding='utf-8') as f:
        json.dump(viz_data, f, ensure_ascii=False, indent=2)
    
    print(f"   ✅ Сохранено: {viz_path}")


# ============================================================================
# ЗАГРУЗКА МОДЕЛИ
# ============================================================================

def load_elengal(model_path: str) -> Tuple[ElengalV1, 'WordTokenizer', dict]:
    """
    Загружает сохранённую модель Elengal.
    
    Args:
        model_path: путь к файлу модели (elengal_model.pt)
    
    Returns:
        model: загруженная модель
        tokenizer: токенизатор
        checkpoint: словарь с метаданными (losses, config и т.д.)
    """
    print(f"\n📂 Загрузка модели из {model_path}...")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Восстанавливаем конфигурацию
    saved_config = checkpoint['config']
    config = ElengalConfig(
        dim=saved_config['dim'],
        n_layers=saved_config['n_layers'],
        genome_dim=saved_config['genome_dim'],
        phase_dim=saved_config['phase_dim'],
        q_base=saved_config['q_base'],
    )
    
    # Создаём модель
    model = ElengalV1(vocab_size=saved_config['vocab_size'], config=config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Восстанавливаем токенизатор
    vocab = checkpoint['vocab']
    tokenizer = WordTokenizer.__new__(WordTokenizer)
    tokenizer.vocab = vocab
    tokenizer.word2idx = {w: i for i, w in enumerate(vocab)}
    tokenizer.idx2word = {i: w for i, w in enumerate(vocab)}
    
    print(f"   ✅ Модель загружена")
    print(f"   Vocab size: {len(vocab)}")
    print(f"   Final loss: {checkpoint.get('final_loss', 'N/A')}")
    
    return model, tokenizer, checkpoint


def generate_with_saved_model(model_path: str, prompt: str, max_tokens: int = 30):
    """
    Генерирует текст с сохранённой модели.
    
    Args:
        model_path: путь к модели
        prompt: начальный текст
        max_tokens: количество токенов для генерации
    """
    model, tokenizer, _ = load_elengal(model_path)
    model.eval()
    
    # Кодируем промпт
    tokens = tokenizer.encode(prompt)
    prompt_tensor = torch.tensor([tokens])
    
    # Генерируем
    with torch.no_grad():
        generated, state = model.generate(prompt_tensor, max_new_tokens=max_tokens, temperature=0.8)
    
    # Декодируем
    generated_text = tokenizer.decode(generated[0].tolist())
    
    print(f"\n🎨 Генерация:")
    print(f"   Промпт: '{prompt}'")
    print(f"   Результат: '{generated_text}'")
    
    # Физическое состояние
    summary = state.get_physical_summary()
    print(f"\n📊 Физическое состояние:")
    print(f"   q_param: {summary['q_param_mean']:.4f}")
    print(f"   mass: {summary['mass_mean']:.4f}")
    print(f"   energy: {summary['energy_mean']:.4f}")
    
    return generated_text


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  ELENGAL v1.0 — ТЕСТ ОБУЧЕНИЯ")
    print("  Словесная токенизация")
    print("=" * 60)
    
    # Сначала тест q-экспоненты
    test_q_exp()
    
    # Обучение
    model, tokenizer, losses = train()
    
    print("\n" + "=" * 60)
    print("  ТЕСТ ЗАВЕРШЁН")
    print("=" * 60)
