# RUNBOOK: запуск проєкту локально

Ця інструкція допоможе запустити проєкт після завантаження з GitHub. Приклади команд наведено для macOS або Linux у терміналі `zsh` чи `bash`.

## 1. Перейдіть до папки проєкту

Відкрийте термінал і перейдіть до папки, у яку завантажили або розпакували репозиторій. Замініть шлях нижче на фактичне розташування проєкту:

```bash
cd /path/to/capstone
```

Переконайтеся, що це коренева папка проєкту:

```bash
pwd
ls
```

У списку мають бути `pipeline.py`, `requirements.txt` і папки проєкту, зокрема `datasets`.

## 2. Перевірте Python і Conda

Для створення ізольованого середовища потрібна Conda або Miniconda:

```bash
conda --version
```

Якщо команда `conda` не знайдена, встановіть Miniconda або Anaconda й відкрийте новий термінал. Якщо `conda activate` повідомляє, що оболонку не ініціалізовано, виконайте `conda init zsh` для `zsh` або `conda init bash` для `bash`, а потім відкрийте новий термінал.

## 3. Створіть середовище Python

Виконайте команду з кореневої папки проєкту:

```bash
conda create --prefix ./.venv python=3.11 pip spyder spyder-notebook nbconvert -c conda-forge
```

Conda створить середовище `.venv` у папці проєкту. Spyder і плагін Notebook потрібні лише для роботи через графічний інтерфейс; пайплайн можна запускати без Spyder.

## 4. Активуйте середовище та встановіть залежності

```bash
conda activate ./.venv
python --version
```

Встановіть бібліотеки, перелічені у `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

Переконайтеся, що активовано Python із середовища `.venv`:

```bash
which python
```

## 5. Перевірте наявність набору даних

Основний набір даних має зберігатися в папці [`datasets`](datasets/) під назвою [`data.csv`](datasets/data.csv). Пайплайн за замовчуванням завантажує саме цей файл:

Приклад структури CSV — [`example_data.csv`](datasets/example_data.csv). У ньому є лише заголовки колонок, без рядків даних; цей файл потрібен як зразок схеми й не підходить для навчання.

Якщо `data.csv` відсутній у завантаженій копії репозиторію, отримайте набір даних окремо й помістіть його в папку `datasets` під цією назвою. Або вкажіть фактичний шлях через `--source` чи в `CONFIG["common"]["source"]` у `pipeline.py`. Наприклад:

```bash
python pipeline.py --source "/path/to/your/data.csv" --stage all
```

## 6. Запустіть пайплайн

Щоб запустити всі кроки етапу 1 зі стандартними налаштуваннями:

```bash
python pipeline.py --stage all
```

Щоб вибрати модель чи стратегію порога, можна передати параметри командного рядка. Наприклад:

```bash
python pipeline.py --stage all --model catboost --threshold-strategy f1
```

Результати зберігаються в `outputs/<дата>/run_<номер>/`. Шлях до створеної папки запуску пайплайн виводить у термінал. Для налаштувань, яких немає серед параметрів командного рядка, змініть відповідні значення в `CONFIG` у `pipeline.py` перед запуском.

## 7. Запуск етапу 2

Етап 2 потребує правил фільтрації A/B для кожного тегу та заповнених результатів випадкового аудиту. До першого запуску налаштуйте `CONFIG["stage2"]["filter_rules"]` у `pipeline.py`. Для SAR-PU також задайте `CONFIG["stage2"]["sar_pu"]["propensity_features"]`.

Спочатку запустіть підготовку даних із вибраним методом, наприклад:

```bash
python pipeline.py --research-stage 2 --stage2-method nnpu --stage preparation
```

Підготовка створить вибірки аудиту та папку запуску. Запустіть перевірку аудиту для цього запуску, вказавши його шлях:

```bash
python pipeline.py --research-stage 2 --stage2-method nnpu --stage audit --run-dir "outputs/<дата>/run_<номер>"
```

Якщо результати аудиту ще не заповнені, пайплайн створить шаблони або виведе помилки й зупиниться. Заповніть файли відповідно до інструкцій у [документації пайплайна](documents/instruktsiia_pipeline_uk.md). Після цього запустіть навчання для того самого запуску:

```bash
python pipeline.py --research-stage 2 --stage2-method nnpu --stage train --run-dir "outputs/<дата>/run_<номер>"
```

Для окремої SAR-PU-гілки вкажіть `--stage2-method sar_pu` та задайте `propensity_features` для тегів у `CONFIG["stage2"]["sar_pu"]` у `pipeline.py`. Навчання зупиниться, якщо для тегу не задано propensity-ознак або бракує потрібних результатів аудиту.

Після навчання запустіть оцінювання для того самого запуску:

```bash
python pipeline.py --research-stage 2 --stage2-method nnpu --stage evaluate --run-dir "outputs/<дата>/run_<номер>"
```

Цей крок готує незалежну вибірку для оцінювання. Заповніть створені шаблони аудиту. Коли вони готові, запустіть окремий крок калібрування:

```bash
python pipeline.py --research-stage 2 --stage2-method nnpu --stage calibrate --run-dir "outputs/<дата>/run_<номер>"
```

Під час використання SAR-PU замініть `nnpu` на `sar_pu` в командах продовження того самого запуску.

## 8. Запуск через Spyder (необов'язково)

З активного середовища запустіть Spyder:

```bash
spyder
```

У Spyder відкрийте `pipeline.py`. Для запуску командного рядка можна використовувати термінал Spyder. Щоб працювати з ноутбуками `.ipynb`, відкрийте панель **View → Panes → Notebook**. Наявні файли `.py` є скриптами Python; ноутбук створюється окремим файлом `.ipynb`.

## 9. Завершіть роботу

Закрийте Spyder, якщо він запущений, і деактивуйте середовище:

```bash
conda deactivate
```

Під час наступного запуску перейдіть до папки проєкту та активуйте середовище:

```bash
cd /path/to/capstone
conda activate ./.venv
```

## Довідка

- [Документація Spyder Notebook](https://docs.spyder-ide.org/current/plugins/notebook.html)
- [Встановлення Spyder через Conda](https://docs.spyder-ide.org/current/installation.html)
- [Документація Conda](https://docs.conda.io/)
