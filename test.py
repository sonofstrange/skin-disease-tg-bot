from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from telegram.error import TimedOut
from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime, timedelta
import json
import os
import io

# Для графиков
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

TOKEN = "YOUR_TOKEN_HERE"
img_count = 0

# Файлы для хранения данных
AGREED_USERS_FILE = "agreed_users.json"
DIAGNOSES_FILE = "diagnoses.json"
STATS_FILE = "stats.json"
ADMIN_SESSIONS_FILE = "admin_sessions.json"

# Админ креденшалы
ADMIN_LOGIN = "adminlogin"
ADMIN_PASSWORD = "admin123"

# Загружаем список согласившихся пользователей
agreed_users = set()
if os.path.exists(AGREED_USERS_FILE):
    try:
        with open(AGREED_USERS_FILE, 'r') as f:
            agreed_users = set(json.load(f))
    except:
        agreed_users = set()

# Загружаем админ сессии
admin_sessions = set()
if os.path.exists(ADMIN_SESSIONS_FILE):
    try:
        with open(ADMIN_SESSIONS_FILE, 'r') as f:
            admin_sessions = set(json.load(f))
    except:
        admin_sessions = set()

# Загружаем статистику
stats_data = {"requests": [], "photos": []}
if os.path.exists(STATS_FILE):
    try:
        with open(STATS_FILE, 'r') as f:
            stats_data = json.load(f)
    except:
        stats_data = {"requests": [], "photos": []}


def save_admin_sessions():
    with open(ADMIN_SESSIONS_FILE, 'w') as f:
        json.dump(list(admin_sessions), f)


def save_stats():
    with open(STATS_FILE, 'w', encoding='utf-8') as f:
        json.dump(stats_data, f, ensure_ascii=False, indent=2)


def log_request(user_id: int, request_type: str):
    stats_data["requests"].append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_id": user_id,
        "type": request_type
    })
    save_stats()


def log_photo(filename: str, user_id: int, ai_prediction: str, ai_confidence: float):
    stats_data["photos"].append({
        "filename": filename,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_id": user_id,
        "ai_prediction": ai_prediction,
        "ai_confidence": ai_confidence,
        "user_verdict": None
    })
    save_stats()


def update_photo_verdict(filename: str, user_verdict: str):
    for photo in stats_data["photos"]:
        if photo["filename"] == filename or photo["filename"].endswith(filename):
            photo["user_verdict"] = user_verdict
            save_stats()
            break


DIAGNOSES_MAP = {
    "panu": "Разноцветный лишай",
    "rosacea": "Розацеа",
    "herpes": "Герпес",
    "eksim": "Экзема",
    "acne": "Акне",
    "skin": "Здоровая кожа",
    "other": "Больше болезней"
}

POPULAR_DIAGNOSES = {
    "dermatitis": "Дерматит",
    "chickenpox": "Ветрянка",
    "scabies": "Чесотка",
    "папиллома": "Папиллома",
    "psoriasis": "Псориаз",
    "papillomas_and_warts": "Папилломы и бородавки",
    "burns": "Ожог",
    "allergy": "Аллергическая реакция",
    "other": "Болезни нет в списке"
}

ALL_DIAGNOSES = {**DIAGNOSES_MAP, **POPULAR_DIAGNOSES}

USELESS_TEXT = """ПОЛЬЗОВАТЕЛЬСКОЕ СОГЛАШЕНИЕ
для сервиса предварительной оценки состояния кожи через Telegram-бота

1. Общие положения
1.1. Настоящее Пользовательское соглашение регулирует отношения между Русановым Николаем Алексеевичем (далее — «Разработчик») и пользователем (далее — «Пользователь») телеграм-бота «Поиск кожный заболеваний» (далее — «Сервис»).
1.2. Используя Сервис, Пользователь подтверждает, что ознакомился и согласен со всеми условиями настоящего Соглашения.

2. Характер Сервиса
2.1. Сервис предоставляет исключительно предварительную, ознакомительную информацию на основе алгоритмов машинного обучения и не является медицинским сервисом, средством диагностики, лечения или заменой консультации врача.
2.2. Разработчик не несет ответственности за решения, принятые Пользователем на основе информации, полученной от Сервиса. Окончательный диагноз может поставить только врач-дерматолог при очном осмотре.

3. Обработка пользовательских данных
3.1. Отправляя фотографию через Сервис, Пользователь добровольно и осознанно дает согласие на обработку и хранение этой фотографии.
3.2. Цели обработки:
* Анализ изображения с помощью нейросетевой модели для классификации.
* Улучшение работы алгоритма (обучение модели) на обезличенных данных (см. п. 3.4).
* Техническое хранение для обеспечения работоспособности Сервиса.
3.3. Фотографии хранятся на защищенном сервере, доступ к которому имеет только Разработчик.
3.4. Для целей улучшения алгоритма (дообучения модели) с фотографий удаляются все метаданные и любые признаки, позволяющие идентифицировать личность Пользователя. Таким образом, для обучения используется обезличенный набор данных.

4. Права Пользователя
4.1. Пользователь имеет право отозвать согласие на обработку своих фотографий, направив запрос на электронную почту Разработчика. После получения запроса фотографии Пользователя будут удалены в течение 30 (тридцати) календарных дней.

5. Возрастное ограничение
5.1. Сервисом могут пользоваться лица, достигшие 16 лет. Если Пользователю меньше 16 лет, он должен получить разрешение от родителя (законного представителя), который соглашается с условиями настоящего Соглашения.

6. Прочие условия
6.1. Разработчик вправе вносить изменения в настоящее Соглашение. Актуальная версия всегда доступена в описании Telegram-бота.
6.2. Продолжая использование Сервиса после внесения изменений, Пользователь подтверждает свое согласие с новой редакцией Соглашения.

7. Контактная информация
7.1. По всем вопросам, связанным с обработкой данных, Пользователь может обращаться к Разработчику:
Русанов Николай Алексеевич
Электронная почта: strange.z2tablet@gmail.com"""

if not os.path.exists(DIAGNOSES_FILE):
    with open(DIAGNOSES_FILE, 'w', encoding='utf-8') as f:
        json.dump([], f)


def save_agreed_user(user_id):
    agreed_users.add(user_id)
    with open(AGREED_USERS_FILE, 'w') as f:
        json.dump(list(agreed_users), f)


# ===================== АДМИН-ПАНЕЛЬ =====================

def is_admin(user_id: int) -> bool:
    return user_id in admin_sessions


def get_admin_menu_keyboard():
    """Возвращает клавиатуру главного меню админки"""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Статистика за сегодня", callback_data="admin::stats::today")],
        [InlineKeyboardButton("Статистика за неделю", callback_data="admin::stats::week")],
        [InlineKeyboardButton("Статистика за месяц", callback_data="admin::stats::month")],
        [InlineKeyboardButton("Последние 5 фото", callback_data="admin::photos::5::all")],
        [InlineKeyboardButton("Последние 10 фото", callback_data="admin::photos::10::all")],
        [InlineKeyboardButton("Фото по статусу", callback_data="admin::photos_filter")],
        [InlineKeyboardButton("Фото за период", callback_data="admin::photos_period")],
        [InlineKeyboardButton("Общая статистика", callback_data="admin::general")],
        [InlineKeyboardButton("Выйти из админки", callback_data="admin::logout")]
    ])


def get_status_filter_keyboard():
    """Клавиатура выбора фильтра по статусу"""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Все фото", callback_data="admin::photos::10::all")],
        [InlineKeyboardButton("Подтвержденные", callback_data="admin::photos::10::confirmed")],
        [InlineKeyboardButton("Отклоненные", callback_data="admin::photos::10::rejected")],
        [InlineKeyboardButton("Без отзыва", callback_data="admin::photos::10::pending")],
        [InlineKeyboardButton("<< Назад", callback_data="admin::menu")]
    ])


def generate_daily_stats_chart(days: int = 7) -> io.BytesIO:
    plt.figure(figsize=(12, 6))

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    daily_counts = Counter()
    daily_photos = Counter()

    for req in stats_data["requests"]:
        try:
            req_date = datetime.strptime(req["timestamp"], "%Y-%m-%d %H:%M:%S").date()
            if start_date.date() <= req_date <= end_date.date():
                daily_counts[req_date] += 1
        except:
            continue

    for photo in stats_data["photos"]:
        try:
            photo_date = datetime.strptime(photo["timestamp"], "%Y-%m-%d %H:%M:%S").date()
            if start_date.date() <= photo_date <= end_date.date():
                daily_photos[photo_date] += 1
        except:
            continue

    all_dates = []
    current = start_date
    while current <= end_date:
        all_dates.append(current.date())
        current += timedelta(days=1)

    requests_values = [daily_counts.get(d, 0) for d in all_dates]
    photos_values = [daily_photos.get(d, 0) for d in all_dates]

    x = range(len(all_dates))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar([i - width / 2 for i in x], requests_values, width, label='Vse zaprosy', color='#3498db')
    bars2 = ax.bar([i + width / 2 for i in x], photos_values, width, label='Fotografii', color='#e74c3c')

    ax.set_xlabel('Data')
    ax.set_ylabel('Kolichestvo')
    ax.set_title(f'Statistika za poslednie {days} dnej')
    ax.set_xticks(x)
    ax.set_xticklabels([d.strftime('%d.%m') for d in all_dates], rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    plt.close()

    return buf


def generate_diagnoses_pie_chart() -> io.BytesIO:
    plt.figure(figsize=(10, 8))

    diagnoses_count = Counter()
    user_verdicts_count = Counter()

    for photo in stats_data["photos"]:
        ai_pred = photo.get("ai_prediction", "unknown")
        diagnoses_count[ai_pred] += 1

        user_verdict = photo.get("user_verdict")
        if user_verdict:
            user_verdicts_count[user_verdict] += 1

    if not diagnoses_count:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Net dannyh', ha='center', va='center', fontsize=20)
        ax.axis('off')
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

        labels1 = [ALL_DIAGNOSES.get(k, k) for k in diagnoses_count.keys()]
        sizes1 = list(diagnoses_count.values())
        colors1 = plt.cm.Set3(np.linspace(0, 1, len(labels1)))

        ax1.pie(sizes1, labels=labels1, autopct='%1.1f%%', colors=colors1, startangle=90)
        ax1.set_title('Predskazaniya II')

        if user_verdicts_count:
            labels2 = [ALL_DIAGNOSES.get(k, k) for k in user_verdicts_count.keys()]
            sizes2 = list(user_verdicts_count.values())
            colors2 = plt.cm.Set2(np.linspace(0, 1, len(labels2)))

            ax2.pie(sizes2, labels=labels2, autopct='%1.1f%%', colors=colors2, startangle=90)
            ax2.set_title('Verdikty polzovatelej')
        else:
            ax2.text(0.5, 0.5, 'Net verdiktov', ha='center', va='center', fontsize=14)
            ax2.axis('off')

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    plt.close()

    return buf


def generate_hourly_chart() -> io.BytesIO:
    plt.figure(figsize=(12, 6))

    hourly_counts = Counter()

    for req in stats_data["requests"]:
        try:
            req_hour = datetime.strptime(req["timestamp"], "%Y-%m-%d %H:%M:%S").hour
            hourly_counts[req_hour] += 1
        except:
            continue

    hours = list(range(24))
    counts = [hourly_counts.get(h, 0) for h in hours]

    plt.bar(hours, counts, color='#9b59b6', edgecolor='black')
    plt.xlabel('Chas dnya')
    plt.ylabel('Kolichestvo zaprosov')
    plt.title('Aktivnost po chasam (za vse vremya)')
    plt.xticks(hours)
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    plt.close()

    return buf


def generate_accuracy_chart() -> io.BytesIO:
    plt.figure(figsize=(10, 6))

    confirmed = 0
    rejected = 0
    no_feedback = 0

    for photo in stats_data["photos"]:
        user_verdict = photo.get("user_verdict")
        ai_prediction = photo.get("ai_prediction")

        if user_verdict is None:
            no_feedback += 1
        elif user_verdict == ai_prediction:
            confirmed += 1
        else:
            rejected += 1

    labels = ['Podtverzhdeno', 'Otkloneno', 'Bez otzyva']
    sizes = [confirmed, rejected, no_feedback]
    colors = ['#2ecc71', '#e74c3c', '#95a5a6']
    explode = (0.05, 0.05, 0)

    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    plt.title('Tochnost predskazanij II')

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    plt.close()

    return buf


def get_photos_filtered(start_date: datetime = None, end_date: datetime = None,
                        limit: int = None, status_filter: str = "all") -> list:
    """Получает фотографии с фильтрами"""
    photos = stats_data["photos"].copy()

    # Фильтр по дате
    if start_date and end_date:
        filtered = []
        for photo in photos:
            try:
                photo_date = datetime.strptime(photo["timestamp"], "%Y-%m-%d %H:%M:%S")
                if start_date <= photo_date <= end_date:
                    filtered.append(photo)
            except:
                continue
        photos = filtered

    # Фильтр по статусу
    if status_filter != "all":
        filtered = []
        for photo in photos:
            user_verdict = photo.get("user_verdict")
            ai_prediction = photo.get("ai_prediction")

            if status_filter == "confirmed":
                if user_verdict and user_verdict == ai_prediction:
                    filtered.append(photo)
            elif status_filter == "rejected":
                if user_verdict and user_verdict != ai_prediction:
                    filtered.append(photo)
            elif status_filter == "pending":
                if user_verdict is None:
                    filtered.append(photo)
        photos = filtered

    # Сортируем по дате (новые первые)
    photos.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    if limit:
        photos = photos[:limit]

    return photos


async def send_photo_with_info(context: ContextTypes.DEFAULT_TYPE, chat_id: int, photo_data: dict):
    """Отправляет фото с информацией о диагнозе"""
    filename = photo_data.get("filename", "")
    if filename.startswith("images_from_bot/"):
        filepath = filename
    else:
        filepath = f"images_from_bot/{filename}"

    ai_prediction = photo_data.get("ai_prediction", "Неизвестно")
    ai_confidence = photo_data.get("ai_confidence", 0)
    user_verdict = photo_data.get("user_verdict")
    timestamp = photo_data.get("timestamp", "Неизвестно")
    user_id = photo_data.get("user_id", "Неизвестно")

    ai_diagnosis_name = ALL_DIAGNOSES.get(ai_prediction, ai_prediction)
    user_diagnosis_name = ALL_DIAGNOSES.get(user_verdict, user_verdict) if user_verdict else "Не оставлен"

    # Определяем статус
    if user_verdict is None:
        status = "Ожидает отзыва"
    elif user_verdict == ai_prediction:
        status = "Подтверждено"
    else:
        status = "Отклонено"

    caption = (
        f"Дата: {timestamp}\n"
        f"User ID: {user_id}\n\n"
        f"Вердикт ИИ: {ai_diagnosis_name}\n"
        f"Уверенность: {ai_confidence:.1%}\n\n"
        f"Вердикт пользователя: {user_diagnosis_name}\n"
        f"Статус: {status}"
    )

    if os.path.exists(filepath):
        try:
            with open(filepath, 'rb') as photo_file:
                await context.bot.send_photo(
                    chat_id=chat_id,
                    photo=photo_file,
                    caption=caption
                )
        except Exception as e:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"Ошибка отправки фото: {filename}\n{caption}\n\nОшибка: {e}"
            )
    else:
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"Файл не найден: {filename}\n\n{caption}"
        )


async def admin_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик кнопок админ-панели"""
    query = update.callback_query
    user_id = query.from_user.id

    if not is_admin(user_id):
        await query.answer("Доступ запрещен!", show_alert=True)
        return

    await query.answer()

    data = query.data
    parts = data.split("::")

    if len(parts) < 2:
        return

    action = parts[1]

    if action == "logout":
        admin_sessions.discard(user_id)
        save_admin_sessions()
        await query.edit_message_text("Вы вышли из админ-панели.")
        return

    if action == "menu":
        await query.edit_message_text(
            "АДМИН-ПАНЕЛЬ\n\nВыберите действие:",
            reply_markup=get_admin_menu_keyboard()
        )
        return

    if action == "photos_filter":
        await query.edit_message_text(
            "Выберите фильтр по статусу:",
            reply_markup=get_status_filter_keyboard()
        )
        return

    if action == "photos_period":
        context.user_data['admin_waiting_period'] = True
        await query.edit_message_text(
            "Введите период в формате:\n"
            "ДД.ММ.ГГГГ ДД.ММ.ГГГГ\n\n"
            "Например: 01.01.2024 31.01.2024\n\n"
            "Или введите 'отмена' для возврата в меню."
        )
        return

    if action == "stats":
        period = parts[2] if len(parts) > 2 else "week"

        if period == "today":
            days = 1
        elif period == "week":
            days = 7
        elif period == "month":
            days = 30
        else:
            days = 7

        await query.edit_message_text("Генерация статистики...")

        try:
            daily_chart = generate_daily_stats_chart(days)
            await context.bot.send_photo(
                chat_id=query.message.chat_id,
                photo=daily_chart,
                caption=f"Запросы за последние {days} дней"
            )

            diagnoses_chart = generate_diagnoses_pie_chart()
            await context.bot.send_photo(
                chat_id=query.message.chat_id,
                photo=diagnoses_chart,
                caption="Распределение диагнозов"
            )

            hourly_chart = generate_hourly_chart()
            await context.bot.send_photo(
                chat_id=query.message.chat_id,
                photo=hourly_chart,
                caption="Активность по часам"
            )

            accuracy_chart = generate_accuracy_chart()
            await context.bot.send_photo(
                chat_id=query.message.chat_id,
                photo=accuracy_chart,
                caption="Точность предсказаний"
            )

            keyboard = [[InlineKeyboardButton("<< Назад в меню", callback_data="admin::menu")]]
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text="Статистика сгенерирована!",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
        except Exception as e:
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text=f"Ошибка генерации статистики: {e}"
            )

    elif action == "photos":
        # Формат: admin::photos::limit::status_filter
        limit_str = parts[2] if len(parts) > 2 else "5"
        status_filter = parts[3] if len(parts) > 3 else "all"

        try:
            limit = int(limit_str)
        except:
            limit = 5

        status_names = {
            "all": "все",
            "confirmed": "подтвержденные",
            "rejected": "отклоненные",
            "pending": "без отзыва"
        }

        await query.edit_message_text(
            f"Загрузка фотографий (фильтр: {status_names.get(status_filter, status_filter)})...")

        photos = get_photos_filtered(limit=limit, status_filter=status_filter)

        if not photos:
            keyboard = [[InlineKeyboardButton("<< Назад в меню", callback_data="admin::menu")]]
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text="Фотографии не найдены.",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            return

        for photo_data in photos:
            await send_photo_with_info(context, query.message.chat_id, photo_data)

        # Кнопки для смены фильтра и навигации
        keyboard = [
            [
                InlineKeyboardButton("Все", callback_data=f"admin::photos::{limit}::all"),
                InlineKeyboardButton("Подтв.", callback_data=f"admin::photos::{limit}::confirmed"),
            ],
            [
                InlineKeyboardButton("Откл.", callback_data=f"admin::photos::{limit}::rejected"),
                InlineKeyboardButton("Ожидает", callback_data=f"admin::photos::{limit}::pending"),
            ],
            [
                InlineKeyboardButton("5 фото", callback_data=f"admin::photos::5::{status_filter}"),
                InlineKeyboardButton("10 фото", callback_data=f"admin::photos::10::{status_filter}"),
                InlineKeyboardButton("20 фото", callback_data=f"admin::photos::20::{status_filter}"),
            ],
            [InlineKeyboardButton("<< Назад в меню", callback_data="admin::menu")]
        ]
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=f"Показано {len(photos)} фото (фильтр: {status_names.get(status_filter, status_filter)})",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    elif action == "general":
        total_users = len(agreed_users)
        total_requests = len(stats_data["requests"])
        total_photos = len(stats_data["photos"])

        confirmed = sum(1 for p in stats_data["photos"]
                        if p.get("user_verdict") == p.get("ai_prediction") and p.get("user_verdict"))
        rejected = sum(1 for p in stats_data["photos"]
                       if p.get("user_verdict") and p.get("user_verdict") != p.get("ai_prediction"))
        no_feedback = sum(1 for p in stats_data["photos"] if not p.get("user_verdict"))

        accuracy = (confirmed / (confirmed + rejected) * 100) if (confirmed + rejected) > 0 else 0

        today = datetime.now().date()
        today_requests = sum(1 for r in stats_data["requests"]
                             if r.get("timestamp", "").startswith(today.strftime("%Y-%m-%d")))
        today_photos = sum(1 for p in stats_data["photos"]
                           if p.get("timestamp", "").startswith(today.strftime("%Y-%m-%d")))

        stats_text = (
            "ОБЩАЯ СТАТИСТИКА\n\n"
            f"Всего пользователей: {total_users}\n"
            f"Всего запросов: {total_requests}\n"
            f"Всего фотографий: {total_photos}\n\n"
            f"СЕГОДНЯ:\n"
            f"  Запросов: {today_requests}\n"
            f"  Фотографий: {today_photos}\n\n"
            f"ТОЧНОСТЬ ИИ:\n"
            f"  Подтверждено: {confirmed}\n"
            f"  Отклонено: {rejected}\n"
            f"  Без отзыва: {no_feedback}\n"
            f"  Точность: {accuracy:.1f}%"
        )

        keyboard = [[InlineKeyboardButton("<< Назад в меню", callback_data="admin::menu")]]
        await query.edit_message_text(
            stats_text,
            reply_markup=InlineKeyboardMarkup(keyboard)
        )


# ===================== ОСНОВНЫЕ ОБРАБОТЧИКИ =====================

async def send_agreement(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("Согласен", callback_data="agree")],
        [InlineKeyboardButton("Не согласен", callback_data="disagree")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    agreement_text = "Пожалуйста, ознакомьтесь с пользовательским соглашением перед использованием бота:\n\n" + USELESS_TEXT

    await update.message.reply_text(
        agreement_text,
        reply_markup=reply_markup,
        parse_mode=None
    )


async def agreement_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id

    if query.data == "agree":
        save_agreed_user(user_id)
        await query.edit_message_text(
            "Спасибо за согласие! Теперь вы можете пользоваться всеми функциями бота.",
            reply_markup=None
        )
        welcome_text = (
            "Привет!\n\n"
            "Я бот, который анализирует изображения и находит на них кожные заболевания. Когда ты отправишь картинку, "
            "я мгновенно скажу тебе, есть ли там заболевания, или кожа здорова.\n\n"
            "Используй /help, чтобы узнать больше о доступных командах.\n"
            "Используй /photo, чтобы узнать, как правильно делать фото."
        )
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=welcome_text
        )
    elif query.data == "disagree":
        await query.edit_message_text(
            "Без согласия с пользовательским соглашением вы не сможете пользоваться ботом.\n"
            "Если вы передумаете, нажмите /start для повторного просмотра соглашения.",
            reply_markup=None
        )


async def check_agreement(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    user_id = update.effective_user.id
    if user_id not in agreed_users:
        await send_agreement(update, context)
        return False
    return True


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("Получил /start.")
    user_id = update.effective_user.id
    log_request(user_id, "start")

    if user_id in agreed_users:
        welcome_text = (
            "Привет!\n\n"
            "Я бот, который анализирует изображения и находит на них кожные заболевания. Когда ты отправишь картинку, "
            "я мгновенно скажу тебе, есть ли там заболевания, или кожа здорова.\n\n"
            "Используй /help, чтобы узнать больше о доступных командах.\n"
            "Используй /photo, чтобы узнать, как правильно делать фото."
        )
        await update.message.reply_text(welcome_text)
    else:
        await send_agreement(update, context)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("Получил /help.")
    user_id = update.effective_user.id
    log_request(user_id, "help")

    if not await check_agreement(update, context):
        return

    help_text = (
        "*Справка по Боту*\n\n"
        "Доступные команды:\n"
        "/start - Начать работу с ботом и увидеть приветствие.\n"
        "/help - Показать это справочное сообщение.\n"
        "/photo - Справка о том, как правильно делать фото.\n\n"
        "*Как пользоваться:*\n"
        "1. Сначала отправь мне изображение кожи (Подробнее /photo).\n"
        "2. Затем проанализирую и отправлю тебе информацию о заболевании.\n\n"
        f"Принимается большинство стандартных форматов изображений.\nВремя обработки не более {5 + img_count // 10} секунд."
    )
    await update.message.reply_text(help_text, parse_mode='Markdown')


async def photo_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("Получил /photo.")
    user_id = update.effective_user.id
    log_request(user_id, "photo_help")

    if not await check_agreement(update, context):
        return

    photo_text = (
        "Для получения наиболее точных результатов при отправке фотографии следуйте следующим правилам:\n"
        " 1. Участок кожи, который вызывает у вас подозрения, должен находиться в центре кадра.\n"
        " 2. Расположите камеру на расстоянии 20–30 см от кожи.\n"
        " 3. Фотография должна быть чёткой и сделана при хорошем, естественном освещении.\n\n"
        "После отправки снимок будет автоматически обработан, и искусственный интеллект предоставит предварительный анализ."
    )
    await update.message.reply_text(photo_text)


def save_diagnosis(filename, diagnosis, confirmed_by_model=True):
    try:
        with open(DIAGNOSES_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)

        diagnosis_entry = {
            "filename": filename,
            "diagnosis": diagnosis,
            "confirmed_by_model": confirmed_by_model
        }

        data.append(diagnosis_entry)

        with open(DIAGNOSES_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Диагноз сохранен: {filename} - {diagnosis}")

        # Обновляем статистику
        base_filename = os.path.basename(filename)
        update_photo_verdict(base_filename, diagnosis)

    except Exception as e:
        print(f"Ошибка при сохранении диагноза: {e}")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик текстовых сообщений"""
    text = update.message.text.strip()
    user_id = update.effective_user.id

    print(f"Получен текст от {user_id}: {text}")

    # Проверяем команду входа в админку
    if text == f"{ADMIN_LOGIN} {ADMIN_PASSWORD}":
        admin_sessions.add(user_id)
        save_admin_sessions()

        try:
            await update.message.delete()
        except:
            pass

        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Вы вошли в админ-панель!\n\nВыберите действие:",
            reply_markup=get_admin_menu_keyboard()
        )
        return

    # Проверяем, ожидает ли админ ввод периода
    if context.user_data.get('admin_waiting_period') and is_admin(user_id):
        context.user_data['admin_waiting_period'] = False

        if text.lower() == 'отмена':
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="АДМИН-ПАНЕЛЬ\n\nВыберите действие:",
                reply_markup=get_admin_menu_keyboard()
            )
            return

        try:
            dates = text.split()
            if len(dates) != 2:
                raise ValueError("Неверный формат - нужно две даты")

            start_date = datetime.strptime(dates[0], "%d.%m.%Y")
            end_date = datetime.strptime(dates[1], "%d.%m.%Y").replace(hour=23, minute=59, second=59)

            if start_date > end_date:
                start_date, end_date = end_date, start_date

            # Показываем выбор фильтра по статусу
            keyboard = [
                [InlineKeyboardButton("Все", callback_data=f"admin::photos_date::{dates[0]}::{dates[1]}::all")],
                [InlineKeyboardButton("Подтвержденные",
                                      callback_data=f"admin::photos_date::{dates[0]}::{dates[1]}::confirmed")],
                [InlineKeyboardButton("Отклоненные",
                                      callback_data=f"admin::photos_date::{dates[0]}::{dates[1]}::rejected")],
                [InlineKeyboardButton("Без отзыва",
                                      callback_data=f"admin::photos_date::{dates[0]}::{dates[1]}::pending")],
                [InlineKeyboardButton("<< Назад", callback_data="admin::menu")]
            ]

            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"Период: {dates[0]} - {dates[1]}\n\nВыберите фильтр по статусу:",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )

        except ValueError as e:
            await update.message.reply_text(
                f"Неверный формат даты!\n\n"
                "Используйте формат: ДД.ММ.ГГГГ ДД.ММ.ГГГГ\n"
                "Например: 01.01.2024 31.01.2024\n\n"
                "Или введите 'отмена' для возврата."
            )
            context.user_data['admin_waiting_period'] = True
        return

    # Обычная проверка согласия для не-админов
    if user_id not in agreed_users:
        await send_agreement(update, context)


async def admin_photos_date_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик для фото за период с фильтром"""
    query = update.callback_query
    user_id = query.from_user.id

    if not is_admin(user_id):
        await query.answer("Доступ запрещен!", show_alert=True)
        return

    await query.answer()

    # Формат: admin::photos_date::start::end::filter
    parts = query.data.split("::")
    if len(parts) < 5:
        return

    start_str = parts[2]
    end_str = parts[3]
    status_filter = parts[4]

    try:
        start_date = datetime.strptime(start_str, "%d.%m.%Y")
        end_date = datetime.strptime(end_str, "%d.%m.%Y").replace(hour=23, minute=59, second=59)
    except:
        await query.edit_message_text("Ошибка парсинга даты")
        return

    status_names = {
        "all": "все",
        "confirmed": "подтвержденные",
        "rejected": "отклоненные",
        "pending": "без отзыва"
    }

    await query.edit_message_text(f"Загрузка фотографий за {start_str} - {end_str}...")

    photos = get_photos_filtered(start_date=start_date, end_date=end_date, status_filter=status_filter)

    if not photos:
        keyboard = [[InlineKeyboardButton("<< Назад в меню", callback_data="admin::menu")]]
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text="Фотографии за указанный период не найдены.",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return

    for photo_data in photos:
        await send_photo_with_info(context, query.message.chat_id, photo_data)

    keyboard = [
        [
            InlineKeyboardButton("Все", callback_data=f"admin::photos_date::{start_str}::{end_str}::all"),
            InlineKeyboardButton("Подтв.", callback_data=f"admin::photos_date::{start_str}::{end_str}::confirmed"),
        ],
        [
            InlineKeyboardButton("Откл.", callback_data=f"admin::photos_date::{start_str}::{end_str}::rejected"),
            InlineKeyboardButton("Ожидает", callback_data=f"admin::photos_date::{start_str}::{end_str}::pending"),
        ],
        [InlineKeyboardButton("<< Назад в меню", callback_data="admin::menu")]
    ]
    await context.bot.send_message(
        chat_id=query.message.chat_id,
        text=f"Показано {len(photos)} фото за {start_str} - {end_str} (фильтр: {status_names.get(status_filter, status_filter)})",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик нажатий на кнопки"""
    query = update.callback_query
    user_id = query.from_user.id

    # Проверяем админские кнопки
    if query.data.startswith("admin::"):
        if query.data.startswith("admin::photos_date::"):
            await admin_photos_date_callback(update, context)
        else:
            await admin_callback_handler(update, context)
        return

    # Проверяем согласие
    if user_id not in agreed_users:
        await query.answer("Сначала необходимо согласиться с пользовательским соглашением!", show_alert=True)
        return

    await query.answer()

    parts = query.data.split('::')

    if len(parts) < 2:
        print(f"Ошибка: некорректный callback_data: {query.data}")
        return

    action = parts[0]

    if action == "confirm":
        filename = parts[1]
        filepath = f"images_from_bot/{filename}"

        original_diagnosis = context.user_data.get('last_diagnosis', 'unknown')
        save_diagnosis(filepath, original_diagnosis, confirmed_by_model=True)

        await query.edit_message_text(
            text="Спасибо за ваш ответ. Это поможет нам улучшать приложение!",
            reply_markup=None
        )

    elif action == "reject":
        filename = parts[1]

        keyboard = []
        diagnoses_items = list(DIAGNOSES_MAP.items())

        for i in range(0, len(diagnoses_items), 2):
            row = []
            for j in range(2):
                if i + j < len(diagnoses_items):
                    key, value = diagnoses_items[i + j]
                    if key == "other":
                        callback_action = "other"
                    else:
                        callback_action = "select"
                    row.append(InlineKeyboardButton(value, callback_data=f"{callback_action}::{key}::{filename}"))
            keyboard.append(row)

        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_text(
            text="Пожалуйста, выберите правильный диагноз из списка:",
            reply_markup=reply_markup
        )

    elif action == "other":
        if len(parts) < 3:
            return

        key = parts[1]
        filename = parts[2]

        keyboard = []
        for popular_key, value in POPULAR_DIAGNOSES.items():
            keyboard.append([InlineKeyboardButton(value, callback_data=f"popular::{popular_key}::{filename}")])

        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_text(
            text="Выберите наиболее подходящий вариант из распространенных диагнозов:",
            reply_markup=reply_markup
        )

    elif action == "select":
        if len(parts) < 3:
            return

        key = parts[1]
        filename = parts[2]
        filepath = f"images_from_bot/{filename}"

        save_diagnosis(filepath, key, confirmed_by_model=False)

        await query.edit_message_text(
            text="Спасибо за ваш ответ. Это поможет нам улучшать приложение!",
            reply_markup=None
        )

    elif action == "popular":
        if len(parts) < 3:
            return

        key = parts[1]
        filename = parts[2]
        filepath = f"images_from_bot/{filename}"

        save_diagnosis(filepath, key, confirmed_by_model=False)

        await query.edit_message_text(
            text="Спасибо за ваш ответ. Это поможет нам улучшать приложение!",
            reply_markup=None
        )


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global img_count

    user_id = update.effective_user.id
    if user_id not in agreed_users:
        await send_agreement(update, context)
        return

    print(f"Получил изображение. В очереди {img_count}")
    img_count += 1
    log_request(user_id, "image")

    try:
        photo_file = await update.message.photo[-1].get_file()
        photo_bytes = await photo_file.download_as_bytearray()
        nparr = np.frombuffer(photo_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

        def process_image_for_model(img, target_size=244, scale=1.0):
            h, w = img.shape[:2]

            # Базовый масштаб: короткая сторона = target_size (cover, без чёрных полей)
            base_scale = target_size / min(h, w)

            # Пользовательский scale поверх базового
            final_scale = base_scale * scale

            new_w = int(w * final_scale)
            new_h = int(h * final_scale)
            scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

            # Паддинг если вдруг изображение меньше target_size (при очень малом scale)
            pad_h = max(0, target_size - new_h)
            pad_w = max(0, target_size - new_w)

            if pad_h > 0 or pad_w > 0:
                top = pad_h // 2
                bottom = pad_h - top
                left = pad_w // 2
                right = pad_w - left
                scaled = cv2.copyMakeBorder(
                    scaled, top, bottom, left, right,
                    cv2.BORDER_CONSTANT, value=(0, 0, 0)
                )
                new_h, new_w = scaled.shape[:2]

            # Центральный кроп до target_size × target_size
            center_x = new_w // 2
            center_y = new_h // 2
            half = target_size // 2

            x1 = center_x - half
            y1 = center_y - half
            x2 = x1 + target_size
            y2 = y1 + target_size

            if x1 < 0:
                x1, x2 = 0, target_size
            if y1 < 0:
                y1, y2 = 0, target_size
            if x2 > new_w:
                x2, x1 = new_w, new_w - target_size
            if y2 > new_h:
                y2, y1 = new_h, new_h - target_size

            cropped = scaled[y1:y2, x1:x2]

            # Финальный фолбэк без растяжения
            ch, cw = cropped.shape[:2]
            if ch != target_size or cw != target_size:
                pad_h2 = max(0, target_size - ch)
                pad_w2 = max(0, target_size - cw)
                cropped = cv2.copyMakeBorder(
                    cropped,
                    pad_h2 // 2, pad_h2 - pad_h2 // 2,
                    pad_w2 // 2, pad_w2 - pad_w2 // 2,
                    cv2.BORDER_CONSTANT, value=(0, 0, 0)
                )
                cropped = cropped[:target_size, :target_size]

            return cropped

        processed_image = process_image_for_model(image, target_size=244, scale=1.5)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"image_{timestamp}.jpg"
        filepath = f"images_from_bot/{filename}"

        os.makedirs("images_from_bot", exist_ok=True)
        cv2.imwrite(filepath, processed_image)

        result = model(processed_image)[0]
        predict_img = result.names[result.probs.top5[0]]
        probability = float(result.probs.top1conf)

        print(f"Класс: {predict_img}, уверенность: {probability}")

        log_photo(filename, user_id, predict_img, probability)

        context.user_data['last_filename'] = filepath
        context.user_data['last_diagnosis'] = predict_img

        confirm_text = f"Правильно ли бот определил болезнь: \n*{DIAGNOSES_MAP.get(predict_img)}*?\n\nВыберите один из вариантов:"
        if predict_img == "panu":
            reply_message = (
                "*Изображение обработано!*\n\n"
                f"{f'Предполагаемое заболевание кожи - *Разноцветный лишай* с вероятностью {probability:.2%}.' if probability >= 0.75 else f'Обнаружены признаки, которые могут соответствовать *Разноцветному лишаю*, но вероятность составляет {probability:.2%}.'}\n\n"
                "- *Описание:* Грибковая инфекция, проявляется в виде шелушащихся пятен белого, розового или коричневого цвета. Не заразна.\n\n"
                "- *Рекомендации:* Противогрибковые шампуни, лосьоны или кремы (например: кетоконазол, клотримазол). В сложных случаях — таблетки по назначению врача.\n\n"
                "- *Профилактика:* Соблюдайте правила личной гигиены, носите свободную одежду из натуральных тканей, регулярно принимайте душ, особенно после потоотделения. Не травмировать кожу абразивными скрабами и пилингами.\n\n"
                "- *Серьёзность заболевания:* Низкая. Не опасен для здоровья, но требует лечения для устранения пятен.\n\n"
                "*Внимание*, ИИ часто выводит данный результат на здоровую кожу. Если вы уверены, что у вас разноцветный лишай, то обратитесь к Дерматологу."
            )
        elif predict_img == "rosacea":
            reply_message = (
                "*Изображение обработано!*\n\n"
                f"{f'Предполагаемое заболевание кожи - *Розацеа* с вероятностью {probability:.2%}.' if probability >= 0.75 else f'Обнаружены признаки, которые могут соответствовать *Розацеа*, но вероятность составляет {probability:.2%}.'}\n\n"
                "- *Описание:* Хроническое заболевание кожи лица. Проявляется в виде покраснения, сосудистых звездочек, позже — бугорки и гнойнички.\n\n"
                "- *Рекомендации:* Избегать триггеров (острая еда, алкоголь, солнце). Наружные гели (метронидазол) или антибиотики по рецепту врача. Лазерная терапия.\n\n"
                "- *Профилактика:* Невозможно разработать первичную профилактику, но можно предотвратить или снизить частоту обострений, избегая провоцирующих факторов (горячая пища, алкоголь, солнце, стресс), используя щадящий уход за кожей и защищая ее от солнца, ветра и холода.\n\n"
                "- *Серьёзность заболевания:* Низкая. Не опасна, но носит хронический характер и влияет на внешность.\n\n"
                "Мы рекомендуем обратится вам к Дерматологу для получения лечения и уточнения диагноза. ИИ может ошибаться."
            )
        elif predict_img == "herpes":
            reply_message = (
                "*Изображение обработано!*\n\n"
                f"{f'Предполагаемое заболевание кожи - *Герпес* с вероятностью {probability:.2%}.' if probability >= 0.75 else f'Обнаружены признаки, которые могут соответствовать *Герпес*, но вероятность составляет {probability:.2%}.'}\n\n"
                "- *Описание:* Вирусная инфекция. Появляются на коже (чаще всего на губах или вокруг них) и слизистых оболочках, сначала вызывают зуд и покраснение, затем лопаются, образуя язвочки, которые покрываются корочкой.\n\n"
                "- *Рекомендации:* Противовирусные мази (ацикловир) и таблетки. Начинать лечение при первых симптомах (зуд, жжение).\n\n"
                "- *Профилактика:* Cоблюдайте правила личной гигиены, не трогайте высыпания, используйте индивидуальные предметы гигиены (полотенца, зубные щетки) и косметику. Укрепляйте иммунитет, ведите здоровый образ жизни, правильно питайтесь и избегайте переохлаждения.\n\n"
                "- *Серьёзность заболевания:* Средняя. Для большинства — косметическая проблема. Опасен для людей с иммунодефицитом.\n\n"
                "Мы рекомендуем обратится вам к Дерматологу для получения лечения и уточнения диагноза. ИИ может ошибаться."
            )
        elif predict_img == "eksim":
            reply_message = (
                "*Изображение обработано!*\n\n"
                f"{f'Предполагаемое заболевание кожи - *Экзема* с вероятностью {probability:.2%}.' if probability >= 0.75 else f'Обнаружены признаки, которые могут соответствовать *Экзема*, но вероятность составляет {probability:.2%}.'}\n\n"
                "- *Описание:* Экзема – это хроническое незаразное воспалительное заболевание кожи, вызывающее зуд, покраснение, отечность и высыпания, которые в процессе могут мокнуть, покрываться корками и шелушиться.\n\n"
                "- *Рекомендации:* Увлажняющие средства (эмоленты), гормональные мази (кортикостероиды) для снятия воспаления, антигистаминные препараты от зуда.\n\n"
                "- *Профилактика:* Устранение индивидуальных триггеров, которые могут спровоцировать обострение заболевания, например: аллергены, раздражители и стресс. Важно также использовать увлажняющие кремы и носить одежду из натуральных тканей.\n\n"
                "- *Серьёзность заболевания:* Средняя. Не заразна, но может значительно ухудшать качество жизни из-за зуда.\n\n"
                "Мы рекомендуем обратится вам к Дерматологу для получения лечения и уточнения диагноза. ИИ может ошибаться."
            )
        elif predict_img == "acne":
            reply_message = (
                "*Изображение обработано!*\n"
                f"{f'Предполагаемое заболевание кожи - *Акне* с вероятностью {probability:.2%}.' if probability >= 0.75 else f'Обнаружены признаки, которые могут соответствовать *Акне*, но вероятность составляет {probability:.2%}.'}\n\n"
                "- *Описание:* Воспаление сальных желез. Проявляется как черные точки (комедоны), красные прыщики (папулы) и гнойнички (пустулы).\n\n"
                "- *Рекомендации:* Уход средствами с салициловой кислотой, бензоилпероксидом. Ретиноиды и антибиотики (наружно и внутрь) по назначению врача.\n\n"
                "- *Профилактика:* Следует соблюдать гигиену кожи, избегать выдавливания прыщей, использовать правильно подобранные косметические средства (не содержащие спирта), соблюдать сбалансированное питание и поддерживать здоровый образ жизни. Не прикасаться к лицу грязными руками в течение дня.\n\n"
                "- *Серьёзность заболевания:* Низкая. Распространенное состояние, но тяжелые формы могут оставлять рубцы и требовать лечения у дерматолога.\n\n"
                "Мы рекомендуем обратится вам к Дерматологу для получения лечения и уточнения диагноза. ИИ может ошибаться."
            )
        else:
            confirm_text = "Правильно ли бот определил, что ваша кожа здорова?\n\nВыберите один из вариантов:"
            reply_message = (
                "*Изображение обработано!*\n"
                f"Ваша кожа здорова с вероятностью {probability:.2%}! Или данного заболевания нет в списке...\n"
                "Если у Вас есть подозрение на заболевание, то рекомендуем обратиться к Дерматологу для определения болезни. ИИ может ошибаться."
            )

        await update.message.reply_text(reply_message, parse_mode='Markdown')

        keyboard = [
            [InlineKeyboardButton("Диагноз подтвердился", callback_data=f"confirm::{filename}")],
            [InlineKeyboardButton("Диагноз не тот", callback_data=f"reject::{filename}")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        confirmation_message = await update.message.reply_text(
            str(confirm_text),
            parse_mode='Markdown',
            reply_markup=reply_markup
        )

        context.user_data['confirmation_message_id'] = confirmation_message.message_id


    except Exception as e:
        print(f"Ошибка при обработке изображения: {e}")
        if isinstance(e, TimedOut):
            await update.message.reply_text(
                "⏳ Сервер не успел получить фотографию — это временная проблема.\n\n"
                "Пожалуйста, отправьте фото ещё раз."
            )
        else:
            await update.message.reply_text(
                "К сожалению, не удалось обработать изображение. Пожалуйста, попробуйте отправить другое изображение или через пару минут.\n\n"
                f"Ошибка: {e}"
            )

    img_count -= 1


model = YOLO('new_best.pt')

application = Application.builder().token(TOKEN).build()

# Регистрируем обработчики
application.add_handler(CommandHandler("start", start))
application.add_handler(CommandHandler("help", help_command))
application.add_handler(CommandHandler("photo", photo_command))
application.add_handler(CallbackQueryHandler(agreement_handler, pattern="^(agree|disagree)$"))
application.add_handler(CallbackQueryHandler(button_callback))
application.add_handler(MessageHandler(filters.PHOTO, handle_image))
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

print("Бот запущен...")
application.run_polling()
