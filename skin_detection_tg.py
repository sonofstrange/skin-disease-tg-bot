from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)
from telegram.error import TimedOut
import cv2
import numpy as np
from datetime import datetime, timedelta
import json
import os
import io
import asyncio

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

import onnxruntime as ort
from PIL import Image, ImageOps

# ===================== НАСТРОЙКИ =====================

TOKEN = "YOUR_TONEK_HERE"

img_count = 0

ONNX_MODEL_PATH = "best_model.onnx"
IMG_SIZE = 244
CLASS_NAMES = ["acne", "eksim", "herpes", "panu", "rosacea", "skin"]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PAD_COLOR = (180, 150, 130)

CONFIDENCE_THRESHOLD = 0.50
MARGIN_THRESHOLD = 0.10
CONFUSION_MARGIN = 0.18

SKIN_OVERRIDE_MIN = 0.35
DISEASE_STRONG_THRESHOLD = 0.88
DISEASE_MARGIN_STRONG = 0.30

SKIN_OVERRIDE_TOP1_MAX = 0.62
SKIN_OVERRIDE_MARGIN_MAX = 0.08

MAX_PHOTOS_PER_MESSAGE = 4

MEDIA_GROUP_BUFFER = {}
MEDIA_GROUP_TASKS = {}
MEDIA_GROUP_DELAY = 1.5

AGREED_USERS_FILE = "agreed_users.json"
DIAGNOSES_FILE = "diagnoses.json"
STATS_FILE = "stats.json"
ADMIN_SESSIONS_FILE = "admin_sessions.json"

ADMIN_LOGIN = "adm"
ADMIN_PASSWORD = "123"

# ===================== ЗАГРУЗКА МОДЕЛИ =====================

print(f"Загрузка модели: {ONNX_MODEL_PATH}")
ort_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
ORT_INPUT_NAME = ort_session.get_inputs()[0].name
ORT_OUTPUT_NAME = ort_session.get_outputs()[0].name
print("Модель загружена ✓")

# ===================== ЗАГРУЗКА ДАННЫХ =====================

agreed_users = set()
if os.path.exists(AGREED_USERS_FILE):
    try:
        with open(AGREED_USERS_FILE, "r") as f:
            agreed_users = set(json.load(f))
    except:
        agreed_users = set()

admin_sessions = set()
if os.path.exists(ADMIN_SESSIONS_FILE):
    try:
        with open(ADMIN_SESSIONS_FILE, "r") as f:
            admin_sessions = set(json.load(f))
    except:
        admin_sessions = set()

stats_data = {"requests": [], "photos": []}
if os.path.exists(STATS_FILE):
    try:
        with open(STATS_FILE, "r") as f:
            stats_data = json.load(f)
    except:
        stats_data = {"requests": [], "photos": []}


# ===================== СЛУЖЕБНЫЕ ФУНКЦИИ =====================


def save_admin_sessions():
    with open(ADMIN_SESSIONS_FILE, "w") as f:
        json.dump(list(admin_sessions), f)


def save_stats():
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats_data, f, ensure_ascii=False, indent=2)


def log_request(user_id: int, request_type: str):
    stats_data["requests"].append(
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_id": user_id,
            "type": request_type,
        }
    )
    save_stats()


def log_photo(filename: str, user_id: int, ai_prediction: str, ai_confidence: float):
    stats_data["photos"].append(
        {
            "filename": filename,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_id": user_id,
            "ai_prediction": ai_prediction,
            "ai_confidence": ai_confidence,
            "user_verdict": None,
        }
    )
    save_stats()


def update_photo_verdict(filename: str, user_verdict: str):
    for photo in stats_data["photos"]:
        if photo["filename"] == filename or photo["filename"].endswith(filename):
            photo["user_verdict"] = user_verdict
            save_stats()
            break


# ===================== МАППИНГИ =====================

DIAGNOSES_MAP = {
    "panu": "Разноцветный лишай",
    "rosacea": "Розацеа",
    "herpes": "Герпес",
    "eksim": "Экзема",
    "acne": "Акне",
    "skin": "Здоровая кожа",
    "other": "Больше болезней",
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
    "other": "Болезни нет в списке",
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
    with open(DIAGNOSES_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)


def save_agreed_user(user_id):
    agreed_users.add(user_id)
    with open(AGREED_USERS_FILE, "w") as f:
        json.dump(list(agreed_users), f)


# ===================== ПРЕДОБРАБОТКА (ЧИСТАЯ, КАК ПРИ ТЕСТЕ) =====================


def prepare_for_model(img_bgr: np.ndarray) -> tuple:
    """
    Предобработка ТОЧНО как при обучении/валидации:
    - BGR -> RGB
    - Resize: короткая сторона = IMG_SIZE * 1.05
    - CenterCrop до IMG_SIZE x IMG_SIZE
    - Нормализация ImageNet

    Возвращает:
    - input_tensor
    - pil_cropped_image (для сохранения / отладки)
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    target_short = int(IMG_SIZE * 1.05)
    w, h = pil_img.size

    if w < h:
        new_w = target_short
        new_h = int(h * target_short / w)
    else:
        new_h = target_short
        new_w = int(w * target_short / h)

    pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    left = (new_w - IMG_SIZE) // 2
    top = (new_h - IMG_SIZE) // 2
    pil_img = pil_img.crop((left, top, left + IMG_SIZE, top + IMG_SIZE))

    arr = np.array(pil_img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)

    return arr.astype(np.float32), pil_img


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


# ===================== ПОСТ-ОБРАБОТКА =====================

CONFUSED_PAIRS = {
    frozenset({"eksim", "rosacea"}): (
        "⚠️ Модель затрудняется отличить *Экзему* от *Розацеа*. "
        "Эти заболевания внешне похожи.\n"
        "Рекомендуем обратиться к дерматологу для точного диагноза."
    ),
    frozenset({"acne", "rosacea"}): (
        "⚠️ Модель затрудняется отличить *Акне* от *Розацеа*. "
        "Эти заболевания могут выглядеть похоже.\n"
        "Рекомендуем обратиться к дерматологу."
    ),
    frozenset({"eksim", "herpes"}): (
        "⚠️ Модель затрудняется отличить *Экзему* от *Герпеса*. "
        "Если есть пузырьки — скорее герпес. Если шелушение — скорее экзема.\n"
        "Рекомендуем обратиться к дерматологу."
    ),
}


def run_inference(img_bgr: np.ndarray) -> dict:
    """
    Один center-crop, без TTA, без pad, без агрессивной постобработки.
    """
    inp, _ = prepare_for_model(img_bgr)

    outputs = ort_session.run([ORT_OUTPUT_NAME], {ORT_INPUT_NAME: inp})
    logits = outputs[0][0]
    probs = softmax(logits)

    sorted_idx = np.argsort(probs)[::-1]
    top1_idx = int(sorted_idx[0])
    top2_idx = int(sorted_idx[1])

    top1_class = CLASS_NAMES[top1_idx]
    top2_class = CLASS_NAMES[top2_idx]
    top1_conf = float(probs[top1_idx])
    top2_conf = float(probs[top2_idx])
    margin = top1_conf - top2_conf

    skin_idx = CLASS_NAMES.index("skin")
    skin_conf = float(probs[skin_idx])

    is_uncertain = (top1_conf < CONFIDENCE_THRESHOLD) or (margin < MARGIN_THRESHOLD)

    # Очень мягкий fallback в skin:
    # только если skin уже на 2 месте,
    # и разница совсем маленькая
    if top1_class != "skin":
        if (
                top2_class == "skin"
                and top1_conf < SKIN_OVERRIDE_TOP1_MAX
                and margin < SKIN_OVERRIDE_MARGIN_MAX
                and skin_conf >= SKIN_OVERRIDE_MIN
        ):
            top2_class = top1_class
            top2_conf = top1_conf
            top1_class = "skin"
            top1_conf = skin_conf
            is_uncertain = True

    confusion_warning = None
    pair = frozenset({top1_class, top2_class})
    if margin < CONFUSION_MARGIN and pair in CONFUSED_PAIRS:
        confusion_warning = CONFUSED_PAIRS[pair]
        is_uncertain = True

    return {
        "class": top1_class,
        "confidence": top1_conf,
        "top2_class": top2_class,
        "top2_conf": top2_conf,
        "margin": margin,
        "uncertain": is_uncertain,
        "confusion_warning": confusion_warning,
        "all_probs": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
    }


# ===================== ГЕНЕРАЦИЯ ТЕКСТА =====================


def build_reply_message(result: dict) -> str:
    predict_img = result["class"]
    probability = result["confidence"]
    uncertain = result["uncertain"]
    confusion = result.get("confusion_warning")

    high_conf = probability >= 0.75

    warnings = ""
    if confusion:
        warnings += f"\n{confusion}\n"
        warnings += (
            f"\nВариант 1: *{DIAGNOSES_MAP.get(result['class'], result['class'])}* "
            f"({result['confidence']:.2%})\n"
            f"Вариант 2: *{DIAGNOSES_MAP.get(result['top2_class'], result['top2_class'])}* "
            f"({result['top2_conf']:.2%})\n"
        )
    elif uncertain:
        warnings += (
            f"\n⚠️ *Внимание:* уверенность модели невысока ({probability:.2%}). "
            f"Результат следует воспринимать с осторожностью. "
            f"Возможно также: *{DIAGNOSES_MAP.get(result['top2_class'], result['top2_class'])}* "
            f"({result['top2_conf']:.2%}).\n"
        )

    if predict_img == "panu":
        main_text = (
            f"{f'Предполагаемое заболевание кожи - *Разноцветный лишай* с вероятностью {probability:.2%}.' if high_conf else f'Обнаружены признаки, которые могут соответствовать *Разноцветному лишаю*, но вероятность составляет {probability:.2%}.'}\n\n"
            "- *Описание:* Грибковая инфекция, проявляется в виде шелушащихся пятен белого, розового или коричневого цвета. Не заразна.\n\n"
            "- *Рекомендации:* Противогрибковые шампуни, лосьоны или кремы (например: кетоконазол, клотримазол). В сложных случаях — таблетки по назначению врача.\n\n"
            "- *Профилактика:* Соблюдайте правила личной гигиены, носите свободную одежду из натуральных тканей, регулярно принимайте душ, особенно после потоотделения. Не травмировать кожу абразивными скрабами и пилингами.\n\n"
            "- *Серьёзность заболевания:* Низкая. Не опасен для здоровья, но требует лечения для устранения пятен.\n\n"
            "*Внимание*, ИИ часто выводит данный результат на здоровую кожу. Если вы уверены, что у вас разноцветный лишай, то обратитесь к Дерматологу."
        )
    elif predict_img == "rosacea":
        main_text = (
            f"{f'Предполагаемое заболевание кожи - *Розацеа* с вероятностью {probability:.2%}.' if high_conf else f'Обнаружены признаки, которые могут соответствовать *Розацеа*, но вероятность составляет {probability:.2%}.'}\n\n"
            "- *Описание:* Хроническое заболевание кожи лица. Проявляется в виде покраснения, сосудистых звездочек, позже — бугорки и гнойнички.\n\n"
            "- *Рекомендации:* Избегать триггеров (острая еда, алкоголь, солнце). Наружные гели (метронидазол) или антибиотики по рецепту врача. Лазерная терапия.\n\n"
            "- *Профилактика:* Невозможно разработать первичную профилактику, но можно предотвратить или снизить частоту обострений, избегая провоцирующих факторов (горячая пища, алкоголь, солнце, стресс), используя щадящий уход за кожей и защищая ее от солнца, ветра и холода.\n\n"
            "- *Серьёзность заболевания:* Низкая. Не опасна, но носит хронический характер и влияет на внешность.\n\n"
            "Мы рекомендуем обратится вам к Дерматологу для получения лечения и уточнения диагноза. ИИ может ошибаться."
        )
    elif predict_img == "herpes":
        main_text = (
            f"{f'Предполагаемое заболевание кожи - *Герпес* с вероятностью {probability:.2%}.' if high_conf else f'Обнаружены признаки, которые могут соответствовать *Герпес*, но вероятность составляет {probability:.2%}.'}\n\n"
            "- *Описание:* Вирусная инфекция. Появляются на коже (чаще всего на губах или вокруг них) и слизистых оболочках, сначала вызывают зуд и покраснение, затем лопаются, образуя язвочки, которые покрываются корочкой.\n\n"
            "- *Рекомендации:* Противовирусные мази (ацикловир) и таблетки. Начинать лечение при первых симптомах (зуд, жжение).\n\n"
            "- *Профилактика:* Cоблюдайте правила личной гигиены, не трогайте высыпания, используйте индивидуальные предметы гигиены (полотенца, зубные щетки) и косметику. Укрепляйте иммунитет, ведите здоровый образ жизни, правильно питайтесь и избегайте переохлаждения.\n\n"
            "- *Серьёзность заболевания:* Средняя. Для большинства — косметическая проблема. Опасен для людей с иммунодефицитом.\n\n"
            "Мы рекомендуем обратится вам к Дерматологу для получения лечения и уточнения диагноза. ИИ может ошибаться."
        )
    elif predict_img == "eksim":
        main_text = (
            f"{f'Предполагаемое заболевание кожи - *Экзема* с вероятностью {probability:.2%}.' if high_conf else f'Обнаружены признаки, которые могут соответствовать *Экзема*, но вероятность составляет {probability:.2%}.'}\n\n"
            "- *Описание:* Экзема – это хроническое незаразное воспалительное заболевание кожи, вызывающее зуд, покраснение, отечность и высыпания, которые в процессе могут мокнуть, покрываться корками и шелушиться.\n\n"
            "- *Рекомендации:* Увлажняющие средства (эмоленты), гормональные мази (кортикостероиды) для снятия воспаления, антигистаминные препараты от зуда.\n\n"
            "- *Профилактика:* Устранение индивидуальных триггеров, которые могут спровоцировать обострение заболевания, например: аллергены, раздражители и стресс. Важно также использовать увлажняющие кремы и носить одежду из натуральных тканей.\n\n"
            "- *Серьёзность заболевания:* Средняя. Не заразна, но может значительно ухудшать качество жизни из-за зуда.\n\n"
            "Мы рекомендуем обратится вам к Дерматологу для получения лечения и уточнения диагноза. ИИ может ошибаться."
        )
    elif predict_img == "acne":
        main_text = (
            f"{f'Предполагаемое заболевание кожи - *Акне* с вероятностью {probability:.2%}.' if high_conf else f'Обнаружены признаки, которые могут соответствовать *Акне*, но вероятность составляет {probability:.2%}.'}\n\n"
            "- *Описание:* Воспаление сальных желез. Проявляется как черные точки (комедоны), красные прыщики (папулы) и гнойнички (пустулы).\n\n"
            "- *Рекомендации:* Уход средствами с салициловой кислотой, бензоилпероксидом. Ретиноиды и антибиотики (наружно и внутрь) по назначению врача.\n\n"
            "- *Профилактика:* Следует соблюдать гигиену кожи, избегать выдавливания прыщей, использовать правильно подобранные косметические средства (не содержащие спирта), соблюдать сбалансированное питание и поддерживать здоровый образ жизни. Не прикасаться к лицу грязными руками в течение дня.\n\n"
            "- *Серьёзность заболевания:* Низкая. Распространенное состояние, но тяжелые формы могут оставлять рубцы и требовать лечения у дерматолога.\n\n"
            "Мы рекомендуем обратится вам к Дерматологу для получения лечения и уточнения диагноза. ИИ может ошибаться."
        )
    else:
        main_text = (
            f"Ваша кожа здорова с вероятностью {probability:.2%}! Или данного заболевания нет в списке...\n"
            "Если у Вас есть подозрение на заболевание, то рекомендуем обратиться к Дерматологу для определения болезни. ИИ может ошибаться."
        )

    return f"*Изображение обработано!*\n{warnings}\n{main_text}"


def build_confirm_text(predict_img: str, uncertain: bool) -> str:
    if predict_img == "skin":
        base = "Правильно ли бот определил, что ваша кожа здорова?"
    else:
        diagnosis_name = DIAGNOSES_MAP.get(predict_img, predict_img)
        base = f"Правильно ли бот определил болезнь: \n*{diagnosis_name}*?"

    if uncertain:
        base += "\n\n⚠️ Уверенность модели невысока — пожалуйста, проверьте результат."

    return base + "\n\nВыберите один из вариантов:"


# ===================== АДМИН-ПАНЕЛЬ =====================


def is_admin(user_id: int) -> bool:
    return user_id in admin_sessions


def get_admin_menu_keyboard():
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("Статистика за сегодня", callback_data="admin::stats::today")],
            [InlineKeyboardButton("Статистика за неделю", callback_data="admin::stats::week")],
            [InlineKeyboardButton("Статистика за месяц", callback_data="admin::stats::month")],
            [InlineKeyboardButton("Последние 5 фото", callback_data="admin::photos::5::all")],
            [InlineKeyboardButton("Последние 10 фото", callback_data="admin::photos::10::all")],
            [InlineKeyboardButton("Фото по статусу", callback_data="admin::photos_filter")],
            [InlineKeyboardButton("Фото за период", callback_data="admin::photos_period")],
            [InlineKeyboardButton("Общая статистика", callback_data="admin::general")],
            [InlineKeyboardButton("Выйти из админки", callback_data="admin::logout")],
        ]
    )


def get_status_filter_keyboard():
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("Все фото", callback_data="admin::photos::10::all")],
            [InlineKeyboardButton("Подтвержденные", callback_data="admin::photos::10::confirmed")],
            [InlineKeyboardButton("Отклоненные", callback_data="admin::photos::10::rejected")],
            [InlineKeyboardButton("Без отзыва", callback_data="admin::photos::10::pending")],
            [InlineKeyboardButton("<< Назад", callback_data="admin::menu")],
        ]
    )


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
    bars1 = ax.bar([i - width / 2 for i in x], requests_values, width, label="Vse zaprosy", color="#3498db")
    bars2 = ax.bar([i + width / 2 for i in x], photos_values, width, label="Fotografii", color="#e74c3c")

    ax.set_xlabel("Data")
    ax.set_ylabel("Количество")
    ax.set_title(f"Статистика за последние {days} дней")
    ax.set_xticks(x)
    ax.set_xticklabels([d.strftime("%d.%m") for d in all_dates], rotation=45)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.annotate(
                f"{int(height)}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.annotate(
                f"{int(height)}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
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
        ax.text(0.5, 0.5, "Нет данных", ha="center", va="center", fontsize=20)
        ax.axis("off")
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

        labels1 = [ALL_DIAGNOSES.get(k, k) for k in diagnoses_count.keys()]
        sizes1 = list(diagnoses_count.values())
        colors1 = plt.cm.Set3(np.linspace(0, 1, len(labels1)))
        ax1.pie(sizes1, labels=labels1, autopct="%1.1f%%", colors=colors1, startangle=90)
        ax1.set_title("Предсказания ИИ")

        if user_verdicts_count:
            labels2 = [ALL_DIAGNOSES.get(k, k) for k in user_verdicts_count.keys()]
            sizes2 = list(user_verdicts_count.values())
            colors2 = plt.cm.Set2(np.linspace(0, 1, len(labels2)))
            ax2.pie(sizes2, labels=labels2, autopct="%1.1f%%", colors=colors2, startangle=90)
            ax2.set_title("Вердикты пользователей")
        else:
            ax2.text(0.5, 0.5, "Нет вердиктов", ha="center", va="center", fontsize=14)
            ax2.axis("off")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
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

    plt.bar(hours, counts, color="#9b59b6", edgecolor="black")
    plt.xlabel("Час дня")
    plt.ylabel("Количество запросов")
    plt.title("Активность по часам")
    plt.xticks(hours)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
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

    labels = ["Потверждено", "Отклонено", "Без ответа"]
    sizes = [confirmed, rejected, no_feedback]
    colors = ["#2ecc71", "#e74c3c", "#95a5a6"]
    explode = (0.05, 0.05, 0)

    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct="%1.1f%%", shadow=True, startangle=90)
    plt.title("Точность предсказаний ИИ")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    plt.close()
    return buf


def get_photos_filtered(start_date=None, end_date=None, limit=None, status_filter="all"):
    photos = stats_data["photos"].copy()

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

    photos.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    if limit:
        photos = photos[:limit]

    return photos


async def send_photo_with_info(context: ContextTypes.DEFAULT_TYPE, chat_id: int, photo_data: dict):
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
            with open(filepath, "rb") as photo_file:
                await context.bot.send_photo(chat_id=chat_id, photo=photo_file, caption=caption)
        except Exception as e:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"Ошибка отправки фото: {filename}\n{caption}\n\nОшибка: {e}",
            )
    else:
        await context.bot.send_message(chat_id=chat_id, text=f"Файл не найден: {filename}\n\n{caption}")


async def admin_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
        await query.edit_message_text("АДМИН-ПАНЕЛЬ\n\nВыберите действие:", reply_markup=get_admin_menu_keyboard())
        return

    if action == "photos_filter":
        await query.edit_message_text("Выберите фильтр по статусу:", reply_markup=get_status_filter_keyboard())
        return

    if action == "photos_period":
        context.user_data["admin_waiting_period"] = True
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
                chat_id=query.message.chat_id, photo=daily_chart, caption=f"Запросы за последние {days} дней"
            )

            diagnoses_chart = generate_diagnoses_pie_chart()
            await context.bot.send_photo(
                chat_id=query.message.chat_id, photo=diagnoses_chart, caption="Распределение диагнозов"
            )

            hourly_chart = generate_hourly_chart()
            await context.bot.send_photo(
                chat_id=query.message.chat_id, photo=hourly_chart, caption="Активность по часам"
            )

            accuracy_chart = generate_accuracy_chart()
            await context.bot.send_photo(
                chat_id=query.message.chat_id, photo=accuracy_chart, caption="Точность предсказаний"
            )

            keyboard = [[InlineKeyboardButton("<< Назад в меню", callback_data="admin::menu")]]
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text="Статистика сгенерирована!",
                reply_markup=InlineKeyboardMarkup(keyboard),
            )
        except Exception as e:
            await context.bot.send_message(
                chat_id=query.message.chat_id, text=f"Ошибка генерации статистики: {e}"
            )

    elif action == "photos":
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
            "pending": "без отзыва",
        }

        await query.edit_message_text(
            f"Загрузка фотографий (фильтр: {status_names.get(status_filter, status_filter)})..."
        )

        photos = get_photos_filtered(limit=limit, status_filter=status_filter)

        if not photos:
            keyboard = [[InlineKeyboardButton("<< Назад в меню", callback_data="admin::menu")]]
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text="Фотографии не найдены.",
                reply_markup=InlineKeyboardMarkup(keyboard),
            )
            return

        for photo_data in photos:
            await send_photo_with_info(context, query.message.chat_id, photo_data)

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
            [InlineKeyboardButton("<< Назад в меню", callback_data="admin::menu")],
        ]
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=f"Показано {len(photos)} фото (фильтр: {status_names.get(status_filter, status_filter)})",
            reply_markup=InlineKeyboardMarkup(keyboard),
        )

    elif action == "general":
        total_users = len(agreed_users)
        total_requests = len(stats_data["requests"])
        total_photos = len(stats_data["photos"])

        confirmed = sum(
            1
            for p in stats_data["photos"]
            if p.get("user_verdict") == p.get("ai_prediction") and p.get("user_verdict")
        )
        rejected = sum(
            1
            for p in stats_data["photos"]
            if p.get("user_verdict") and p.get("user_verdict") != p.get("ai_prediction")
        )
        no_feedback = sum(1 for p in stats_data["photos"] if not p.get("user_verdict"))

        accuracy = (confirmed / (confirmed + rejected) * 100) if (confirmed + rejected) > 0 else 0

        today = datetime.now().date()
        today_requests = sum(
            1 for r in stats_data["requests"] if r.get("timestamp", "").startswith(today.strftime("%Y-%m-%d"))
        )
        today_photos = sum(
            1 for p in stats_data["photos"] if p.get("timestamp", "").startswith(today.strftime("%Y-%m-%d"))
        )

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
        await query.edit_message_text(stats_text, reply_markup=InlineKeyboardMarkup(keyboard))


# ===================== ОСНОВНЫЕ ОБРАБОТЧИКИ =====================


async def send_agreement(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("Согласен", callback_data="agree")],
        [InlineKeyboardButton("Не согласен", callback_data="disagree")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    agreement_text = (
            "Пожалуйста, ознакомьтесь с пользовательским соглашением перед использованием бота:\n\n" + USELESS_TEXT
    )

    await update.message.reply_text(agreement_text, reply_markup=reply_markup, parse_mode=None)


async def agreement_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id

    if query.data == "agree":
        save_agreed_user(user_id)
        await query.edit_message_text(
            "Спасибо за согласие! Теперь вы можете пользоваться всеми функциями бота.", reply_markup=None
        )
        welcome_text = (
            "Привет!\n\n"
            "Я бот, который анализирует изображения и находит на них кожные заболевания. Когда ты отправишь картинку, "
            "я мгновенно скажу тебе, есть ли там заболевания, или кожа здорова.\n\n"
            "Используй /help, чтобы узнать больше о доступных командах.\n"
            "Используй /photo, чтобы узнать, как правильно делать фото."
        )
        await context.bot.send_message(chat_id=query.message.chat_id, text=welcome_text)
    elif query.data == "disagree":
        await query.edit_message_text(
            "Без согласия с пользовательским соглашением вы не сможете пользоваться ботом.\n"
            "Если вы передумаете, нажмите /start для повторного просмотра соглашения.",
            reply_markup=None,
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
        "1. Сначала отправь мне изображение кожи (можно до 4 штук сразу — результат усреднится).\n"
        "2. Затем проанализирую и отправлю тебе информацию о заболевании.\n\n"
        f"Принимается большинство стандартных форматов изображений.\nВремя обработки не более {5 + img_count // 10} секунд."
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")


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
        " 3. Фотография должна быть чёткой и сделана при хорошем, естественном освещении.\n"
        " 4. Можно отправить до 4 фото сразу — результаты усреднятся для большей точности.\n\n"
        "После отправки снимок будет автоматически обработан, и искусственный интеллект предоставит предварительный анализ."
    )
    await update.message.reply_text(photo_text)


def save_diagnosis(filename, diagnosis, confirmed_by_model=True):
    try:
        with open(DIAGNOSES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        diagnosis_entry = {
            "filename": filename,
            "diagnosis": diagnosis,
            "confirmed_by_model": confirmed_by_model,
        }

        data.append(diagnosis_entry)

        with open(DIAGNOSES_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Диагноз сохранен: {filename} - {diagnosis}")

        base_filename = os.path.basename(filename)
        update_photo_verdict(base_filename, diagnosis)

    except Exception as e:
        print(f"Ошибка при сохранении диагноза: {e}")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    user_id = update.effective_user.id

    print(f"Получен текст от {user_id}: {text}")

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
            reply_markup=get_admin_menu_keyboard(),
        )
        return

    if context.user_data.get("admin_waiting_period") and is_admin(user_id):
        context.user_data["admin_waiting_period"] = False

        if text.lower() == "отмена":
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="АДМИН-ПАНЕЛЬ\n\nВыберите действие:",
                reply_markup=get_admin_menu_keyboard(),
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

            keyboard = [
                [InlineKeyboardButton("Все", callback_data=f"admin::photos_date::{dates[0]}::{dates[1]}::all")],
                [
                    InlineKeyboardButton(
                        "Подтвержденные", callback_data=f"admin::photos_date::{dates[0]}::{dates[1]}::confirmed"
                    )
                ],
                [
                    InlineKeyboardButton(
                        "Отклоненные", callback_data=f"admin::photos_date::{dates[0]}::{dates[1]}::rejected"
                    )
                ],
                [
                    InlineKeyboardButton(
                        "Без отзыва", callback_data=f"admin::photos_date::{dates[0]}::{dates[1]}::pending"
                    )
                ],
                [InlineKeyboardButton("<< Назад", callback_data="admin::menu")],
            ]

            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"Период: {dates[0]} - {dates[1]}\n\nВыберите фильтр по статусу:",
                reply_markup=InlineKeyboardMarkup(keyboard),
            )

        except ValueError:
            await update.message.reply_text(
                "Неверный формат даты!\n\n"
                "Используйте формат: ДД.ММ.ГГГГ ДД.ММ.ГГГГ\n"
                "Например: 01.01.2024 31.01.2024\n\n"
                "Или введите 'отмена' для возврата."
            )
            context.user_data["admin_waiting_period"] = True
        return

    if user_id not in agreed_users:
        await send_agreement(update, context)


async def admin_photos_date_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id

    if not is_admin(user_id):
        await query.answer("Доступ запрещен!", show_alert=True)
        return

    await query.answer()

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
        "pending": "без отзыва",
    }

    await query.edit_message_text(f"Загрузка фотографий за {start_str} - {end_str}...")

    photos = get_photos_filtered(start_date=start_date, end_date=end_date, status_filter=status_filter)

    if not photos:
        keyboard = [[InlineKeyboardButton("<< Назад в меню", callback_data="admin::menu")]]
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text="Фотографии за указанный период не найдены.",
            reply_markup=InlineKeyboardMarkup(keyboard),
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
        [InlineKeyboardButton("<< Назад в меню", callback_data="admin::menu")],
    ]
    await context.bot.send_message(
        chat_id=query.message.chat_id,
        text=f"Показано {len(photos)} фото за {start_str} - {end_str} (фильтр: {status_names.get(status_filter, status_filter)})",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


# ===================== ОБРАБОТКА ФОТО =====================


async def download_photo_with_retry(context, photo_file_id, max_retries=3):
    """Скачивание фото с retry при TimedOut."""
    for attempt in range(max_retries):
        try:
            photo_file = await context.bot.get_file(photo_file_id)
            return await photo_file.download_as_bytearray()
        except TimedOut:
            if attempt < max_retries - 1:
                print(f"Таймаут скачивания, попытка {attempt + 2}/{max_retries}")
                await asyncio.sleep(2)
            else:
                raise


async def process_single_photo(context, photo_file_id, photo_index, total_photos, chat_id, user_id):
    """Обработка одного фото."""
    try:
        photo_bytes = await download_photo_with_retry(context, photo_file_id)

        nparr = np.frombuffer(photo_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"❌ Не удалось прочитать фото {photo_index + 1}/{total_photos}.",
            )
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"image_{timestamp}_{photo_index}.jpg"

        os.makedirs("images_from_bot", exist_ok=True)
        os.makedirs("images_from_bot/originals", exist_ok=True)
        os.makedirs("images_from_bot/processed", exist_ok=True)

        # Сохраняем оригинал
        cv2.imwrite(f"images_from_bot/originals/{filename}", image)

        # Делаем ровно то, что видит модель: resize + center crop до 244x244
        _, pil_cropped = prepare_for_model(image)

        # Сохраняем обработанную версию (то, что реально видит модель)
        pil_cropped.save(f"images_from_bot/processed/{filename}", quality=95)
        pil_cropped.save(f"images_from_bot/{filename}", quality=95)

        # Инференс
        result = run_inference(image)
        predict_img = result["class"]
        probability = result["confidence"]

        print(
            f"Фото {photo_index + 1}/{total_photos} | "
            f"{predict_img} | {probability:.2%} | "
            f"uncertain={result['uncertain']}"
        )

        log_photo(filename, user_id, predict_img, probability)

        if "diagnosis_by_file" not in context.user_data:
            context.user_data["diagnosis_by_file"] = {}
        context.user_data["diagnosis_by_file"][filename] = predict_img

        header = f"*Фото {photo_index + 1} из {total_photos}*\n\n" if total_photos > 1 else ""

        await context.bot.send_message(
            chat_id=chat_id,
            text=header + build_reply_message(result),
            parse_mode="Markdown",
        )

        confirm_text = build_confirm_text(predict_img, result["uncertain"])
        if total_photos > 1:
            confirm_text = f"Фото {photo_index + 1}/{total_photos}\n\n" + confirm_text

        keyboard = [
            [InlineKeyboardButton("Диагноз подтвердился", callback_data=f"confirm::{filename}")],
            [InlineKeyboardButton("Диагноз не тот", callback_data=f"reject::{filename}")],
        ]

        await context.bot.send_message(
            chat_id=chat_id,
            text=confirm_text,
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(keyboard),
        )

    except TimedOut:
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"⏳ Не удалось скачать фото {photo_index + 1}/{total_photos}. Попробуйте ещё раз.",
        )
    except Exception as e:
        print(f"Ошибка фото {photo_index + 1}: {e}")
        import traceback
        traceback.print_exc()
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"❌ Ошибка при обработке фото {photo_index + 1}: {e}",
        )


async def process_album_group(context, photo_ids, chat_id, user_id):
    """
    Обработка альбома:
    - скачиваем все фото
    - считаем probs для каждого
    - усредняем вероятности
    - выдаём один итоговый ответ
    """
    total = min(len(photo_ids), MAX_PHOTOS_PER_MESSAGE)

    probs_list = []
    per_photo_classes = []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    album_filename = f"album_{timestamp}.jpg"

    os.makedirs("images_from_bot", exist_ok=True)
    os.makedirs("images_from_bot/originals", exist_ok=True)

    for idx, fid in enumerate(photo_ids[:MAX_PHOTOS_PER_MESSAGE]):
        try:
            photo_bytes = await download_photo_with_retry(context, fid)
            nparr = np.frombuffer(photo_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                continue

            # Сохраняем оригинал отдельного фото
            single_filename = f"images_from_bot/originals/{album_filename[:-4]}_{idx + 1}.jpg"
            cv2.imwrite(single_filename, image)

            # Инференс без пост-обработки (чистые probs)
            inp, _ = prepare_for_model(image)
            outputs = ort_session.run([ORT_OUTPUT_NAME], {ORT_INPUT_NAME: inp})
            logits = outputs[0][0]
            probs = softmax(logits)

            probs_list.append(probs)

            top_class = CLASS_NAMES[int(np.argmax(probs))]
            per_photo_classes.append(top_class)

        except Exception as e:
            print(f"Ошибка в альбоме, фото {idx + 1}: {e}")

    if len(probs_list) == 0:
        await context.bot.send_message(chat_id=chat_id, text="❌ Не удалось обработать ни одно фото из альбома.")
        return

    # Усреднение вероятностей
    avg_probs = np.mean(np.stack(probs_list, axis=0), axis=0)

    # Определяем результат из усреднённых probs
    sorted_idx = np.argsort(avg_probs)[::-1]
    top1_idx = int(sorted_idx[0])
    top2_idx = int(sorted_idx[1])

    top1_class = CLASS_NAMES[top1_idx]
    top2_class = CLASS_NAMES[top2_idx]
    top1_conf = float(avg_probs[top1_idx])
    top2_conf = float(avg_probs[top2_idx])
    margin = top1_conf - top2_conf

    skin_idx = CLASS_NAMES.index("skin")
    skin_conf = float(avg_probs[skin_idx])

    disagreement = len(set(per_photo_classes)) > 1
    is_uncertain = disagreement or (top1_conf < CONFIDENCE_THRESHOLD) or (margin < MARGIN_THRESHOLD)

    # Skin-safe для альбома
    if top1_class != "skin":
        if top1_conf < DISEASE_STRONG_THRESHOLD and skin_conf >= SKIN_OVERRIDE_MIN:
            if (top2_class == "skin") or (margin < DISEASE_MARGIN_STRONG):
                top2_class = top1_class
                top2_conf = top1_conf
                top1_class = "skin"
                top1_conf = skin_conf
                is_uncertain = True

    # Дополнительный skin-bias: если половина фото = skin
    skin_votes = sum(1 for c in per_photo_classes if c == "skin")
    if skin_votes >= (total + 1) // 2:
        if top1_class != "skin" and (top1_conf < 0.95 or margin < 0.45):
            top2_class = top1_class
            top2_conf = top1_conf
            top1_class = "skin"
            top1_conf = skin_conf
            is_uncertain = True

    confusion_warning = None
    pair = frozenset({top1_class, top2_class})
    if margin < CONFUSION_MARGIN and pair in CONFUSED_PAIRS:
        confusion_warning = CONFUSED_PAIRS[pair]
        is_uncertain = True

    # Формируем album_warning
    album_warning = None
    if disagreement:
        pretty = []
        for i, cls_name in enumerate(per_photo_classes, start=1):
            pretty.append(f"{i}) {DIAGNOSES_MAP.get(cls_name, cls_name)}")
        album_warning = (
                "⚠️ Фото дали разные результаты. Итог — по *среднему*.\n\n"
                "По каждому фото:\n" + "\n".join(pretty)
        )

    if album_warning and confusion_warning:
        confusion_warning = album_warning + "\n\n" + confusion_warning
    elif album_warning:
        confusion_warning = album_warning

    result = {
        "class": top1_class,
        "confidence": top1_conf,
        "top2_class": top2_class,
        "top2_conf": top2_conf,
        "margin": margin,
        "uncertain": is_uncertain,
        "confusion_warning": confusion_warning,
        "all_probs": {CLASS_NAMES[i]: float(avg_probs[i]) for i in range(len(CLASS_NAMES))},
    }

    predict_img = result["class"]
    probability = result["confidence"]

    print(f"АЛЬБОМ {total} фото | {predict_img} | {probability:.2%} | uncertain={result['uncertain']}")

    log_photo(album_filename, user_id, predict_img, probability)

    if "diagnosis_by_file" not in context.user_data:
        context.user_data["diagnosis_by_file"] = {}
    context.user_data["diagnosis_by_file"][album_filename] = predict_img

    header = f"*Анализ альбома: {total} фото*\n\n"
    await context.bot.send_message(
        chat_id=chat_id,
        text=header + build_reply_message(result),
        parse_mode="Markdown",
    )

    confirm_text = f"Альбом из {total} фото\n\n" + build_confirm_text(predict_img, result["uncertain"])

    keyboard = [
        [InlineKeyboardButton("Диагноз подтвердился", callback_data=f"confirm::{album_filename}")],
        [InlineKeyboardButton("Диагноз не тот", callback_data=f"reject::{album_filename}")],
    ]

    await context.bot.send_message(
        chat_id=chat_id,
        text=confirm_text,
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def flush_media_group(context, key, chat_id, user_id):
    """Ждёт пока Telegram дошлёт все фото альбома, потом обрабатывает."""
    try:
        await asyncio.sleep(MEDIA_GROUP_DELAY)

        photo_ids = MEDIA_GROUP_BUFFER.pop(key, [])
        MEDIA_GROUP_TASKS.pop(key, None)

        if not photo_ids:
            return

        total = min(len(photo_ids), MAX_PHOTOS_PER_MESSAGE)

        if total > 1:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"📷 Получено {total} фото. Считаю *единый* результат по альбому...",
                parse_mode="Markdown",
            )
            await process_album_group(context, photo_ids[:MAX_PHOTOS_PER_MESSAGE], chat_id, user_id)
        else:
            await process_single_photo(
                context=context,
                photo_file_id=photo_ids[0],
                photo_index=0,
                total_photos=1,
                chat_id=chat_id,
                user_id=user_id,
            )

    except asyncio.CancelledError:
        return
    except Exception as e:
        print(f"Ошибка flush_media_group: {e}")
        import traceback

        traceback.print_exc()


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global img_count

    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    if user_id not in agreed_users:
        await send_agreement(update, context)
        return

    print(f"Получил изображение от {user_id}. В очереди {img_count}")
    img_count += 1
    log_request(user_id, "image")

    try:
        photo = update.message.photo[-1]
        media_group_id = update.message.media_group_id

        if media_group_id:
            key = f"{chat_id}_{media_group_id}"

            if key not in MEDIA_GROUP_BUFFER:
                MEDIA_GROUP_BUFFER[key] = []

            MEDIA_GROUP_BUFFER[key].append(photo.file_id)

            if key in MEDIA_GROUP_TASKS:
                task = MEDIA_GROUP_TASKS[key]
                if not task.done():
                    task.cancel()

            MEDIA_GROUP_TASKS[key] = asyncio.create_task(flush_media_group(context, key, chat_id, user_id))

        else:
            await process_single_photo(
                context=context,
                photo_file_id=photo.file_id,
                photo_index=0,
                total_photos=1,
                chat_id=chat_id,
                user_id=user_id,
            )

    except TimedOut:
        await update.message.reply_text(
            "⏳ Сервер не успел получить фотографию — это временная проблема.\n\n"
            "Пожалуйста, отправьте фото ещё раз."
        )
    except Exception as e:
        print(f"Ошибка при обработке изображения: {e}")
        import traceback

        traceback.print_exc()
        await update.message.reply_text(
            "К сожалению, не удалось обработать изображение. "
            "Попробуйте отправить другое или через пару минут.\n\n"
            f"Ошибка: {e}"
        )
    finally:
        img_count -= 1


# ===================== CALLBACK КНОПКИ =====================


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id

    if query.data.startswith("admin::"):
        if query.data.startswith("admin::photos_date::"):
            await admin_photos_date_callback(update, context)
        else:
            await admin_callback_handler(update, context)
        return

    if user_id not in agreed_users:
        await query.answer("Сначала необходимо согласиться с пользовательским соглашением!", show_alert=True)
        return

    await query.answer()

    parts = query.data.split("::")

    if len(parts) < 2:
        print(f"Ошибка: некорректный callback_data: {query.data}")
        return

    action = parts[0]

    if action == "confirm":
        filename = parts[1]
        filepath = f"images_from_bot/{filename}"

        diagnosis_map = context.user_data.get("diagnosis_by_file", {})
        original_diagnosis = diagnosis_map.get(filename, "unknown")

        save_diagnosis(filepath, original_diagnosis, confirmed_by_model=True)

        await query.edit_message_text(
            text="Спасибо за ваш ответ. Это поможет нам улучшать приложение!", reply_markup=None
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
            text="Пожалуйста, выберите правильный диагноз из списка:", reply_markup=reply_markup
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
            text="Выберите наиболее подходящий вариант из распространенных диагнозов:", reply_markup=reply_markup
        )

    elif action == "select":
        if len(parts) < 3:
            return

        key = parts[1]
        filename = parts[2]
        filepath = f"images_from_bot/{filename}"

        save_diagnosis(filepath, key, confirmed_by_model=False)

        await query.edit_message_text(
            text="Спасибо за ваш ответ. Это поможет нам улучшать приложение!", reply_markup=None
        )

    elif action == "popular":
        if len(parts) < 3:
            return

        key = parts[1]
        filename = parts[2]
        filepath = f"images_from_bot/{filename}"

        save_diagnosis(filepath, key, confirmed_by_model=False)

        await query.edit_message_text(
            text="Спасибо за ваш ответ. Это поможет нам улучшать приложение!", reply_markup=None
        )


# ===================== ЗАПУСК =====================

application = ApplicationBuilder().token(TOKEN).connect_timeout(30).read_timeout(30).write_timeout(30).build()

application.add_handler(CommandHandler("start", start))
application.add_handler(CommandHandler("help", help_command))
application.add_handler(CommandHandler("photo", photo_command))
application.add_handler(CallbackQueryHandler(agreement_handler, pattern="^(agree|disagree)$"))
application.add_handler(CallbackQueryHandler(button_callback))
application.add_handler(MessageHandler(filters.PHOTO, handle_image))
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

print("Бот запущен...")
application.run_polling()
