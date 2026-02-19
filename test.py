from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime
import json
import os

# Для локального использования бота введите токен бота, получаемый через BotFather.
TOKEN = "PASTE_BOT_TOKEN_HERE"
img_count = 0

# Файл для хранения ID пользователей, согласившихся с соглашением
AGREED_USERS_FILE = "agreed_users.json"

# Загружаем список согласившихся пользователей
agreed_users = set()
if os.path.exists(AGREED_USERS_FILE):
    try:
        with open(AGREED_USERS_FILE, 'r') as f:
            agreed_users = set(json.load(f))
    except:
        agreed_users = set()

DIAGNOSES_MAP = {
    "panu": "Разноцветный лишай",
    "rosacea": "Розацеа",
    "herpes": "Герпес",
    "eksim": "Экзема",
    "acne": "Акне",
    "skin": "Здоровая кожа",
    "other": "Больше болезней"
}

# Расширенный список популярных кожных заболеваний
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

DIAGNOSES_FILE = "diagnoses.json"

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
    """Сохраняет ID пользователя в список согласившихся"""
    agreed_users.add(user_id)
    with open(AGREED_USERS_FILE, 'w') as f:
        json.dump(list(agreed_users), f)


async def send_agreement(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отправляет пользовательское соглашение с кнопками"""
    keyboard = [
        [InlineKeyboardButton("✅ Согласен", callback_data="agree")],
        [InlineKeyboardButton("❌ Не согласен", callback_data="disagree")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    agreement_text = "⚠️ Пожалуйста, ознакомьтесь с пользовательским соглашением перед использованием бота:\n\n" + USELESS_TEXT

    await update.message.reply_text(
        agreement_text,
        reply_markup=reply_markup,
        parse_mode=None
    )


async def agreement_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик кнопок соглашения"""
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id

    if query.data == "agree":
        save_agreed_user(user_id)
        await query.edit_message_text(
            "✅ Спасибо за согласие! Теперь вы можете пользоваться всеми функциями бота.",
            reply_markup=None
        )
        # Отправляем приветственное сообщение
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
            "❌ Без согласия с пользовательским соглашением вы не сможете пользоваться ботом.\n"
            "Если вы передумаете, нажмите /start для повторного просмотра соглашения.",
            reply_markup=None
        )


async def check_agreement(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Проверяет согласие пользователя перед выполнением команд"""
    user_id = update.effective_user.id
    if user_id not in agreed_users:
        await send_agreement(update, context)
        return False
    return True


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    print("Получил /start.")
    user_id = update.effective_user.id

    # Если пользователь уже согласился - показываем обычное приветствие
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
        # Показываем соглашение для новых пользователей
        await send_agreement(update, context)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /help с проверкой соглашения"""
    print("Получил /help.")
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
    """Обработчик команды /photo с проверкой соглашения"""
    print("Получил /photo.")
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
    """Сохраняет диагноз в JSON файл"""
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
    except Exception as e:
        print(f"Ошибка при сохранении диагноза: {e}")


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик нажатий на кнопки диагнозов (требует согласия)"""
    query = update.callback_query
    user_id = query.from_user.id

    # Проверяем согласие перед обработкой кнопок
    if user_id not in agreed_users:
        await query.answer("⚠️ Сначала необходимо согласиться с пользовательским соглашением!", show_alert=True)
        await send_agreement(update, context)
        return

    await query.answer()

    parts = query.data.split('::')

    if len(parts) < 2:
        # Некорректные данные
        print(f"Ошибка: некорректный callback_data: {query.data}")
        return

    action = parts[0]

    if action == "confirm":
        filename = parts[1]
        filepath = f"images_from_bot/{filename}"

        # Диагноз подтвердился - сохраняем исходный диагноз
        original_diagnosis = context.user_data.get('last_diagnosis', 'unknown')
        save_diagnosis(filepath, original_diagnosis, confirmed_by_model=True)

        await query.edit_message_text(
            text="Спасибо за ваш ответ. Это поможет нам улучшать приложение!",
            reply_markup=None
        )

    elif action == "reject":
        filename = parts[1]

        # Диагноз не тот - показываем кнопки для выбора правильного диагноза
        keyboard = []

        # Получаем все диагнозы из основного списка
        diagnoses_items = list(DIAGNOSES_MAP.items())

        # Группируем по 2 диагноза в строке
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
            print(f"Ошибка: некорректный callback_data для 'other': {query.data}")
            return

        key = parts[1]
        filename = parts[2]

        # Пользователь выбрал "Болезни нет в списке" из основного списка
        # Показываем список популярных диагнозов
        keyboard = []

        # Создаем кнопки для каждого популярного диагноза (по одной на строку)
        for popular_key, value in POPULAR_DIAGNOSES.items():
            keyboard.append([InlineKeyboardButton(value, callback_data=f"popular::{popular_key}::{filename}")])

        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_text(
            text="Выберите наиболее подходящий вариант из распространенных диагнозов:",
            reply_markup=reply_markup
        )

    elif action == "select":
        if len(parts) < 3:
            print(f"Ошибка: некорректный callback_data для 'select': {query.data}")
            return

        key = parts[1]
        filename = parts[2]
        filepath = f"images_from_bot/{filename}"

        # Пользователь выбрал диагноз из основного списка (кроме "Болезни нет в списке")
        save_diagnosis(filepath, key, confirmed_by_model=False)

        await query.edit_message_text(
            text="Спасибо за ваш ответ. Это поможет нам улучшать приложение!",
            reply_markup=None
        )

    elif action == "popular":
        # Формат: popular::key::filename
        if len(parts) < 3:
            print(f"Ошибка: некорректный callback_data для 'popular': {query.data}")
            return

        key = parts[1]
        filename = parts[2]
        filepath = f"images_from_bot/{filename}"

        # Пользователь выбрал диагноз из списка популярных
        save_diagnosis(filepath, key, confirmed_by_model=False)

        await query.edit_message_text(
            text="Спасибо за ваш ответ. Это поможет нам улучшать приложение!",
            reply_markup=None
        )


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик изображений с проверкой соглашения"""
    global img_count

    # Проверяем согласие перед обработкой изображения
    user_id = update.effective_user.id
    if user_id not in agreed_users:
        await send_agreement(update, context)
        return

    print(f"Получил изображение. В очереди {img_count}")
    img_count += 1

    try:
        photo_file = await update.message.photo[-1].get_file()
        photo_bytes = await photo_file.download_as_bytearray()
        nparr = np.frombuffer(photo_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

        def process_image_for_model(img, target_size=244, scale=1.25):
            """Обрабатывает изображение: увеличение + обрезка центра"""
            h, w = img.shape[:2]
            new_w, new_h = int(w * scale), int(w * scale)
            scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

            center_x, center_y = new_w // 2, new_h // 2
            half_target = target_size // 2

            x1 = max(0, center_x - half_target)
            y1 = max(0, center_y - half_target)
            x2 = min(new_w, x1 + target_size)
            y2 = min(new_h, y1 + target_size)

            if x2 - x1 < target_size:
                x1 = max(0, new_w - target_size)
                x2 = new_w
            if y2 - y1 < target_size:
                y1 = max(0, new_h - target_size)
                y2 = new_h

            cropped = scaled[y1:y2, x1:x2]

            if cropped.shape[0] != target_size or cropped.shape[1] != target_size:
                cropped = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)

            return cropped

        processed_image = process_image_for_model(image, target_size=244, scale=0.7)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"image_{timestamp}.jpg"
        filepath = f"images_from_bot/{filename}"
        cv2.imwrite(filepath, processed_image)

        result = model(processed_image)[0]
        predict_img = result.names[result.probs.top5[0]]
        probability = float(result.probs.top1conf)

        print(f"Класс: {predict_img}, уверенность: {probability}")

        context.user_data['last_filename'] = filepath
        context.user_data['last_diagnosis'] = predict_img

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
                "- *Профилактика:* Следует соблюдать гигиену кожи, избегать выдавливания прыщей, использовать правильно подобранные косметические средства (не содержащие спирта), соблюдать сбалансированное питание и поддерживать здоровый образ жизни. Не прикасаться к лицу грязными руками в течение дня    "
                ".\n\n"
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
        await update.message.reply_text(
            "К сожалению, не удалось обработать изображение. Пожалуйста, попробуйте отправить другое изображение или через пару минут.\n\n"
            f"Ошибка: {e}")

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

print("Бот запущен...")
application.run_polling()