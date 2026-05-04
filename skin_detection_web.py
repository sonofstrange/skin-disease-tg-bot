from flask import (Flask, render_template, request, redirect,
                   url_for, session, send_file)
import onnxruntime as ort
from PIL import Image, ImageOps
import cv2
import numpy as np
from datetime import datetime, timedelta
import json
import os
import io
import base64
from collections import Counter
import uuid

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = "change_me_secret_key_12345"

AGREED_USERS_FILE = "agreed_users.json"
DIAGNOSES_FILE = "diagnoses.json"
STATS_FILE = "stats.json"
ADMIN_LOGIN = "adm"
ADMIN_PASSWORD = "123"
IMG_DIR = "images_from_bot"
os.makedirs(IMG_DIR, exist_ok=True)

ONNX_MODEL_PATH = "best_model.onnx"
IMG_SIZE = 244
CLASS_NAMES = ["acne", "eksim", "herpes", "panu", "rosacea", "skin"]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CONFIDENCE_THRESHOLD = 0.55
MARGIN_THRESHOLD = 0.15
CONFUSION_MARGIN = 0.20
SKIN_OVERRIDE_MIN = 0.25
DISEASE_STRONG_THRESHOLD = 0.75
DISEASE_MARGIN_STRONG = 0.25

CONFUSED_PAIRS = {
    frozenset({"eksim", "rosacea"}): "Модель затрудняется отличить Экзему от Розацеа. Рекомендуем обратиться к дерматологу.",
    frozenset({"acne", "rosacea"}): "Модель затрудняется отличить Акне от Розацеа. Рекомендуем обратиться к дерматологу.",
    frozenset({"eksim", "herpes"}): "Модель затрудняется отличить Экзему от Герпеса. Рекомендуем обратиться к дерматологу.",
}

print(f"Загрузка модели: {ONNX_MODEL_PATH}")
ort_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
ORT_INPUT_NAME = ort_session.get_inputs()[0].name
ORT_OUTPUT_NAME = ort_session.get_outputs()[0].name
print("Модель загружена")


def load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return default


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


agreed_users = set(load_json(AGREED_USERS_FILE, []))
stats_data = load_json(STATS_FILE, {"requests": [], "photos": []})
diagnoses_list = load_json(DIAGNOSES_FILE, [])


def save_agreed():
    save_json(AGREED_USERS_FILE, list(agreed_users))


def save_stats():
    save_json(STATS_FILE, stats_data)


def save_diagnoses():
    save_json(DIAGNOSES_FILE, diagnoses_list)


def log_request(uid, rtype):
    stats_data["requests"].append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_id": uid, "type": rtype,
    })
    save_stats()


def log_photo(filename, uid, ai_pred, ai_conf):
    stats_data["photos"].append({
        "filename": filename,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_id": uid, "ai_prediction": ai_pred,
        "ai_confidence": ai_conf, "user_verdict": None,
    })
    save_stats()


def update_photo_verdict(filename, verdict):
    base = os.path.basename(filename)
    for p in stats_data["photos"]:
        if os.path.basename(p["filename"]) == base:
            p["user_verdict"] = verdict
            save_stats()
            return


DIAGNOSES_MAP = {
    "panu": "Разноцветный лишай",
    "rosacea": "Розацеа",
    "herpes": "Герпес",
    "eksim": "Экзема",
    "acne": "Акне",
    "skin": "Здоровая кожа",
}

POPULAR_DIAGNOSES = {
    "dermatitis": "Дерматит",
    "chickenpox": "Ветрянка",
    "scabies": "Чесотка",
    "papilloma": "Папиллома",
    "psoriasis": "Псориаз",
    "papillomas_and_warts": "Папилломы и бородавки",
    "burns": "Ожог",
    "allergy": "Аллергическая реакция",
    "other_popular": "Болезни нет в списке",
}

ALL_DIAGNOSES = {**DIAGNOSES_MAP, **POPULAR_DIAGNOSES}

DISEASE_INFO = {
    "panu": {
        "name": "Разноцветный лишай",
        "description": "Грибковая инфекция, проявляется в виде шелушащихся пятен белого, розового или коричневого цвета. Не заразна.",
        "recommendations": "Противогрибковые шампуни, лосьоны или кремы (кетоконазол, клотримазол). В сложных случаях — таблетки по назначению врача.",
        "prevention": "Соблюдайте правила личной гигиены, носите свободную одежду из натуральных тканей, регулярно принимайте душ.",
        "severity": "Низкая. Не опасен для здоровья, но требует лечения.",
        "warning": "ИИ часто выводит данный результат на здоровую кожу. Если вы уверены в диагнозе — обратитесь к дерматологу.",
    },
    "rosacea": {
        "name": "Розацеа",
        "description": "Хроническое заболевание кожи лица. Покраснение, сосудистые звездочки, бугорки и гнойнички.",
        "recommendations": "Избегать триггеров (острая еда, алкоголь, солнце). Наружные гели (метронидазол) или антибиотики по рецепту.",
        "prevention": "Избегайте провоцирующих факторов, используйте щадящий уход и защиту от солнца.",
        "severity": "Низкая. Хронический характер, влияет на внешность.",
        "warning": "Рекомендуем обратиться к дерматологу. ИИ может ошибаться.",
    },
    "herpes": {
        "name": "Герпес",
        "description": "Вирусная инфекция. Пузырьки на коже (чаще на губах), зуд, покраснение, затем язвочки с корочкой.",
        "recommendations": "Противовирусные мази (ацикловир) и таблетки. Начинать лечение при первых симптомах.",
        "prevention": "Личная гигиена, не трогайте высыпания, индивидуальные предметы гигиены. Укрепляйте иммунитет.",
        "severity": "Средняя. Опасен для людей с иммунодефицитом.",
        "warning": "Рекомендуем обратиться к дерматологу. ИИ может ошибаться.",
    },
    "eksim": {
        "name": "Экзема",
        "description": "Хроническое незаразное воспалительное заболевание кожи: зуд, покраснение, отечность, высыпания.",
        "recommendations": "Эмоленты, кортикостероиды для снятия воспаления, антигистаминные от зуда.",
        "prevention": "Устранение триггеров (аллергены, стресс). Увлажняющие кремы, одежда из натуральных тканей.",
        "severity": "Средняя. Не заразна, но ухудшает качество жизни.",
        "warning": "Рекомендуем обратиться к дерматологу. ИИ может ошибаться.",
    },
    "acne": {
        "name": "Акне",
        "description": "Воспаление сальных желез: черные точки, красные прыщики, гнойнички.",
        "recommendations": "Салициловая кислота, бензоилпероксид. Ретиноиды и антибиотики по назначению врача.",
        "prevention": "Гигиена кожи, не выдавливайте прыщи, правильная косметика, сбалансированное питание.",
        "severity": "Низкая. Тяжелые формы могут оставлять рубцы.",
        "warning": "Рекомендуем обратиться к дерматологу. ИИ может ошибаться.",
    },
    "skin": {
        "name": "Здоровая кожа",
        "description": "Признаков заболевания не обнаружено.",
        "recommendations": "Продолжайте ухаживать за кожей.",
        "prevention": "Правильное питание, защита от УФ, достаточный сон.",
        "severity": "Нет.",
        "warning": "Если есть подозрения — обратитесь к дерматологу. ИИ может ошибаться.",
    },
}

AGREEMENT_TEXT = """ПОЛЬЗОВАТЕЛЬСКОЕ СОГЛАШЕНИЕ
для сервиса предварительной оценки состояния кожи

1. Общие положения
1.1. Настоящее соглашение регулирует отношения между Разработчиком и Пользователем сервиса CheckSkin.
1.2. Используя Сервис, Пользователь подтверждает согласие со всеми условиями.

2. Характер Сервиса
2.1. Сервис предоставляет предварительную информацию на основе машинного обучения и не является медицинским сервисом.
2.2. Окончательный диагноз может поставить только врач-дерматолог.

3. Обработка данных
3.1. Отправляя фотографию, Пользователь даёт согласие на обработку и хранение изображения.
3.2. Цели: анализ, улучшение алгоритма, техническое хранение.
3.3. Для обучения модели используется обезличенный набор данных.

4. Права Пользователя
4.1. Пользователь может отозвать согласие, написав на strange.z2tablet@gmail.com. Данные удаляются в течение 30 дней.

5. Возрастное ограничение — 16 лет.

6. Разработчик вправе вносить изменения в Соглашение.

Контакт: strange.z2tablet@gmail.com"""


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def prepare_for_model(img_bgr):
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
    pil_cropped = pil_img.crop((left, top, left + IMG_SIZE, top + IMG_SIZE))

    arr = np.array(pil_cropped, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)

    return arr.astype(np.float32), pil_cropped


def run_inference(img_bgr):
    inp, pil_cropped = prepare_for_model(img_bgr)

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

    if top1_class != "skin" and top2_class == "skin":
        if (top1_conf < DISEASE_STRONG_THRESHOLD
                and margin < DISEASE_MARGIN_STRONG
                and skin_conf >= SKIN_OVERRIDE_MIN):
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
        "pil_cropped": pil_cropped,
    }


def run_multi_inference(images_bgr):
    """Усреднение вероятностей по нескольким фото."""
    all_probs = []
    pil_crops = []

    for img in images_bgr:
        inp, pil_cropped = prepare_for_model(img)
        outputs = ort_session.run([ORT_OUTPUT_NAME], {ORT_INPUT_NAME: inp})
        logits = outputs[0][0]
        probs = softmax(logits)
        all_probs.append(probs)
        pil_crops.append(pil_cropped)

    avg_probs = np.mean(np.stack(all_probs, axis=0), axis=0)

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

    is_uncertain = (top1_conf < CONFIDENCE_THRESHOLD) or (margin < MARGIN_THRESHOLD)

    if top1_class != "skin" and top2_class == "skin":
        if (top1_conf < DISEASE_STRONG_THRESHOLD
                and margin < DISEASE_MARGIN_STRONG
                and skin_conf >= SKIN_OVERRIDE_MIN):
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
        "all_probs": {CLASS_NAMES[i]: float(avg_probs[i]) for i in range(len(CLASS_NAMES))},
        "pil_crops": pil_crops,
        "count": len(images_bgr),
    }


def buf_to_b64(buf):
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def chart_daily(days=7):
    end = datetime.now()
    start = end - timedelta(days=days)
    dr, dp = Counter(), Counter()
    for r in stats_data["requests"]:
        try:
            d = datetime.strptime(r["timestamp"], "%Y-%m-%d %H:%M:%S").date()
            if start.date() <= d <= end.date():
                dr[d] += 1
        except:
            pass
    for p in stats_data["photos"]:
        try:
            d = datetime.strptime(p["timestamp"], "%Y-%m-%d %H:%M:%S").date()
            if start.date() <= d <= end.date():
                dp[d] += 1
        except:
            pass

    dates = []
    cur = start
    while cur <= end:
        dates.append(cur.date())
        cur += timedelta(days=1)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    x = range(len(dates))
    w = 0.35
    ax.bar([i - w / 2 for i in x], [dr.get(d, 0) for d in dates], w, label="Запросы", color="#4a90d9")
    ax.bar([i + w / 2 for i in x], [dp.get(d, 0) for d in dates], w, label="Фото", color="#d94a4a")
    ax.set_xticks(list(x))
    ax.set_xticklabels([d.strftime("%d.%m") for d in dates], rotation=45, fontsize=8)
    ax.set_title(f"За последние {days} дн.", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches='tight')
    plt.close()
    return buf_to_b64(buf)


def chart_pie():
    ai_c = Counter(p.get("ai_prediction", "?") for p in stats_data["photos"])
    uv_c = Counter(p["user_verdict"] for p in stats_data["photos"] if p.get("user_verdict"))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.5))
    if ai_c:
        a1.pie(list(ai_c.values()),
               labels=[ALL_DIAGNOSES.get(k, k) for k in ai_c],
               autopct="%1.0f%%", startangle=90, textprops={'fontsize': 7})
        a1.set_title("Предсказания ИИ", fontsize=10)
    else:
        a1.text(0.5, 0.5, "Нет данных", ha="center")
        a1.axis("off")
    if uv_c:
        a2.pie(list(uv_c.values()),
               labels=[ALL_DIAGNOSES.get(k, k) for k in uv_c],
               autopct="%1.0f%%", startangle=90, textprops={'fontsize': 7})
        a2.set_title("Вердикты", fontsize=10)
    else:
        a2.text(0.5, 0.5, "Нет вердиктов", ha="center")
        a2.axis("off")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches='tight')
    plt.close()
    return buf_to_b64(buf)


def chart_hourly():
    hc = Counter()
    for r in stats_data["requests"]:
        try:
            hc[datetime.strptime(r["timestamp"], "%Y-%m-%d %H:%M:%S").hour] += 1
        except:
            pass
    fig, ax = plt.subplots(figsize=(8, 3))
    hours = list(range(24))
    ax.bar(hours, [hc.get(h, 0) for h in hours], color="#8e44ad")
    ax.set_xlabel("Час", fontsize=8)
    ax.set_ylabel("Запросов", fontsize=8)
    ax.set_title("Активность по часам", fontsize=10)
    ax.set_xticks(hours)
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches='tight')
    plt.close()
    return buf_to_b64(buf)


def chart_accuracy():
    c = r = n = 0
    for p in stats_data["photos"]:
        uv = p.get("user_verdict")
        if uv is None:
            n += 1
        elif uv == p.get("ai_prediction"):
            c += 1
        else:
            r += 1
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.pie([c, r, n], labels=["Подтв.", "Откл.", "Без отв."],
           colors=["#27ae60", "#e74c3c", "#95a5a6"], autopct="%1.0f%%",
           explode=(0.04, 0.04, 0), shadow=True, startangle=90,
           textprops={'fontsize': 8})
    ax.set_title("Точность ИИ", fontsize=10)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches='tight')
    plt.close()
    return buf_to_b64(buf)


def get_uid():
    if "user_id" not in session:
        session["user_id"] = str(uuid.uuid4())
    return session["user_id"]


def is_agreed():
    return get_uid() in agreed_users


def is_admin():
    return session.get("is_admin", False)


@app.route("/")
def index():
    log_request(get_uid(), "visit")
    if not is_agreed():
        return redirect(url_for("agreement"))
    return redirect(url_for("dashboard"))


@app.route("/agreement", methods=["GET", "POST"])
def agreement():
    if request.method == "POST":
        if request.form.get("choice") == "agree":
            agreed_users.add(get_uid())
            save_agreed()
            return redirect(url_for("dashboard"))
        return render_template("agreement.html", text=AGREEMENT_TEXT,
                               error="Без согласия использование сервиса невозможно.")
    return render_template("agreement.html", text=AGREEMENT_TEXT, error=None)


@app.route("/dashboard")
def dashboard():
    if not is_agreed():
        return redirect(url_for("agreement"))
    return render_template("dashboard.html")


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if not is_agreed():
        return redirect(url_for("agreement"))

    if request.method == "POST":
        uid = get_uid()
        log_request(uid, "image")

        files = request.files.getlist("images")
        if not files or all(f.filename == "" for f in files):
            return render_template("upload.html", error="Файл не выбран")

        try:
            images_bgr = []
            for f in files[:4]:
                if f.filename == "":
                    continue
                raw = np.frombuffer(f.read(), np.uint8)
                img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                if img is not None:
                    images_bgr.append(img)

            if not images_bgr:
                return render_template("upload.html", error="Не удалось прочитать ни одно изображение")

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(os.path.join(IMG_DIR, "originals"), exist_ok=True)
            os.makedirs(os.path.join(IMG_DIR, "processed"), exist_ok=True)

            if len(images_bgr) == 1:
                result = run_inference(images_bgr[0])
                pil_cropped = result.pop("pil_cropped")

                filename = f"image_{ts}.jpg"
                cv2.imwrite(os.path.join(IMG_DIR, "originals", filename), images_bgr[0])
                pil_cropped.save(os.path.join(IMG_DIR, "processed", filename), quality=95)
                pil_cropped.save(os.path.join(IMG_DIR, filename), quality=95)

                log_photo(filename, uid, result["class"], result["confidence"])
                session["last_filename"] = filename
                session["last_diagnosis"] = result["class"]
                photo_count = 1

            else:
                result = run_multi_inference(images_bgr)
                pil_crops = result.pop("pil_crops")
                photo_count = result.pop("count")

                filename = f"album_{ts}.jpg"
                for idx, (img, crop) in enumerate(zip(images_bgr, pil_crops)):
                    single_name = f"album_{ts}_{idx + 1}.jpg"
                    cv2.imwrite(os.path.join(IMG_DIR, "originals", single_name), img)
                    crop.save(os.path.join(IMG_DIR, "processed", single_name), quality=95)

                pil_crops[0].save(os.path.join(IMG_DIR, filename), quality=95)
                log_photo(filename, uid, result["class"], result["confidence"])
                session["last_filename"] = filename
                session["last_diagnosis"] = result["class"]

            pred = result["class"]
            prob = result["confidence"]

            info = DISEASE_INFO.get(pred, {
                "name": ALL_DIAGNOSES.get(pred, pred),
                "description": "Нет данных.",
                "recommendations": "Обратитесь к дерматологу.",
                "prevention": "—", "severity": "—",
                "warning": "ИИ может ошибаться.",
            })

            pct = round(prob * 100, 1)
            level = "high" if pct >= 75 else ("med" if pct >= 50 else "low")

            return render_template(
                "result.html",
                info=info, pct=pct, level=level,
                filename=filename, prediction=pred,
                uncertain=result["uncertain"],
                confusion_warning=result.get("confusion_warning"),
                top2_class=ALL_DIAGNOSES.get(result["top2_class"], result["top2_class"]),
                top2_conf=round(result["top2_conf"] * 100, 1),
                photo_count=photo_count,
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            return render_template("upload.html", error=str(e))

    return render_template("upload.html", error=None)


@app.route("/verdict/<action>", methods=["GET", "POST"])
def verdict(action):
    if not is_agreed():
        return redirect(url_for("agreement"))

    filename = request.args.get("f", session.get("last_filename", ""))

    if action == "confirm":
        diagnosis = session.get("last_diagnosis", "unknown")
        diagnoses_list.append({
            "filename": os.path.join(IMG_DIR, filename),
            "diagnosis": diagnosis, "confirmed_by_model": True,
        })
        save_diagnoses()
        update_photo_verdict(filename, diagnosis)
        return render_template("verdict_thanks.html")

    elif action == "select":
        chosen = request.args.get("d", "")
        if chosen:
            diagnoses_list.append({
                "filename": os.path.join(IMG_DIR, filename),
                "diagnosis": chosen, "confirmed_by_model": False,
            })
            save_diagnoses()
            update_photo_verdict(filename, chosen)
            return render_template("verdict_thanks.html")
        return render_template("verdict_select.html",
                               diagnoses=DIAGNOSES_MAP,
                               popular=POPULAR_DIAGNOSES,
                               filename=filename)

    return redirect(url_for("dashboard"))


@app.route("/img/<filename>")
def serve_image(filename):
    path = os.path.join(IMG_DIR, filename)
    if os.path.exists(path):
        return send_file(path)
    return "Not found", 404


@app.route("/help")
def help_page():
    if not is_agreed():
        return redirect(url_for("agreement"))
    log_request(get_uid(), "help")
    return render_template("help.html")


@app.route("/photo-guide")
def photo_guide():
    if not is_agreed():
        return redirect(url_for("agreement"))
    log_request(get_uid(), "photo_help")
    return render_template("photo_guide.html")


@app.route("/admin", methods=["GET", "POST"])
def admin_login():
    if is_admin():
        return redirect(url_for("admin_panel"))
    error = None
    if request.method == "POST":
        if (request.form.get("login") == ADMIN_LOGIN and
                request.form.get("password") == ADMIN_PASSWORD):
            session["is_admin"] = True
            return redirect(url_for("admin_panel"))
        error = "Неверный логин или пароль"
    return render_template("admin_login.html", error=error)


@app.route("/admin/logout")
def admin_logout():
    session.pop("is_admin", None)
    return redirect(url_for("admin_login"))


@app.route("/admin/panel")
def admin_panel():
    if not is_admin():
        return redirect(url_for("admin_login"))

    period = request.args.get("period", "week")
    status_filter = request.args.get("status", "all")
    limit = int(request.args.get("limit", 10))
    date_start = request.args.get("date_start", "")
    date_end = request.args.get("date_end", "")

    total_users = len(agreed_users)
    total_requests = len(stats_data["requests"])
    total_photos = len(stats_data["photos"])

    confirmed = sum(1 for p in stats_data["photos"]
                    if p.get("user_verdict") and p["user_verdict"] == p.get("ai_prediction"))
    rejected = sum(1 for p in stats_data["photos"]
                   if p.get("user_verdict") and p["user_verdict"] != p.get("ai_prediction"))
    no_feedback = sum(1 for p in stats_data["photos"] if not p.get("user_verdict"))
    accuracy = round(confirmed / (confirmed + rejected) * 100, 1) if (confirmed + rejected) > 0 else 0

    today = datetime.now().date().strftime("%Y-%m-%d")
    today_req = sum(1 for r in stats_data["requests"] if r.get("timestamp", "").startswith(today))
    today_photos = sum(1 for p in stats_data["photos"] if p.get("timestamp", "").startswith(today))

    days = {"today": 1, "week": 7, "month": 30}.get(period, 7)
    c_daily = chart_daily(days)
    c_pie = chart_pie()
    c_hourly = chart_hourly()
    c_accuracy = chart_accuracy()

    photos = stats_data["photos"].copy()

    if date_start and date_end:
        try:
            sd = datetime.strptime(date_start, "%d.%m.%Y")
            ed = datetime.strptime(date_end, "%d.%m.%Y").replace(hour=23, minute=59, second=59)
            photos = [p for p in photos
                      if sd <= datetime.strptime(p["timestamp"], "%Y-%m-%d %H:%M:%S") <= ed]
        except:
            pass

    if status_filter == "confirmed":
        photos = [p for p in photos if p.get("user_verdict") and p["user_verdict"] == p.get("ai_prediction")]
    elif status_filter == "rejected":
        photos = [p for p in photos if p.get("user_verdict") and p["user_verdict"] != p.get("ai_prediction")]
    elif status_filter == "pending":
        photos = [p for p in photos if not p.get("user_verdict")]

    photos.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    photos = photos[:limit]

    photo_list = []
    for p in photos:
        ai = p.get("ai_prediction", "")
        uv = p.get("user_verdict")
        if uv is None:
            st = "pending"
        elif uv == ai:
            st = "confirmed"
        else:
            st = "rejected"
        photo_list.append({
            "filename": os.path.basename(p.get("filename", "")),
            "timestamp": p.get("timestamp", ""),
            "user_id": p.get("user_id", ""),
            "ai_prediction": ALL_DIAGNOSES.get(ai, ai),
            "ai_confidence": round(p.get("ai_confidence", 0) * 100, 1),
            "user_verdict": ALL_DIAGNOSES.get(uv, uv) if uv else "Не оставлен",
            "status": st,
        })

    return render_template(
        "admin_panel.html",
        total_users=total_users, total_requests=total_requests,
        total_photos=total_photos, confirmed=confirmed,
        rejected=rejected, no_feedback=no_feedback,
        accuracy=accuracy, today_req=today_req,
        today_photos=today_photos,
        c_daily=c_daily, c_pie=c_pie,
        c_hourly=c_hourly, c_accuracy=c_accuracy,
        photos=photo_list, period=period,
        status_filter=status_filter, limit=limit,
        date_start=date_start, date_end=date_end,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)