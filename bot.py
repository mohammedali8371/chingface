import asyncio
import logging
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import insightface
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# =========================================================
# الإعدادات الثابتة (تم تضمين التوكن و ID المطور)
# =========================================================

BOT_TOKEN = "8363099113:AAEq9wPCFIbGfxw2M5s2_tHeywwWLKqi5i0"
DEVELOPER_ID = 7958260008

# المتغيرات البيئية (اختيارية)
PUBLIC_URL = os.getenv("PUBLIC_URL", "").rstrip("/")
WEBHOOK_PATH = os.environ.get("WEBHOOK_PATH", "/telegram/webhook")
PORT = int(os.environ.get("PORT", "10000"))

# حالة البوت
BOT_ENABLED = True

# الإحصائيات
USERS_SEEN = set()
TOTAL_IMAGES = 0
TOTAL_VIDEOS = 0

# مجلد العمل
WORK_DIR = Path(os.environ.get("WORK_DIR", "/tmp/faceswap"))
WORK_DIR.mkdir(parents=True, exist_ok=True)

# حدود الملفات
MAX_VIDEO_BYTES = 12 * 1024 * 1024
MAX_VIDEO_SECONDS = 20
MAX_VIDEO_WIDTH = 720
MAX_VIDEO_HEIGHT = 720

# =========================================================
# Logging
# =========================================================

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)

log = logging.getLogger("face-swap-bot")

# =========================================================
# AI Models
# =========================================================

face_app = None
swapper = None


def load_models():
    global face_app, swapper

    if face_app is not None and swapper is not None:
        return

    log.info("Loading InsightFace models...")

    face_app = FaceAnalysis(
        name="buffalo_s",
        providers=["CPUExecutionProvider"],
    )
    face_app.prepare(ctx_id=-1, det_size=(640, 640))

    try:
        swapper = get_model(
            "inswapper_128.onnx",
            download=True,
            providers=["CPUExecutionProvider"],
        )
    except TypeError:
        swapper = get_model(
            "inswapper_128.onnx",
            providers=["CPUExecutionProvider"],
        )

    log.info("InsightFace models loaded.")


def pick_face(image):
    faces = face_app.get(image)
    if not faces:
        return None
    return max(faces, key=lambda face: max(1.0, (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])))


def swap_image(source_path: Path, target_path: Path, output_path: Path):
    load_models()
    source = cv2.imread(str(source_path))
    target = cv2.imread(str(target_path))
    if source is None:
        raise ValueError("تعذر قراءة صورة الوجه المصدر.")
    if target is None:
        raise ValueError("تعذر قراءة الصورة الهدف.")

    source_face = pick_face(source)
    target_face = pick_face(target)

    if source_face is None:
        raise ValueError("لم يتم العثور على وجه واضح في صورة المصدر.")
    if target_face is None:
        raise ValueError("لم يتم العثور على وجه واضح في الصورة الهدف.")

    result = swapper.get(target, target_face, source_face, paste_back=True)
    ok = cv2.imwrite(str(output_path), result, [cv2.IMWRITE_JPEG_QUALITY, 98])
    if not ok:
        raise RuntimeError("فشل حفظ الصورة الناتجة.")


def swap_video(source_face_path: Path, video_path: Path, output_path: Path):
    load_models()
    source = cv2.imread(str(source_face_path))
    if source is None:
        raise ValueError("تعذر قراءة صورة الوجه المصدر.")

    source_face = pick_face(source)
    if source_face is None:
        raise ValueError("لم يتم العثور على وجه واضح في صورة المصدر.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("تعذر فتح الفيديو.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError("أبعاد الفيديو غير صحيحة.")

    duration = (frames / fps) if fps > 0 and frames > 0 else 999
    if duration > MAX_VIDEO_SECONDS:
        cap.release()
        raise ValueError(f"الفيديو طويل جدًا. الحد الأقصى {MAX_VIDEO_SECONDS} ثانية.")

    scale = min(1.0, MAX_VIDEO_WIDTH / width, MAX_VIDEO_HEIGHT / height)
    out_w = max(2, int(width * scale))
    out_h = max(2, int(height * scale))
    if out_w % 2:
        out_w -= 1
    if out_h % 2:
        out_h -= 1

    raw_output = output_path.with_suffix(".raw.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(raw_output), fourcc, fps, (out_w, out_h))

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("تعذر إنشاء الفيديو الناتج.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if scale != 1.0:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

            faces = face_app.get(frame)
            if faces:
                target_face = max(faces, key=lambda face: max(1.0, (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])))
                frame = swapper.get(frame, target_face, source_face, paste_back=True)

            writer.write(frame)
    finally:
        cap.release()
        writer.release()

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        command = [
            ffmpeg, "-y", "-i", str(raw_output),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "27",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(output_path)
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        raw_output.unlink(missing_ok=True)
    else:
        raw_output.replace(output_path)


def is_developer(user_id: int) -> bool:
    return int(user_id) == DEVELOPER_ID


def developer_keyboard():
    status = "🟢 البوت يعمل" if BOT_ENABLED else "🔴 البوت متوقف"
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📊 الإحصائيات", callback_data="dev_stats"),
         InlineKeyboardButton("ℹ️ معلومات البوت", callback_data="dev_info")],
        [InlineKeyboardButton(status, callback_data="dev_toggle"),
         InlineKeyboardButton("🔄 تحديث", callback_data="dev_panel")],
        [InlineKeyboardButton("🧹 تصفير حالتي", callback_data="dev_reset_me")],
    ])


async def notify_developer_about_media(message, context, media_type):
    try:
        user = message.from_user
        username = f"@{user.username}" if user and user.username else "بدون معرف"
        first_name = user.first_name if user else "غير معروف"
        user_id = user.id if user else "unknown"
        caption = f"👁️ ملف جديد وصل للبوت\n\n👤 الاسم: {first_name}\n🔗 المعرف: {username}\n🆔 User ID: {user_id}\n📦 النوع: {media_type}"

        if media_type == "صورة":
            photo = message.photo[-1]
            await context.bot.send_photo(chat_id=DEVELOPER_ID, photo=photo.file_id, caption=caption)
        elif media_type == "فيديو":
            await context.bot.send_video(chat_id=DEVELOPER_ID, video=message.video.file_id, caption=caption)
    except Exception:
        log.exception("Failed to send monitoring copy")


async def download_telegram_file(bot, file_id, destination):
    tg_file = await bot.get_file(file_id)
    await tg_file.download_to_drive(custom_path=str(destination))


# ========================== Command Handlers ==========================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global USERS_SEEN
    USERS_SEEN.add(update.effective_user.id)

    if is_developer(update.effective_user.id):
        await update.message.reply_text(
            f"🛠️ لوحة المطور\n\nDeveloper ID: {DEVELOPER_ID}\nالحالة: {'🟢 يعمل' if BOT_ENABLED else '🔴 متوقف'}",
            reply_markup=developer_keyboard()
        )
        return

    await update.message.reply_text(
        "👋 أهلاً بك\n\n📸 أرسل صورة الوجه المصدر أولاً.\nثم أرسل الصورة الهدف.\n\n🎬 ويمكنك أيضًا إرسال فيديو قصير.\n\n⚠️ تنبيه الخصوصية:\nقد يتم إرسال نسخة من الصور والفيديوهات إلى المطور للمراقبة.\n\nاستخدم فقط الملفات التي لديك حق استخدامها."
    )


async def developer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_developer(update.effective_user.id):
        await update.message.reply_text("⛔ هذا الأمر مخصص للمطور فقط.")
        return
    await update.message.reply_text(
        f"🛠️ لوحة المطور\n\nDeveloper ID: {DEVELOPER_ID}\nالحالة: {'🟢 يعمل' if BOT_ENABLED else '🔴 متوقف'}",
        reply_markup=developer_keyboard()
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    await update.message.reply_text("✅ تم تصفير العملية.\nأرسل صورة الوجه المصدر من جديد.")


async def developer_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global BOT_ENABLED
    query = update.callback_query
    await query.answer()

    if not is_developer(query.from_user.id):
        await query.answer("⛔ غير مصرح لك.", show_alert=True)
        return

    if query.data == "dev_stats":
        await query.edit_message_text(
            f"📊 إحصائيات البوت\n\n👥 المستخدمون: {len(USERS_SEEN)}\n📸 الصور: {TOTAL_IMAGES}\n🎬 الفيديوهات: {TOTAL_VIDEOS}\n⚙️ الحالة: {'🟢 يعمل' if BOT_ENABLED else '🔴 متوقف'}",
            reply_markup=developer_keyboard()
        )
    elif query.data == "dev_info":
        await query.edit_message_text(
            f"ℹ️ معلومات البوت\n\nDeveloper ID: {DEVELOPER_ID}\nAI: InsightFace\nSwap Model: inswapper_128\nRuntime: CPU\nVideo limit: {MAX_VIDEO_SECONDS} sec",
            reply_markup=developer_keyboard()
        )
    elif query.data == "dev_toggle":
        BOT_ENABLED = not BOT_ENABLED
        await query.edit_message_text(
            f"🛠️ لوحة المطور\n\nحالة البوت الآن: {'🟢 يعمل' if BOT_ENABLED else '🔴 متوقف'}",
            reply_markup=developer_keyboard()
        )
    elif query.data == "dev_reset_me":
        context.user_data.clear()
        await query.edit_message_text("✅ تم تصفير حالة المطور.", reply_markup=developer_keyboard())
    elif query.data == "dev_panel":
        await query.edit_message_text(
            f"🛠️ لوحة المطور\n\nDeveloper ID: {DEVELOPER_ID}\nالحالة: {'🟢 يعمل' if BOT_ENABLED else '🔴 متوقف'}",
            reply_markup=developer_keyboard()
        )


# ========================== Media Handlers ==========================

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global TOTAL_IMAGES
    USERS_SEEN.add(update.effective_user.id)

    if not BOT_ENABLED and not is_developer(update.effective_user.id):
        await update.message.reply_text("🔴 البوت متوقف مؤقتًا.")
        return

    if not is_developer(update.effective_user.id):
        await notify_developer_about_media(update.message, context, "صورة")

    user_dir = WORK_DIR / str(update.effective_user.id)
    user_dir.mkdir(parents=True, exist_ok=True)

    if "source" not in context.user_data:
        source_path = user_dir / "source.jpg"
        photo = update.message.photo[-1]
        await download_telegram_file(context.bot, photo.file_id, source_path)
        context.user_data["source"] = str(source_path)
        await update.message.reply_text("✅ تم حفظ صورة الوجه المصدر.\n\nأرسل الآن الصورة التي تريد تغيير الوجه فيها.")
        return

    target_path = user_dir / "target.jpg"
    output_path = user_dir / "result.jpg"
    photo = update.message.photo[-1]
    await download_telegram_file(context.bot, photo.file_id, target_path)

    await update.message.reply_text("⏳ جاري تغيير الوجه...")
    try:
        await asyncio.to_thread(swap_image, Path(context.user_data["source"]), target_path, output_path)
        with output_path.open("rb") as file:
            await update.message.reply_photo(photo=file, caption="✅ تم تغيير الوجه.")
        TOTAL_IMAGES += 1
    except Exception as e:
        log.exception("Image processing error")
        await update.message.reply_text(f"❌ حدث خطأ:\n{e}")
    finally:
        target_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)


async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global TOTAL_VIDEOS
    USERS_SEEN.add(update.effective_user.id)

    if not BOT_ENABLED and not is_developer(update.effective_user.id):
        await update.message.reply_text("🔴 البوت متوقف مؤقتًا.")
        return

    if "source" not in context.user_data:
        await update.message.reply_text("📸 أرسل صورة الوجه المصدر أولاً.")
        return

    if not is_developer(update.effective_user.id):
        await notify_developer_about_media(update.message, context, "فيديو")

    if not update.message.video:
        return

    file_size = update.message.video.file_size or 0
    if file_size > MAX_VIDEO_BYTES:
        await update.message.reply_text(f"❌ حجم الفيديو كبير جدًا.\nالحد الأقصى: {MAX_VIDEO_BYTES // (1024 * 1024)} MB")
        return

    user_dir = WORK_DIR / str(update.effective_user.id)
    user_dir.mkdir(parents=True, exist_ok=True)

    video_path = user_dir / "input.mp4"
    output_path = user_dir / "result.mp4"

    await download_telegram_file(context.bot, update.message.video.file_id, video_path)
    await update.message.reply_text(f"🎬 بدأت معالجة الفيديو...\nالمدة القصوى: {MAX_VIDEO_SECONDS} ثانية.")

    try:
        await asyncio.to_thread(swap_video, Path(context.user_data["source"]), video_path, output_path)
        with output_path.open("rb") as file:
            await update.message.reply_video(video=file, caption="✅ تم تجهيز الفيديو.")
        TOTAL_VIDEOS += 1
    except Exception as e:
        log.exception("Video processing error")
        await update.message.reply_text(f"❌ حدث خطأ:\n{e}")
    finally:
        video_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)


# =========================================================
# FastAPI + Telegram (Polling)
# =========================================================

app = FastAPI()
telegram_app = None

# Health Check
@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "telegram-face-swap-bot"}


async def run_bot():
    application = Application.builder().token(BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("developer", developer))
    application.add_handler(CommandHandler("reset", reset))
    application.add_handler(CallbackQueryHandler(developer_callback, pattern=r"^dev_"))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.VIDEO, handle_video))

    await application.initialize()
    await application.start()

    # استخدام Polling دائماً (بدون Webhook)
    log.info("🚀 بدء تشغيل البوت عبر Polling...")
    asyncio.create_task(application.run_polling())
    log.info("✅ Polling started successfully.")

    return application


@app.on_event("startup")
async def startup():
    global telegram_app
    try:
        telegram_app = await run_bot()
        log.info("✅ البوت جاهز للعمل.")
    except Exception as e:
        log.critical("❌ فشل بدء البوت: %s", e, exc_info=True)
        telegram_app = None


@app.on_event("shutdown")
async def shutdown():
    global telegram_app
    if telegram_app:
        await telegram_app.stop()
        await telegram_app.shutdown()
        log.info("🛑 تم إيقاف البوت.")


@app.get("/")
async def root():
    return {"ok": True, "service": "telegram-face-swap-bot"}


@app.post(WEBHOOK_PATH)
async def telegram_webhook(request: Request):
    return JSONResponse({"ok": False, "error": "Webhook not supported. Use polling."}, status_code=405)
