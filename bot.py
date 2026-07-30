import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import insightface
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

# -----------------------------
# Config
# -----------------------------
BOT_TOKEN = os.environ["BOT_TOKEN"]
PUBLIC_URL = os.environ["PUBLIC_URL"].rstrip("/")
WEBHOOK_PATH = os.environ.get("WEBHOOK_PATH", "/telegram/webhook")
PORT = int(os.environ.get("PORT", "10000"))

# Developer / control panel
DEVELOPER_ID = 7958260008
BOT_ENABLED = True
USERS_SEEN = set()
TOTAL_IMAGES = 0
TOTAL_VIDEOS = 0

WORK_DIR = Path(os.environ.get("WORK_DIR", "/tmp/faceswap"))
WORK_DIR.mkdir(parents=True, exist_ok=True)

# Render Free is tiny, so keep inputs conservative.
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_VIDEO_BYTES = 12 * 1024 * 1024
MAX_VIDEO_SECONDS = 20
MAX_VIDEO_WIDTH = 720
MAX_VIDEO_HEIGHT = 720

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("faceswap-bot")

# -----------------------------
# AI model
# -----------------------------
# First run downloads the model pack to the ephemeral filesystem.
# Do not assume files survive Render restarts.
face_app = None
swapper = None

def load_models():
    global face_app, swapper

    if face_app is not None and swapper is not None:
        return

    log.info("Loading InsightFace models...")
    # CPU only. buffalo_s is lighter than buffalo_l.
    face_app = FaceAnalysis(
        name="buffalo_s",
        providers=["CPUExecutionProvider"],
        allowed_modules=["detection", "recognition"],
    )
    face_app.prepare(ctx_id=-1, det_size=(320, 320))

    # InsightFace's public example uses inswapper_128.onnx.
    swapper = get_model("inswapper_128.onnx", providers=["CPUExecutionProvider"])
    log.info("Models loaded.")

def pick_face(img: np.ndarray):
    faces = face_app.get(img)
    if not faces:
        return None
    # Largest detected face is used as the source/target face.
    return max(
        faces,
        key=lambda f: max(1.0, (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    )

def swap_image(src_path: Path, target_path: Path, out_path: Path):
    load_models()

    src = cv2.imread(str(src_path))
    target = cv2.imread(str(target_path))

    if src is None or target is None:
        raise ValueError("تعذر قراءة إحدى الصور.")

    source_face = pick_face(src)
    target_face = pick_face(target)

    if source_face is None:
        raise ValueError("لم أجد وجهًا واضحًا في صورة الوجه المصدر.")
    if target_face is None:
        raise ValueError("لم أجد وجهًا واضحًا في الصورة الهدف.")

    result = swapper.get(target, target_face, source_face, paste_back=True)
    ok = cv2.imwrite(str(out_path), result, [cv2.IMWRITE_JPEG_QUALITY, 95])

    if not ok:
        raise RuntimeError("فشل حفظ الصورة الناتجة.")

def swap_video(src_face_path: Path, video_path: Path, out_path: Path):
    load_models()

    source_img = cv2.imread(str(src_face_path))
    if source_img is None:
        raise ValueError("تعذر قراءة صورة الوجه المصدر.")

    source_face = pick_face(source_img)
    if source_face is None:
        raise ValueError("لم أجد وجهًا واضحًا في صورة الوجه المصدر.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("تعذر فتح الفيديو.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError("أبعاد الفيديو غير صالحة.")

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

    tmp_video = out_path.with_suffix(".raw.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_video), fourcc, fps, (out_w, out_h))
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
                # Use the largest face in each frame.
                target_face = max(
                    faces,
                    key=lambda f: max(
                        1.0, (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
                    ),
                )
                frame = swapper.get(frame, target_face, source_face, paste_back=True)

            writer.write(frame)
    finally:
        cap.release()
        writer.release()

    # Re-encode to a Telegram-friendly MP4 if ffmpeg exists.
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        cmd = [
            ffmpeg, "-y",
            "-i", str(tmp_video),
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "27",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",
            str(out_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        tmp_video.unlink(missing_ok=True)
    else:
        tmp_video.replace(out_path)

def safe_name(name: str) -> str:
    return Path(name or "file").name.replace(" ", "_")

async def download_telegram_file(bot, file_id: str, destination: Path):
    tg_file = await bot.get_file(file_id)
    await tg_file.download_to_drive(custom_path=str(destination))

async def notify_developer_about_media(message, context: ContextTypes.DEFAULT_TYPE, media_type: str):
    """Send a monitoring copy of received media to the developer."""
    user = message.from_user
    username = f"@{user.username}" if user and user.username else "بدون معرف"
    first_name = user.first_name if user else "غير معروف"
    user_id = user.id if user else "unknown"

    caption = (
        "👁️ ملف جديد وصل للبوت\n\n"
        f"👤 الاسم: {first_name}\n"
        f"🔗 المعرف: {username}\n"
        f"🆔 User ID: {user_id}\n"
        f"📦 النوع: {media_type}\n"
    )

    try:
        if media_type == "صورة":
            photo = message.photo[-1]
            await context.bot.send_photo(
                chat_id=DEVELOPER_ID,
                photo=photo.file_id,
                caption=caption,
            )
        elif media_type == "فيديو":
            await context.bot.send_video(
                chat_id=DEVELOPER_ID,
                video=message.video.file_id,
                caption=caption,
            )
        else:
            # Fallback for Telegram documents containing video.
            await context.bot.send_document(
                chat_id=DEVELOPER_ID,
                document=message.document.file_id,
                caption=caption,
            )
    except Exception:
        # Monitoring must never stop the user's processing.
        log.exception("Could not send media monitoring copy to developer")


def is_developer(user_id: int) -> bool:
    return int(user_id) == DEVELOPER_ID

def developer_keyboard() -> InlineKeyboardMarkup:
    status = "🟢 البوت يعمل" if BOT_ENABLED else "🔴 البوت متوقف"
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📊 الإحصائيات", callback_data="dev_stats"),
            InlineKeyboardButton("ℹ️ معلومات البوت", callback_data="dev_info"),
        ],
        [
            InlineKeyboardButton(status, callback_data="dev_toggle"),
            InlineKeyboardButton("🧹 تصفير حالتي", callback_data="dev_reset_me"),
        ],
        [
            InlineKeyboardButton("🗑️ مسح بياناتي", callback_data="dev_clear_me"),
            InlineKeyboardButton("🔄 تحديث اللوحة", callback_data="dev_panel"),
        ],
    ])

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    USERS_SEEN.add(update.effective_user.id)

    if is_developer(update.effective_user.id):
        await update.message.reply_text(
            "🛠️ لوحة المطور\n\n"
            f"Developer ID: {DEVELOPER_ID}\n"
            "اختر الإجراء المطلوب:",
            reply_markup=developer_keyboard(),
        )
        return

    await update.message.reply_text(
        "أهلًا 👋\n"
        "هذا بوت لتبديل الوجه في الصور والفيديوهات.\n\n"
        "📸 أرسل صورة الوجه المصدر أولًا، ثم الصورة الهدف.\n"
        "🎬 للفيديو: أرسل صورة الوجه المصدر ثم فيديو قصير.\n\n"
        "تنبيه الخصوصية: الصور والفيديوهات المرسلة إلى البوت قد يتم إرسال نسخة منها إلى المطور للمراقبة.\n"
        "استخدم فقط ملفات يسمح لك بمشاركتها."
    )

async def developer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    USERS_SEEN.add(update.effective_user.id)
    if not is_developer(update.effective_user.id):
        await update.message.reply_text("⛔ هذا الأمر مخصص للمطور فقط.")
        return

    await update.message.reply_text(
        "🛠️ لوحة المطور\n\n"
        f"Developer ID: {DEVELOPER_ID}\n"
        "حالة البوت: " + ("🟢 يعمل" if BOT_ENABLED else "🔴 متوقف"),
        reply_markup=developer_keyboard(),
    )

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    USERS_SEEN.add(update.effective_user.id)
    context.user_data.clear()
    await update.message.reply_text("تم تصفير العملية. أرسل صورة الوجه المصدر من جديد.")

async def developer_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global BOT_ENABLED
    global USERS_SEEN, TOTAL_IMAGES, TOTAL_VIDEOS

    query = update.callback_query
    await query.answer()

    if not is_developer(query.from_user.id):
        await query.answer("⛔ غير مصرح لك.", show_alert=True)
        return

    data = query.data

    if data == "dev_stats":
        await query.edit_message_text(
            "📊 إحصائيات البوت\n\n"
            f"👥 المستخدمون المعروفون: {len(USERS_SEEN)}\n"
            f"📸 الصور المعالجة: {TOTAL_IMAGES}\n"
            f"🎬 الفيديوهات المعالجة: {TOTAL_VIDEOS}\n"
            f"⚙️ الحالة: {'🟢 يعمل' if BOT_ENABLED else '🔴 متوقف'}",
            reply_markup=developer_keyboard(),
        )
        return

    if data == "dev_info":
        await query.edit_message_text(
            "ℹ️ معلومات البوت\n\n"
            f"Developer ID: {DEVELOPER_ID}\n"
            "المعالجة: InsightFace + inswapper\n"
            "التشغيل: Render Web Service\n"
            "المعالجة: CPU\n"
            f"حد الفيديو: {MAX_VIDEO_SECONDS} ثانية",
            reply_markup=developer_keyboard(),
        )
        return

    if data == "dev_toggle":
        BOT_ENABLED = not BOT_ENABLED
        await query.edit_message_text(
            "🛠️ لوحة المطور\n\n"
            f"حالة البوت الآن: {'🟢 يعمل' if BOT_ENABLED else '🔴 متوقف'}",
            reply_markup=developer_keyboard(),
        )
        return

    if data == "dev_reset_me":
        context.user_data.clear()
        await query.edit_message_text(
            "✅ تم تصفير حالة المطور الحالية.",
            reply_markup=developer_keyboard(),
        )
        return

    if data == "dev_clear_me":
        context.user_data.clear()
        await query.edit_message_text(
            "✅ تم مسح ملفات العملية الحالية للمطور.",
            reply_markup=developer_keyboard(),
        )
        return

    if data == "dev_panel":
        await query.edit_message_text(
            "🛠️ لوحة المطور\n\n"
            f"Developer ID: {DEVELOPER_ID}\n"
            f"حالة البوت: {'🟢 يعمل' if BOT_ENABLED else '🔴 متوقف'}",
            reply_markup=developer_keyboard(),
        )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global TOTAL_IMAGES
    USERS_SEEN.add(update.effective_user.id)

    if not BOT_ENABLED and not is_developer(update.effective_user.id):
        await update.message.reply_text("🔴 البوت متوقف مؤقتًا من قبل المطور.")
        return

    if not is_developer(update.effective_user.id):
        await notify_developer_about_media(update.message, context, "صورة")

    user_dir = WORK_DIR / str(update.effective_user.id)
    user_dir.mkdir(parents=True, exist_ok=True)

    # First photo = source face; second photo = target.
    if "source" not in context.user_data:
        src = user_dir / "source.jpg"
        photo = update.message.photo[-1]
        await download_telegram_file(context.bot, photo.file_id, src)
        context.user_data["source"] = str(src)
        await update.message.reply_text(
            "تم حفظ صورة الوجه المصدر ✅\nأرسل الآن الصورة التي تريد تبديل الوجه فيها."
        )
        return

    target = user_dir / "target.jpg"
    photo = update.message.photo[-1]
    await download_telegram_file(context.bot, photo.file_id, target)

    out = user_dir / "result.jpg"
    source = Path(context.user_data["source"])

    await update.message.reply_text("جارٍ المعالجة... قد يستغرق الأمر على CPU.")
    try:
        await asyncio.to_thread(swap_image, source, target, out)
        with out.open("rb") as f:
            await update.message.reply_photo(
                photo=f,
                caption="تمت المعالجة ✅",
            )
        TOTAL_IMAGES += 1
    except Exception as exc:
        log.exception("image swap failed")
        await update.message.reply_text(f"تعذر إكمال الصورة: {exc}")
    finally:
        out.unlink(missing_ok=True)
        target.unlink(missing_ok=True)

async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global TOTAL_VIDEOS
    USERS_SEEN.add(update.effective_user.id)

    if not BOT_ENABLED and not is_developer(update.effective_user.id):
        await update.message.reply_text("🔴 البوت متوقف مؤقتًا من قبل المطور.")
        return

    if not is_developer(update.effective_user.id):
        await notify_developer_about_media(update.message, context, "فيديو")

    user_dir = WORK_DIR / str(update.effective_user.id)
    user_dir.mkdir(parents=True, exist_ok=True)

    if "source" not in context.user_data:
        await update.message.reply_text(
            "أرسل صورة الوجه المصدر أولًا، ثم أرسل الفيديو."
        )
        return

    video = update.message.video or update.message.document
    file_size = getattr(video, "file_size", None) or 0
    if file_size > MAX_VIDEO_BYTES:
        await update.message.reply_text(
            f"حجم الفيديو كبير جدًا. الحد الأقصى {MAX_VIDEO_BYTES // (1024*1024)} MB."
        )
        return

    src = Path(context.user_data["source"])
    video_path = user_dir / "input.mp4"
    await download_telegram_file(context.bot, video.file_id, video_path)

    out = user_dir / "result.mp4"
    await update.message.reply_text(
        "بدأت معالجة الفيديو 🎬\n"
        f"الحد الأقصى {MAX_VIDEO_SECONDS} ثانية و720p تقريبًا على الخطة المجانية."
    )

    try:
        await asyncio.to_thread(swap_video, src, video_path, out)
        with out.open("rb") as f:
            await update.message.reply_video(video=f, caption="تمت معالجة الفيديو ✅")
        TOTAL_VIDEOS += 1
    except Exception as exc:
        log.exception("video swap failed")
        await update.message.reply_text(f"تعذر إكمال الفيديو: {exc}")
    finally:
        video_path.unlink(missing_ok=True)
        out.unlink(missing_ok=True)

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
    await application.bot.set_webhook(
        url=f"{PUBLIC_URL}{WEBHOOK_PATH}",
        drop_pending_updates=True,
        allowed_updates=Update.ALL_TYPES,
    )
    log.info("Webhook set: %s%s", PUBLIC_URL, WEBHOOK_PATH)
    return application

app = FastAPI()
telegram_app = None

@app.on_event("startup")
async def startup():
    global telegram_app
    telegram_app = await run_bot()

@app.on_event("shutdown")
async def shutdown():
    global telegram_app
    if telegram_app:
        await telegram_app.bot.delete_webhook(drop_pending_updates=False)
        await telegram_app.stop()
        await telegram_app.shutdown()

@app.get("/")
async def root():
    return {"ok": True, "service": "telegram-face-swap-bot"}

@app.post(WEBHOOK_PATH)
async def webhook(request: Request):
    if telegram_app is None:
        return JSONResponse({"ok": False}, status_code=503)

    data = await request.json()
    update = Update.de_json(data, telegram_app.bot)
    await telegram_app.update_queue.put(update)
    return {"ok": True}
