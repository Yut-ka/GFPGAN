import os
import subprocess
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Папки
UPLOAD_FOLDER = "inputs/whole_imgs"
RESULT_FOLDER = "results/restored_imgs"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET"])
def index():
    return "GFPGAN Flask API работает! Используйте POST /restore с фото."

@app.route("/restore", methods=["POST"])
def restore():
    if 'image' not in request.files:
        return jsonify({"error": "Нет файла image в запросе"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Имя файла пустое"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        # Очистим входную и выходную папки
        for f in os.listdir(UPLOAD_FOLDER):
            os.remove(os.path.join(UPLOAD_FOLDER, f))
        for f in os.listdir(RESULT_FOLDER):
            os.remove(os.path.join(RESULT_FOLDER, f))

        # Сохраняем
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Запускаем GFPGAN
        try:
            subprocess.run([
                "python", "inference_gfpgan.py",
                "-i", UPLOAD_FOLDER,
                "-o", "results",
                "-v", "1.3",
                "-s", "2"
            ], check=True)
        except subprocess.CalledProcessError as e:
            return jsonify({"error": f"Ошибка запуска модели: {e}"}), 500

        # Ищем результат
        result_files = os.listdir(RESULT_FOLDER)
        if result_files:
            return jsonify({"restored_image": f"/{RESULT_FOLDER}/{result_files[0]}"})
        else:
            return jsonify({"error": "Результат не найден"}), 500
    else:
        return jsonify({"error": "Формат файла не поддерживается"}), 400


