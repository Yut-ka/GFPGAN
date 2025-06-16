import os
import subprocess
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "inputs/whole_imgs"
RESULT_FOLDER = "results/restored_imgs"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/restore", methods=["POST"])
def restore():
    if 'image' not in request.files:
        return jsonify({"error": "Нет файла image в запросе"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Имя файла пустое"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        # Очистка папок
        for f in os.listdir(UPLOAD_FOLDER):
            os.remove(os.path.join(UPLOAD_FOLDER, f))
        for f in os.listdir(RESULT_FOLDER):
            os.remove(os.path.join(RESULT_FOLDER, f))

        # Сохранение оригинального файла
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        try:
            # === Этап 1: GFPGAN v1.4 ===
            result1 = subprocess.run([
                "python", "inference_gfpgan.py",
                "-i", UPLOAD_FOLDER,
                "-o", "results",
                "-v", "1.4",
                "--bg_upsampler", "realesrgan",
                "-s", "2"
            ], capture_output=True, text=True)

            if result1.returncode != 0:
                return jsonify({"error": "Ошибка запуска GFPGAN v1.4", "stderr": result1.stderr}), 500

            intermediate_file = os.listdir(RESULT_FOLDER)[0]
            intermediate_path = os.path.join(RESULT_FOLDER, intermediate_file)

            # === Копируем результат в новый путь (чтобы сохранить его)
            step1_filename = "restored_step1_" + filename
            step1_output_path = os.path.join("results", "restored_imgs", step1_filename)
            os.rename(intermediate_path, step1_output_path)

            # === Подготовка входа для RestoreFormer
            os.rename(step1_output_path, os.path.join(UPLOAD_FOLDER, step1_filename))

            # Очистка предыдущих результатов
            for f in os.listdir(RESULT_FOLDER):
                os.remove(os.path.join(RESULT_FOLDER, f))

            # === Этап 2: RestoreFormer
            result2 = subprocess.run([
                "python", "inference_gfpgan.py",
                "-i", UPLOAD_FOLDER,
                "-o", "results",
                "-v", "RestoreFormer",
                "--bg_upsampler", "realesrgan",
                "-s", "2"
            ], capture_output=True, text=True)

            if result2.returncode != 0:
                return jsonify({"error": "Ошибка запуска RestoreFormer", "stderr": result2.stderr}), 500

            final_file = os.listdir(RESULT_FOLDER)[0]
            final_output_path = os.path.join("results", "restored_imgs", "restored_final_" + filename)
            os.rename(os.path.join(RESULT_FOLDER, final_file), final_output_path)

            return jsonify({
                "step1_image": f"/{step1_output_path}",
                "final_image": f"/{final_output_path}"
            })

        except Exception as e:
            return jsonify({"error": f"Ошибка обработки: {str(e)}"}), 500
    else:
        return jsonify({"error": "Формат файла не поддерживается"}), 400
