import os
import subprocess
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from shutil import copyfile
from datetime import datetime

app = Flask(__name__)

UPLOAD_FOLDER = "inputs/whole_imgs"
RESULT_FOLDER = "results/restored_imgs"
BASE_URL = "https://yut-ka-gfpgan-40e7.twc1.net"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/results/restored_imgs/<filename>')
def serve_restored_image(filename):
    return send_from_directory(os.path.join("results", "restored_imgs"), filename)

@app.route("/restore", methods=["POST"])
def restore():
    print("Начало обработки запроса")

    if 'image' not in request.files:
        print("Ошибка: отсутствует поле 'image'")
        return jsonify({"error": "Нет файла image в запросе"}), 400

    file = request.files['image']
    if file.filename == '':
        print("Ошибка: имя файла пустое")
        return jsonify({"error": "Имя файла пустое"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        print(f"Файл принят: {filename}")

        # Очистка папок
        for f in os.listdir(UPLOAD_FOLDER):
            os.remove(os.path.join(UPLOAD_FOLDER, f))
        for f in os.listdir(RESULT_FOLDER):
            os.remove(os.path.join(RESULT_FOLDER, f))
        print("Папки очищены")

        # Сохранение оригинального файла
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        print(f"Файл сохранён в: {filepath}")

        try:
            # === Этап 1: GFPGAN v1.4 ===
            print("Запуск GFPGAN v1.4")
            result1 = subprocess.run([
                "python", "inference_gfpgan.py",
                "-i", UPLOAD_FOLDER,
                "-o", "results",
                "-v", "1.4",
                "--bg_upsampler", "realesrgan",
                "-s", "2"
            ], capture_output=True, text=True)

            if result1.returncode != 0:
                print(f"Ошибка GFPGAN v1.4: {result1.stderr}")
                return jsonify({"error": "Ошибка запуска GFPGAN v1.4", "stderr": result1.stderr}), 500

            intermediate_file = os.listdir(RESULT_FOLDER)[0]
            intermediate_path = os.path.join(RESULT_FOLDER, intermediate_file)
            step1_filename = "restored_step1_" + filename
            step1_output_path = os.path.join(RESULT_FOLDER, step1_filename)
            os.rename(intermediate_path, step1_output_path)
            print(f"Файл этапа 1 сохранён: {step1_output_path}")

            # Подготовка входа для этапа 2
            step1_input_path = os.path.join(UPLOAD_FOLDER, "step1_input.jpg")
            copyfile(step1_output_path, step1_input_path)
            print(f"Файл этапа 1 скопирован как вход для RestoreFormer: {step1_input_path}")

            # Очистка результатов
            for f in os.listdir(RESULT_FOLDER):
                os.remove(os.path.join(RESULT_FOLDER, f))

            # === Этап 2: RestoreFormer ===
            print("Запуск RestoreFormer")
            result2 = subprocess.run([
                "python", "inference_gfpgan.py",
                "-i", UPLOAD_FOLDER,
                "-o", "results",
                "-v", "RestoreFormer",
                "--bg_upsampler", "realesrgan",
                "-s", "2"
            ], capture_output=True, text=True)

            if result2.returncode != 0:
                print(f"Ошибка RestoreFormer: {result2.stderr}")
                return jsonify({"error": "Ошибка запуска RestoreFormer", "stderr": result2.stderr}), 500

            final_file = os.listdir(RESULT_FOLDER)[0]
            final_output_path = os.path.join(RESULT_FOLDER, "restored_final_" + filename)
            os.rename(os.path.join(RESULT_FOLDER, final_file), final_output_path)
            print(f"Файл этапа 2 сохранён: {final_output_path}")

            return jsonify({
                "step1_image": f"{BASE_URL}/results/restored_imgs/{step1_filename}",
                "final_image": f"{BASE_URL}/results/restored_imgs/restored_final_{filename}"
            })

        except Exception as e:
            print(f"Ошибка общего процесса: {str(e)}")
            return jsonify({"error": f"Ошибка обработки: {str(e)}"}), 500

    else:
        print("Ошибка: неподдерживаемый формат")
        return jsonify({"error": "Формат файла не поддерживается"}), 400
