import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from svd import calcMatrix, calcDiagonal  # svd.py'den fonksiyonları alıyoruz

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'


# initial page
@app.route('/')
def index():
    return render_template('index.html')


# post process
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files or 'k_value' not in request.form:
        return "Dosya yüklenmedi veya K değeri eksik."

    file = request.files['file']
    k_value = int(request.form['k_value'])

    if file.filename == '':
        return "Dosya seçilmedi."

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)


        img = cv2.imread(filepath)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_image = gray_image.astype(np.float64)


        U, s, V = np.linalg.svd(gray_image, full_matrices=False)
        low_rank = U[:, :k_value] @ np.diag(s[:k_value]) @ V[:k_value, :]


        Vt_manuel = calcMatrix(gray_image, 1)
        U_manuel = calcMatrix(gray_image, 2)
        Sigma_manuel = calcDiagonal(gray_image)
        low_rank_manuel = U_manuel[:, :k_value] @ np.diag(Sigma_manuel[:k_value]) @ Vt_manuel[:k_value, :]

        # save compressed images
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(low_rank, cmap='gray')
        plt.title(f"Library SVD (K = {k_value})")
        lib_svd_path = os.path.join(app.config['UPLOAD_FOLDER'], "library_svd.png")
        plt.savefig(lib_svd_path)

        plt.subplot(1, 2, 2)
        plt.imshow(low_rank_manuel, cmap='gray')
        plt.title(f"Manual SVD (K = {k_value})")
        manual_svd_path = os.path.join(app.config['UPLOAD_FOLDER'], "manual_svd.png")
        plt.savefig(manual_svd_path)

        # show results
        return render_template('result.html',lib_svd='library_svd.png',manual_svd='manual_svd.png')


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
