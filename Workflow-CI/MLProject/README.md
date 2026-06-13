# Titanic Retrain MLflow Project

Proyek ini digunakan untuk melatih kembali model klasifikasi Titanic secara otomatis menggunakan MLflow Project.

## Struktur Project

*   `MLproject`: File definisi project.
*   `conda.yaml`: Spesifikasi dependencies environment conda.
*   `modelling.py`: Script training model.
*   `titanic_preprocessed.csv`: Dataset preprocessed.

## Cara Menjalankan Secara Lokal

1.  Pastikan Anda telah menginstal dependencies utama:
    ```bash
    pip install mlflow pandas scikit-learn
    ```

2.  Jalankan MLflow Project dari root workspace menggunakan perintah:
    ```bash
    mlflow run Workflow-CI/MLProject --no-conda
    ```

    Jika Anda memiliki Conda terinstal dan ingin menggunakannya untuk isolasi environment, jalankan:
    ```bash
    mlflow run Workflow-CI/MLProject
    ```
