import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
import os
import shutil

def main():
    # Gunakan mlflow tracking lokal (default ./mlruns)
    mlflow.set_experiment("Titanic_Classifier_Retrain")

    # 1. Load dataset
    print("Membaca dataset...")
    df = pd.read_csv("titanic_preprocessed.csv")

    # 2. Pisahkan fitur dan target
    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    # 3. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Aktifkan mlflow.autolog()
    mlflow.autolog()

    # 5. Training Random Forest
    print("Melatih model...")
    with mlflow.start_run(run_name="RandomForest_Retrain") as run:
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")

        # Simpan model secara eksplisit ke folder artifacts/model untuk workflow CI
        # Bersihkan folder jika sudah ada sebelumnya
        artifacts_dir = os.path.join("artifacts", "model")
        if os.path.exists(artifacts_dir):
            shutil.rmtree(artifacts_dir)
        
        os.makedirs(os.path.dirname(artifacts_dir), exist_ok=True)
        mlflow.sklearn.save_model(model, artifacts_dir)
        print(f"Model berhasil dilatih dan disimpan secara lokal di: {artifacts_dir}")

if __name__ == "__main__":
    main()
