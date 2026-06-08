"""Build expanded dataset and train all DeepFracture models."""
from build_dataset import build_dataset
import train

DATA_DIR = "./data"

if __name__ == "__main__":
    print("=" * 50)
    print("DeepFracture — dataset build + training")
    print("=" * 50)
    build_dataset(clean_synthetic=True)

    print("\n==================================")
    print("Training DenseNet169 (20 epochs)...")
    print("==================================")
    train.train_model(data_dir=DATA_DIR, model_type="densenet", num_epochs=20, batch_size=16)

    print("\n==================================")
    print("Training ResNet50 (20 epochs)...")
    print("==================================")
    train.train_model(data_dir=DATA_DIR, model_type="resnet", num_epochs=20, batch_size=16)

    print("\n==================================")
    print("Training Baseline CNN (12 epochs)...")
    print("==================================")
    train.train_model(data_dir=DATA_DIR, model_type="baseline", num_epochs=12, batch_size=16)

    print("\nDone. Restart Streamlit: streamlit run app.py")
