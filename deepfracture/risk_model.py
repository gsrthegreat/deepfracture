import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

class RiskStratificationModel:
    def __init__(self, model_path="risk_rf_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.classes = ['Low Risk', 'Medium Risk', 'High Risk']
        
        if os.path.exists(self.model_path):
            self.load_model()
        else:
            self._train_dummy_model()
            
    def _train_dummy_model(self):
        """
        Trains a Random Forest classifier using advanced image features.
        Features: [confidence_score, mean_intensity, std_dev, edge_density, glcm_contrast, glcm_homogeneity]
        """
        np.random.seed(42)
        n_samples = 150
        
        # Simulate features
        # Low risk: High confidence normal, or low edge density/contrast
        low_X = np.c_[
            np.random.uniform(0, 0.3, n_samples), # confidence score (fracture prob)
            np.random.uniform(100, 150, n_samples), # mean_intensity
            np.random.uniform(20, 50, n_samples),  # std_dev
            np.random.uniform(0.01, 0.05, n_samples), # edge_density
            np.random.uniform(100, 500, n_samples), # contrast
            np.random.uniform(0.4, 0.8, n_samples)  # homogeneity
        ]
        low_y = np.zeros(n_samples)
        
        # Medium risk
        med_X = np.c_[
            np.random.uniform(0.3, 0.7, n_samples),
            np.random.uniform(80, 170, n_samples),
            np.random.uniform(40, 70, n_samples),
            np.random.uniform(0.04, 0.08, n_samples),
            np.random.uniform(400, 1000, n_samples),
            np.random.uniform(0.2, 0.6, n_samples)
        ]
        med_y = np.ones(n_samples)
        
        # High risk: High fracture probability, complex texture/edges
        high_X = np.c_[
            np.random.uniform(0.7, 1.0, n_samples),
            np.random.uniform(50, 200, n_samples),
            np.random.uniform(60, 100, n_samples),
            np.random.uniform(0.07, 0.15, n_samples),
            np.random.uniform(800, 2000, n_samples),
            np.random.uniform(0.05, 0.3, n_samples)
        ]
        high_y = np.full(n_samples, 2)
        
        X = np.vstack([low_X, med_X, high_X])
        y = np.concatenate([low_y, med_y, high_y])
        
        self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.model.fit(X, y)
        
        joblib.dump(self.model, self.model_path)
        print(f"Dummy Random Forest Risk Model trained and saved to {self.model_path}")
        
    def load_model(self):
        self.model = joblib.load(self.model_path)
        
    def predict_risk(self, confidence_score, mean_int, std_dev, edge_density, contrast, homogeneity):
        if self.model is None:
            raise Exception("Model not loaded or trained.")
            
        features = np.array([[confidence_score, mean_int, std_dev, edge_density, contrast, homogeneity]])
        prediction = self.model.predict(features)[0]
        
        return self.classes[int(prediction)]
