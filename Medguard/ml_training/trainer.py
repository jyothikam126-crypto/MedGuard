"""
ML Trainer Module for MedGuard Health Platform
Handles training, evaluation, and saving of ML models for disease prediction
"""

import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from .datasets import (
    DIABETES_DATA, DIABETES_FEATURES,
    HEART_DATA, HEART_FEATURES,
    SYMPTOM_DATA, SYMPTOM_FEATURES, CONDITION_LABELS
)


class MLTrainer:
    """Main ML Trainer class for training all disease prediction models"""
    
    def __init__(self, models_dir='diabetes/model_files'):
        self.models_dir = models_dir
        self.models = {}
        self.scalers = {}
        self.metrics = {}
        self.random_state = 42
        self.min_training_samples = 500
        self.max_training_samples = 1000
        self.target_training_samples = 800
        
    def _normalize_features(self, X):
        """Normalize features using StandardScaler"""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, scaler

    def _resize_binary_dataset(
        self,
        X,
        y,
        categorical_indices=None,
        numeric_noise_scale=0.03
    ):
        """
        Resize a binary-classification dataset to stay in [min_training_samples, max_training_samples].
        If data is too small, bootstrap with slight perturbation on numeric columns.
        """
        if categorical_indices is None:
            categorical_indices = []

        n_samples = len(X)
        if self.min_training_samples <= n_samples <= self.max_training_samples:
            return X, y

        classes, counts = np.unique(y, return_counts=True)
        if len(classes) != 2:
            return X, y

        target_size = (
            self.target_training_samples
            if n_samples < self.min_training_samples
            else self.max_training_samples
        )

        per_class = target_size // 2
        X_resized_list = []
        y_resized_list = []

        for cls in classes:
            cls_idx = np.where(y == cls)[0]
            replace = len(cls_idx) < per_class
            sampled_idx = np.random.choice(cls_idx, size=per_class, replace=replace)
            X_cls = X[sampled_idx].astype(float).copy()
            y_cls = np.full(per_class, cls)

            # Add slight noise only to numeric columns when upsampling.
            if replace and numeric_noise_scale > 0:
                all_indices = np.arange(X.shape[1])
                numeric_indices = np.setdiff1d(all_indices, np.array(categorical_indices))
                if len(numeric_indices) > 0:
                    stds = X[:, numeric_indices].std(axis=0)
                    stds = np.where(stds == 0, 1.0, stds)
                    noise = np.random.normal(
                        loc=0.0,
                        scale=numeric_noise_scale * stds,
                        size=(X_cls.shape[0], len(numeric_indices))
                    )
                    X_cls[:, numeric_indices] = X_cls[:, numeric_indices] + noise

            # Keep categorical columns as valid 0/1 style values.
            if len(categorical_indices) > 0:
                X_cls[:, categorical_indices] = np.clip(
                    np.rint(X_cls[:, categorical_indices]), 0, 1
                )

            X_resized_list.append(X_cls)
            y_resized_list.append(y_cls)

        X_resized = np.vstack(X_resized_list)
        y_resized = np.concatenate(y_resized_list)

        shuffle_idx = np.random.permutation(len(X_resized))
        return X_resized[shuffle_idx], y_resized[shuffle_idx]
    
    def train_diabetes_model(self):
        """Train the Diabetes Prediction Model"""
        print("Training Diabetes Model...")
        
        # Get data
        np.random.seed(self.random_state)
        X = np.array(DIABETES_DATA['features'])
        y = np.array(DIABETES_DATA['labels'])
        X, y = self._resize_binary_dataset(
            X, y, categorical_indices=list(range(1, 16)), numeric_noise_scale=0.02
        )
        print(f"  Diabetes dataset size used for training: {len(X)}")
        
        # Split data - using smaller test size to accommodate 9 classes
        # With only 27 samples and 9 classes, we need smaller test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42
        )
        
        # Scale features
        X_train_scaled, scaler = self._normalize_features(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train multiple models and select the best
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100, max_depth=5, random_state=42
            ),
            'LogisticRegression': LogisticRegression(
                max_iter=1000, random_state=42
            )
        }
        
        best_model = None
        best_score = 0
        best_name = ""
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            score = model.score(X_test_scaled, y_test)
            print(f"  {name}: Accuracy = {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_model = model
                best_name = name
        
        # Use Random Forest as the primary model for consistency
        final_model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42
        )
        final_model.fit(X_train_scaled, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(final_model, X_train_scaled, y_train, cv=5)
        
        # Final evaluation
        y_pred = final_model.predict(X_test_scaled)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'model_type': 'RandomForest'
        }
        
        # Save model and scaler
        self.models['diabetes'] = final_model
        self.scalers['diabetes'] = scaler
        self.metrics['diabetes'] = metrics
        
        # Save to file
        self._save_model('diabetes', final_model, scaler, DIABETES_FEATURES)
        
        print(f"  Diabetes Model trained successfully!")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  CV Score: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")
        
        return metrics
    
    def train_heart_disease_model(self):
        """Train the Heart Disease Prediction Model"""
        print("Training Heart Disease Model...")
        
        # Get data
        np.random.seed(self.random_state)
        X = np.array(HEART_DATA['features'])
        y = np.array(HEART_DATA['labels'])
        X, y = self._resize_binary_dataset(
            X, y, categorical_indices=[1, 2, 5, 6, 8], numeric_noise_scale=0.03
        )
        print(f"  Heart disease dataset size used for training: {len(X)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled, scaler = self._normalize_features(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        
        # Evaluation
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'model_type': 'RandomForest'
        }
        
        # Save model and scaler
        self.models['heart_disease'] = model
        self.scalers['heart_disease'] = scaler
        self.metrics['heart_disease'] = metrics
        
        # Save to file
        self._save_model('heart_disease', model, scaler, HEART_FEATURES)
        
        print(f"  Heart Disease Model trained successfully!")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def train_symptom_checker_model(self):
        """Train the Symptom Checker Model"""
        print("Training Symptom Checker Model...")
        
        # Get data
        X = np.array(SYMPTOM_DATA['features'])
        y = np.array(SYMPTOM_DATA['labels'])
        
        # Split data - use smaller test size (0.1 = ~3 samples) to accommodate 9 classes
        # Also remove stratification since we have only 3 samples per class
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42
        )
        
        # Scale features
        X_train_scaled, scaler = self._normalize_features(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model - Random Forest works well for multi-class
        model = RandomForestClassifier(
            n_estimators=100, max_depth=15, random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        
        # Evaluation
        y_pred = model.predict(X_test_scaled)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'model_type': 'RandomForest'
        }
        
        # Save model and scaler
        self.models['symptom_checker'] = model
        self.scalers['symptom_checker'] = scaler
        self.metrics['symptom_checker'] = metrics
        
        # Save to file
        self._save_model('symptom_checker', model, scaler, SYMPTOM_FEATURES)
        
        print(f"  Symptom Checker Model trained successfully!")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        
        return metrics
    
    def _save_model(self, name, model, scaler, feature_names):
        """Save trained model to disk"""
        # Create directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(self.models_dir, f'{name}_model.pkl')
        joblib.dump(model, model_path)
        
        # Save scaler
        scaler_path = os.path.join(self.models_dir, f'{name}_scaler.pkl')
        joblib.dump(scaler, scaler_path)
        
        # Save feature names
        features_path = os.path.join(self.models_dir, f'{name}_features.pkl')
        joblib.dump(feature_names, features_path)
        
        print(f"  Model saved to: {model_path}")
    
    def train_all(self):
        """Train all ML models"""
        print("\n" + "="*50)
        print("Starting ML Training Pipeline")
        print("="*50)
        
        results = {}
        
        # Train all models
        results['diabetes'] = self.train_diabetes_model()
        results['heart_disease'] = self.train_heart_disease_model()
        results['symptom_checker'] = self.train_symptom_checker_model()
        
        print("\n" + "="*50)
        print("Training Complete!")
        print("="*50)
        
        return results
    
    def get_model_info(self):
        """Get information about trained models"""
        info = {}
        for name in ['diabetes', 'heart_disease', 'symptom_checker']:
            model_path = os.path.join(self.models_dir, f'{name}_model.pkl')
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                info[name] = {
                    'model_type': type(model).__name__,
                    'n_estimators': getattr(model, 'n_estimators', 'N/A'),
                    'features': DIABETES_FEATURES if name == 'diabetes' else 
                               HEART_FEATURES if name == 'heart_disease' else SYMPTOM_FEATURES
                }
        return info


def load_model(name, models_dir='diabetes/model_files'):
    """Load a trained model"""
    model_path = os.path.join(models_dir, f'{name}_model.pkl')
    scaler_path = os.path.join(models_dir, f'{name}_scaler.pkl')
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    return None, None


def predict_diabetes(input_data, models_dir='diabetes/model_files'):
    """Make diabetes prediction using trained model"""
    model, scaler = load_model('diabetes', models_dir)
    
    if model is None:
        return None, "Model not found. Please train the model first."
    
    # Scale input
    input_scaled = scaler.transform([input_data])
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    return {
        'prediction': int(prediction),
        'probability': float(max(probability)),
        'diabetes_probability': float(probability[1]) if len(probability) > 1 else float(probability[0]),
        'result': 'Diabetes Positive' if prediction == 1 else 'Diabetes Negative'
    }


def predict_heart_disease(input_data, models_dir='diabetes/model_files'):
    """Make heart disease prediction using trained model"""
    model, scaler = load_model('heart_disease', models_dir)
    
    if model is None:
        return None, "Model not found. Please train the model first."
    
    # Scale input
    input_scaled = scaler.transform([input_data])
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    return {
        'prediction': int(prediction),
        'probability': float(max(probability)),
        'high_risk_probability': float(probability[1]) if len(probability) > 1 else float(probability[0]),
        'risk_level': 'High Risk' if prediction == 1 else 'Low Risk'
    }


def predict_symptoms(input_data, models_dir='diabetes/model_files'):
    """Make symptom prediction using trained model"""
    model, scaler = load_model('symptom_checker', models_dir)
    
    if model is None:
        return None, "Model not found. Please train the model first."
    
    # Scale input
    input_scaled = scaler.transform([input_data])
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    
    # Get top conditions
    top_indices = np.argsort(probabilities)[::-1][:3]
    conditions = []
    for idx in top_indices:
        conditions.append({
            'name': CONDITION_LABELS[idx],
            'probability': float(probabilities[idx])
        })
    
    return {
        'prediction': int(prediction),
        'primary_condition': CONDITION_LABELS[prediction],
        'confidence': float(max(probabilities)),
        'conditions': conditions
    }


if __name__ == "__main__":
    # Test training
    trainer = MLTrainer()
    results = trainer.train_all()
    
    print("\nTraining Results Summary:")
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
