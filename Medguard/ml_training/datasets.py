"""
ML Training Datasets Module
Contains training data for Diabetes, Heart Disease, and Symptom Checker ML models
"""

import numpy as np
import pandas as pd

# ==================== DIABETES DATASET ====================
# Based on Pima Indians Diabetes Database
# Features: Age, Gender, Polyuria, Polydipsia, Weight Loss, Weakness, Polyphagia, 
#           Genital Thrush, Visual Blurring, Itching, Irritability, Delayed Healing,
#           Partial Paresis, Muscle Stiffness, Alopecia, Obesity

DIABETES_FEATURES = [
    'Age', 'Gender', 'Polyuria', 'Polydipsia', 'sudden_weight_loss', 'weakness',
    'Polyphagia', 'Genital_thrush', 'visual_blurring', 'Itching', 'Irritability',
    'delayed_healing', 'partial_paresis', 'muscle_stiffness', 'Alopecia', 'Obesity'
]

# Sample training data (normalized/encoded values)
# 0 = No/False, 1 = Yes/True, Gender: 0=Female, 1=Male
DIABETES_BASE_DATA = {
    'features': [
        # Positive cases (Diabetes = 1)
        [45, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1],
        [52, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1],
        [38, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1],
        [60, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1],
        [42, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0],
        [55, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
        [48, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1],
        [35, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1],
        [58, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1],
        [41, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0],
        [50, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1],
        [44, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],
        [62, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [39, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
        [53, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1],
        [47, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        [56, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1],
        [40, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
        [59, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1],
        [43, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        
        # Negative cases (Diabetes = 0)
        [30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [25, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [35, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [28, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [42, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [33, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
        [38, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [29, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [45, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [31, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [27, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [36, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [41, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [26, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [34, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [39, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [32, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [29, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [44, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [37, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    ],
    'labels': [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # Positive
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0   # Negative
    ]
}


# ==================== HEART DISEASE DATASET ====================
# Features: age, sex, chest_pain_type, blood_pressure, cholesterol, 
#           fasting_blood_sugar, ecg, max_heart_rate, exercise_angina, st_depression

HEART_FEATURES = [
    'age', 'sex', 'chest_pain_type', 'blood_pressure', 'cholesterol',
    'fasting_blood_sugar', 'ecg', 'max_heart_rate', 'exercise_angina', 'st_depression'
]

# Sample heart disease training data
HEART_BASE_DATA = {
    'features': [
        # High risk cases (1)
        [63, 1, 0, 145, 233, 1, 2, 150, 1, 2.3],
        [67, 1, 1, 160, 286, 0, 2, 108, 1, 1.5],
        [67, 1, 0, 120, 229, 0, 2, 129, 1, 2.6],
        [37, 1, 3, 130, 250, 0, 0, 187, 0, 3.5],
        [41, 0, 2, 130, 204, 0, 2, 172, 0, 1.4],
        [56, 1, 1, 130, 256, 1, 0, 142, 1, 0.6],
        [62, 0, 3, 140, 268, 0, 2, 160, 0, 3.6],
        [57, 1, 0, 120, 354, 0, 2, 163, 1, 0.6],
        [63, 1, 1, 130, 254, 0, 2, 147, 1, 1.4],
        [53, 1, 0, 140, 203, 1, 2, 155, 1, 3.1],
        [57, 1, 1, 140, 192, 0, 0, 148, 0, 0.4],
        [58, 1, 2, 150, 283, 1, 2, 162, 0, 1.0],
        [50, 1, 0, 120, 219, 0, 0, 158, 0, 0.8],
        [54, 1, 3, 140, 239, 0, 1, 160, 1, 0.9],
        [60, 1, 1, 180, 294, 0, 2, 171, 0, 1.5],
        [59, 1, 0, 170, 288, 1, 2, 159, 0, 2.6],
        [52, 1, 1, 128, 225, 1, 1, 150, 1, 0.4],
        [55, 1, 3, 160, 289, 0, 2, 145, 1, 2.0],
        [61, 1, 2, 150, 243, 1, 0, 137, 1, 1.2],
        [58, 1, 0, 130, 197, 1, 2, 131, 1, 1.8],
        
        # Low risk cases (0)
        [34, 0, 2, 110, 183, 0, 0, 156, 0, 0.0],
        [29, 1, 1, 120, 198, 0, 1, 163, 0, 0.0],
        [35, 0, 2, 115, 190, 0, 0, 165, 0, 0.0],
        [28, 1, 0, 100, 170, 0, 0, 175, 0, 0.0],
        [32, 0, 1, 105, 180, 0, 1, 160, 0, 0.0],
        [38, 1, 2, 115, 195, 0, 0, 155, 0, 0.0],
        [30, 0, 0, 95, 165, 0, 0, 180, 0, 0.0],
        [36, 1, 1, 118, 188, 0, 1, 170, 0, 0.0],
        [33, 0, 2, 108, 178, 0, 0, 168, 0, 0.0],
        [31, 1, 0, 102, 172, 0, 0, 178, 0, 0.0],
        [40, 0, 1, 112, 185, 0, 0, 162, 0, 0.0],
        [27, 1, 2, 98, 168, 0, 1, 182, 0, 0.0],
        [39, 0, 0, 110, 190, 0, 1, 158, 0, 0.0],
        [34, 1, 1, 115, 195, 0, 0, 165, 0, 0.0],
        [37, 0, 2, 105, 180, 0, 0, 172, 0, 0.0],
        [29, 1, 0, 100, 175, 0, 1, 176, 0, 0.0],
        [42, 0, 1, 118, 200, 0, 0, 155, 0, 0.0],
        [35, 1, 2, 112, 188, 0, 1, 168, 0, 0.0],
        [38, 0, 0, 108, 182, 0, 0, 165, 0, 0.0],
        [33, 1, 1, 110, 190, 0, 0, 170, 0, 0.0],
    ],
    'labels': [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # High risk
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0   # Low risk
    ]
}

def _expand_binary_dataset(
    base_data,
    target_size,
    categorical_indices=None,
    numeric_bounds=None,
    random_state=42
):
    """
    Expand a binary dataset to target_size using class-balanced bootstrap
    with small perturbations for diversity.
    """
    if categorical_indices is None:
        categorical_indices = []
    if numeric_bounds is None:
        numeric_bounds = {}

    rng = np.random.default_rng(random_state)
    X = np.array(base_data['features'], dtype=float)
    y = np.array(base_data['labels'], dtype=int)

    classes = np.unique(y)
    if len(classes) != 2:
        return {
            'features': X.tolist(),
            'labels': y.tolist()
        }

    per_class = target_size // 2
    X_out = []
    y_out = []
    all_idx = np.arange(X.shape[1])
    numeric_indices = np.setdiff1d(all_idx, np.array(categorical_indices))

    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        sampled_idx = rng.choice(cls_idx, size=per_class, replace=True)
        X_cls = X[sampled_idx].copy()

        # Add subtle noise on numeric columns to avoid pure duplication.
        if len(numeric_indices) > 0:
            stds = X[:, numeric_indices].std(axis=0)
            stds = np.where(stds == 0, 1.0, stds)
            noise = rng.normal(
                loc=0.0,
                scale=0.05 * stds,
                size=(X_cls.shape[0], len(numeric_indices))
            )
            X_cls[:, numeric_indices] += noise

            # Clip numerics to medically plausible ranges.
            for i, col_idx in enumerate(numeric_indices):
                if col_idx in numeric_bounds:
                    lo, hi = numeric_bounds[col_idx]
                    X_cls[:, col_idx] = np.clip(X_cls[:, col_idx], lo, hi)

        # Keep categorical features as 0/1 or small encoded integers.
        for col_idx in categorical_indices:
            col = X_cls[:, col_idx]
            if col_idx in numeric_bounds:
                lo, hi = numeric_bounds[col_idx]
                col = np.clip(np.rint(col), lo, hi)
            else:
                col = np.clip(np.rint(col), 0, 1)
            X_cls[:, col_idx] = col

        X_out.append(X_cls)
        y_out.append(np.full(per_class, cls, dtype=int))

    X_final = np.vstack(X_out)
    y_final = np.concatenate(y_out)
    shuffle_idx = rng.permutation(len(X_final))
    X_final = X_final[shuffle_idx]
    y_final = y_final[shuffle_idx]

    return {
        'features': X_final.tolist(),
        'labels': y_final.tolist()
    }


# Expanded datasets for model training (1000 rows each).
DIABETES_DATA = _expand_binary_dataset(
    DIABETES_BASE_DATA,
    target_size=1000,
    categorical_indices=list(range(1, 16)),
    numeric_bounds={0: (18, 90)},
    random_state=42
)

HEART_DATA = _expand_binary_dataset(
    HEART_BASE_DATA,
    target_size=1000,
    categorical_indices=[1, 2, 5, 6, 8],
    numeric_bounds={
        0: (18, 90),   # age
        2: (0, 3),     # chest_pain_type
        3: (80, 220),  # blood_pressure
        4: (100, 600), # cholesterol
        5: (0, 1),     # fasting_blood_sugar
        6: (0, 2),     # ecg
        7: (60, 220),  # max_heart_rate
        8: (0, 1),     # exercise_angina
        9: (0, 10),    # st_depression
    },
    random_state=42
)


# ==================== SYMPTOM CHECKER DATASET ====================
# Symptom combinations and corresponding conditions

SYMPTOM_FEATURES = [
    'fever', 'cough', 'headache', 'fatigue', 'body_ache', 'nausea', 'vomiting',
    'diarrhea', 'chest_pain', 'shortness_breath', 'dizziness', 'sore_throat',
    'runny_nose', 'joint_pain', 'rash', 'abdominal_pain', 'loss_appetite'
]

# Condition labels
CONDITION_LABELS = [
    'Common Cold', 'Flu', 'COVID-19', 'Migraine', 'Food Poisoning',
    'Anxiety', 'Allergies', 'Stomach Flu', 'Tension Headache'
]

# Symptom checker training data (binary features for each symptom)
SYMPTOM_DATA = {
    'features': [
        # Common Cold
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
        [0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
        
        # Flu
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        
        # COVID-19
        [1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
        
        # Migraine
        [0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        
        # Food Poisoning
        [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        
        # Anxiety
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        
        # Allergies
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0],
        
        # Stomach Flu
        [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        
        # Tension Headache
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ],
    'labels': [
        0, 0, 0,  # Common Cold
        1, 1, 1,  # Flu
        2, 2, 2,  # COVID-19
        3, 3, 3,  # Migraine
        4, 4, 4,  # Food Poisoning
        5, 5, 5,  # Anxiety
        6, 6, 6,  # Allergies
        7, 7, 7,  # Stomach Flu
        8, 8, 8,  # Tension Headache
    ]
}


def get_diabetes_data():
    """Get diabetes training data as DataFrame"""
    df = pd.DataFrame(DIABETES_DATA['features'], columns=DIABETES_FEATURES)
    df['target'] = DIABETES_DATA['labels']
    return df


def get_heart_data():
    """Get heart disease training data as DataFrame"""
    df = pd.DataFrame(HEART_DATA['features'], columns=HEART_FEATURES)
    df['target'] = HEART_DATA['labels']
    return df


def get_symptom_data():
    """Get symptom checker training data as DataFrame"""
    df = pd.DataFrame(SYMPTOM_DATA['features'], columns=SYMPTOM_FEATURES)
    df['target'] = SYMPTOM_DATA['labels']
    return df
