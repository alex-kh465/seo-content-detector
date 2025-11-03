import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import joblib
import os


def create_quality_labels(df):
    """
    Create synthetic quality labels based on word count and readability.
    
    Rules:
    - High: word_count > 1500 and 50 <= readability <= 70
    - Low: word_count < 500 or readability < 30
    - Medium: otherwise
    
    Args:
        df: DataFrame with word_count and flesch_reading_ease columns
        
    Returns:
        Series with quality labels
    """
    labels = []
    
    for _, row in df.iterrows():
        word_count = row['word_count']
        readability = row['flesch_reading_ease']
        
        if word_count > 1500 and 50 <= readability <= 70:
            labels.append('High')
        elif word_count < 500 or readability < 30:
            labels.append('Low')
        else:
            labels.append('Medium')
    
    return pd.Series(labels)


def baseline_predictor(word_count):
    """
    Simple rule-based baseline predictor based on word count.
    
    Args:
        word_count: Word count value
        
    Returns:
        Quality label
    """
    if word_count > 1500:
        return 'High'
    elif word_count < 500:
        return 'Low'
    else:
        return 'Medium'


def train_quality_model(features_csv_path, model_output_path, test_size=0.3, random_state=42):
    """
    Train a RandomForestClassifier to predict content quality.
    
    Args:
        features_csv_path: Path to CSV with features
        model_output_path: Path to save trained model
        test_size: Proportion of data for testing (default: 0.3)
        random_state: Random seed for reproducibility
        
    Returns:
        tuple: (model, X_test, y_test, y_pred, feature_names, evaluation_metrics)
    """
    print(f"Loading features from {features_csv_path}...")
    df = pd.read_csv(features_csv_path)
    
    # Create synthetic labels
    print("Creating quality labels...")
    df['quality_label'] = create_quality_labels(df)
    
    # Display label distribution
    print("\nLabel distribution:")
    print(df['quality_label'].value_counts())
    
    # Prepare features
    feature_columns = ['word_count', 'sentence_count', 'flesch_reading_ease']
    X = df[feature_columns].values
    y = df['quality_label'].values
    
    # Train/test split
    print(f"\nSplitting data (train: {int((1-test_size)*100)}%, test: {int(test_size*100)}%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train Random Forest model
    print("\nTraining RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Baseline predictions
    y_baseline = [baseline_predictor(wc) for wc in X_test[:, 0]]
    
    # Evaluation
    print("\n=== Model Evaluation ===")
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    baseline_accuracy = accuracy_score(y_test, y_baseline)
    baseline_f1 = f1_score(y_test, y_baseline, average='weighted')
    
    print(f"\nRandomForest Model:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score (weighted): {f1:.4f}")
    
    print(f"\nBaseline (rule-based):")
    print(f"  Accuracy: {baseline_accuracy:.4f}")
    print(f"  F1 Score (weighted): {baseline_f1:.4f}")
    
    print(f"\nImprovement over baseline:")
    print(f"  Accuracy: +{(accuracy - baseline_accuracy):.4f}")
    print(f"  F1 Score: +{(f1 - baseline_f1):.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=['Low', 'Medium', 'High'])
    print("\nConfusion Matrix:")
    print(cm)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Save model
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"\nModel saved to {model_output_path}")
    
    # Save predictions to features CSV
    df_with_predictions = df.copy()
    
    # Predict on entire dataset
    X_full = df[feature_columns].values
    df_with_predictions['predicted_quality'] = model.predict(X_full)
    df_with_predictions.to_csv(features_csv_path, index=False)
    print(f"Predictions added to {features_csv_path}")
    
    evaluation_metrics = {
        'accuracy': accuracy,
        'f1_score': f1,
        'baseline_accuracy': baseline_accuracy,
        'baseline_f1': baseline_f1,
        'confusion_matrix': cm,
        'feature_importance': feature_importance
    }
    
    return model, X_test, y_test, y_pred, feature_columns, evaluation_metrics


def predict_quality(model, word_count, sentence_count, flesch_reading_ease):
    """
    Predict quality label for given features.
    
    Args:
        model: Trained RandomForestClassifier
        word_count: Number of words
        sentence_count: Number of sentences
        flesch_reading_ease: Readability score
        
    Returns:
        Quality label (High/Medium/Low)
    """
    features = np.array([[word_count, sentence_count, flesch_reading_ease]])
    prediction = model.predict(features)[0]
    return prediction


if __name__ == "__main__":
    train_quality_model('data/features.csv', 'streamlit_app/models/quality_model.pkl')
