import shap
import matplotlib.pyplot as plt

def plot_shap_summary(pipe, X, y, feature_step, model_step):
    
    # Fit the pipeline on the whole dataset once
    pipe.fit(X,y)

    # Get the transformed feature matrix after feature selection
    X_transformed = pipe.named_steps[feature_step].transform(
        pipe.named_steps["scaler"].transform(
            pipe.named_steps["imputer"].transform(X)))

    # Get the selected feature names and which model we're using
    selected_features = pipe.named_steps[feature_step].get_feature_names()
    model = pipe.named_steps[model_step]

    # SHAP Explainer: detects if the model is linera or tree based, then if it's classification or regression

    if hasattr(model, "coef_"): # Linear models (Lasso, Ridge, ElasticNet, LogisticRegression, SGDClassifier)
        explainer = shap.LinearExplainer(model, X_transformed, feature_names=selected_features)
        shap_values = explainer(X_transformed)

    elif "tree" in str(type(model)).lower(): # Tree models (DecisionTree, RandomForest)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)

    # Checks if it's classification or regression
    if isinstance(shap_values, list):
        for i, class_shap in enumerate(shap_values): #classification
            print(f"SHAP summary for class {i}:")
            shap.summary_plot(class_shap, X_transformed, feature_names=selected_features)
    else:      
        # Regression or binary classification
        shap.summary_plot(shap_values, X_transformed, feature_names=selected_features)
        
