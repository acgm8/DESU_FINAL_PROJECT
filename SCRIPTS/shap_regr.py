import shap
import matplotlib.pyplot as plt

def plot_shap_summary(pipe, X, y, feature_step, model_step, max_force_plots=3, class_names=None):
 
    # Fit the pipeline on the full dataset
    pipe.fit(X, y)

    # Transform X with preprocessing + feature selection
    X_transformed = pipe.named_steps[feature_step].transform(
        pipe.named_steps["scaler"].transform(
            pipe.named_steps["imputer"].transform(X)))

    # Feature names and model
    selected_features = pipe.named_steps[feature_step].get_feature_names()
    model = pipe.named_steps[model_step]

    # Choose SHAP explainer
    if hasattr(model, "coef_"):  # linear models
        explainer = shap.LinearExplainer(model, X_transformed, feature_names=selected_features)
        shap_values = explainer(X_transformed)
    elif "tree" in str(type(model)).lower():  # tree models
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)
    else:
        raise ValueError("Unsupported model type for SHAP.")

    # --- Handle classification vs regression ---
    if hasattr(shap_values, "values") and shap_values.values.ndim == 3:
        # Multiclass: shap_values.values is (n_samples, n_features, n_classes)
        n_classes = shap_values.values.shape[2]
        if class_names is None:
            class_names = [f"Class {i}" for i in range(n_classes)]

        for i in range(n_classes):
            print(f"\nSHAP summary for {class_names[i]}:")

            class_vals = shap_values.values[:, :, i]

            # Dot summary plot
            shap.summary_plot(class_vals, X_transformed, feature_names=selected_features, plot_type="dot", show=True)

            # Bar summary plot
            shap.summary_plot(class_vals, X_transformed, feature_names=selected_features, plot_type="bar", show=True)

            # Optional: force plots
            for j in range(min(max_force_plots, X_transformed.shape[0])):
                shap.force_plot(explainer.expected_value[i], class_vals[j, :],
                                feature_names=selected_features, matplotlib=True)
                plt.show()

    elif isinstance(shap_values, list):
        # TreeExplainer for multiclass
        for i, class_shap in enumerate(shap_values):
            print(f"\nSHAP summary for class {i}:")
            shap.summary_plot(class_shap, X_transformed, feature_names=selected_features, plot_type="dot", show=True)
            shap.summary_plot(class_shap, X_transformed, feature_names=selected_features, plot_type="bar", show=True)

    else:
        # Regression or binary classification
        print("\nSHAP summary (regression or binary classification):")
        shap.summary_plot(shap_values, X_transformed, feature_names=selected_features, plot_type="dot", show=True)
        shap.summary_plot(shap_values, X_transformed, feature_names=selected_features, plot_type="bar", show=True)
