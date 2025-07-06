import joblib
from sklearn.preprocessing import StandardScaler


def run_normalisation(features, config, mode="train"):
    original_shape = features.shape
    features = features.reshape(-1,features.shape[-1])
    if mode == "train":
        
        scaler = StandardScaler()
        features_norm = scaler.fit_transform(features)
        with open(config["scaler_path"], "wb") as f:
            joblib.dump(scaler, f)

        features_norm = features_norm.reshape(*original_shape)

        return features_norm
    
    elif mode=="test":
        with open(config["scaler_path"], "rb") as f:
            scaler = joblib.load(f)
        
        features_norm = scaler.transform(features)
        features_norm = features_norm.reshape(*original_shape)
        return features_norm
    
    else:
        raise Exception("Provide mode=train/test")

