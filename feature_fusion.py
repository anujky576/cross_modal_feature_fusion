import numpy as np
import pandas as pd
import lightgbm as lgb
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def train_emoticon_model(df_train_emoticon, df_val_emoticon, df_test_emoticon):
    print("Training emoticon model....")
    all_characters = ''.join(df_train_emoticon['input_emoticon'])
    char_counts = Counter(all_characters)
    sorted_char_counts = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)

    unicode_chars = [chr(i) for i in range(128512, 128738)]
    char_to_index = {char: idx for idx, char in enumerate(unicode_chars)}

    chars_to_remove = [char for char, _ in sorted_char_counts[:7]]

    def remove_characters(s, chars_to_remove):
        return ''.join([char for char in s if char not in chars_to_remove])

    df_train_emoticon['input_emoticon'] = df_train_emoticon['input_emoticon'].apply(lambda x: remove_characters(x, chars_to_remove))
    df_test_emoticon['input_emoticon'] = df_test_emoticon['input_emoticon'].apply(lambda x: remove_characters(x, chars_to_remove))
    df_val_emoticon['input_emoticon'] = df_val_emoticon['input_emoticon'].apply(lambda x: remove_characters(x, chars_to_remove))

    # String Encoding
    def encode_string(s, char_to_index):
        return [char_to_index[char] if char in char_to_index else -1 for char in s]

    df_train_emoticon['encoded_input'] = df_train_emoticon['input_emoticon'].apply(lambda x: encode_string(x, char_to_index))
    df_val_emoticon['encoded_input'] = df_val_emoticon['input_emoticon'].apply(lambda x: encode_string(x, char_to_index))
    df_test_emoticon['encoded_input'] = df_test_emoticon['input_emoticon'].apply(lambda x: encode_string(x, char_to_index))

    def expand_encoded_columns(encoded_list):
        return (encoded_list + [-1] * 3)[:3]

    df_train_emoticon[['encoded_1', 'encoded_2', 'encoded_3']] = df_train_emoticon['encoded_input'].apply(expand_encoded_columns).apply(pd.Series)
    df_val_emoticon[['encoded_1', 'encoded_2', 'encoded_3']] = df_val_emoticon['encoded_input'].apply(expand_encoded_columns).apply(pd.Series)
    df_test_emoticon[['encoded_1', 'encoded_2', 'encoded_3']] = df_test_emoticon['encoded_input'].apply(expand_encoded_columns).apply(pd.Series)

    df_train_emoticon.drop(columns=['encoded_input'], inplace=True)
    df_val_emoticon.drop(columns=['encoded_input'], inplace=True)
    df_test_emoticon.drop(columns=['encoded_input'], inplace=True)

    X_train_emoticon = df_train_emoticon.drop(columns=['input_emoticon', 'label'])
    y_train_emoticon = df_train_emoticon['label']

    X_val_emoticon = df_val_emoticon.drop(columns=['input_emoticon', 'label'])
    y_val_emoticon = df_val_emoticon['label']

    X_test_emoticon = df_test_emoticon.drop(columns=['input_emoticon'])

    # LGB
    params_emoticon = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.45,
        'feature_fraction': 0.3,
        'verbose': -1,
    }

    num_round_emoticon = 105

    model_emoticon = lgb.train(params_emoticon, lgb.Dataset(X_train_emoticon, label=y_train_emoticon), num_round_emoticon)
    print("Emoticon model trained successfully.\n")
    
    return model_emoticon, X_train_emoticon, y_train_emoticon, X_val_emoticon, y_val_emoticon, X_test_emoticon



def train_text_seq_model(df_train_textseq, df_val_textseq, df_test_textseq):
    print("Training text-seq model....")
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(df_train_textseq.input_str)

    df_train_textseq.input_str = [s[3:] for s in df_train_textseq.input_str]
    df_val_textseq.input_str = [s[3:] for s in df_val_textseq.input_str]

    X_train_seq = tokenizer.texts_to_sequences(df_train_textseq.input_str)
    X_train_padded = pad_sequences(X_train_seq, maxlen=47, padding='post', truncating='post')

    X_val_seq = tokenizer.texts_to_sequences(df_val_textseq.input_str)
    X_val_padded = pad_sequences(X_val_seq, maxlen=47, padding='post', truncating='post')

    X_test_seq = tokenizer.texts_to_sequences(df_test_textseq.input_str)
    X_test_padded = pad_sequences(X_test_seq, maxlen=47, padding='post', truncating='post')

    X_train_textseq = X_train_padded
    X_val_textseq = X_val_padded
    X_test_textseq = X_test_padded

    y_train_textseq = df_train_textseq.label
    y_val_textseq = df_val_textseq.label

    #LSTM model
    model_lstm = Sequential()
    model_lstm.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=64))
    model_lstm.add(LSTM(8))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(Dense(1, activation='sigmoid'))

    model_lstm.compile( loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
    history=model_lstm.fit(X_train_textseq, y_train_textseq, epochs=150, batch_size=32, validation_data=(X_val_textseq, y_val_textseq))
    print(model_lstm.summary())
    print("\n Text-seq model trained successfully.\n")

    return model_lstm, X_train_textseq, y_train_textseq, X_val_textseq, y_val_textseq, X_test_textseq



def train_feat_model(df_train_npz, df_val_npz, df_test_npz):
    print("Training Deep Features model....")
    X_train = df_train_npz['features']
    y_train_npz = df_train_npz['label']

    X_val = df_val_npz['features']
    y_val_npz = df_val_npz['label']

    X_test = df_test_npz['features']

    def concatenate_features(X):
        return X.reshape(X.shape[0], -1)

    X_train_concat = concatenate_features(X_train)
    X_val_concat = concatenate_features(X_val)
    X_test_concat = concatenate_features(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_concat)
    X_val_scaled = scaler.transform(X_val_concat)
    X_test_scaled = scaler.transform(X_test_concat)

    # PCA Analysis
    pca = PCA(n_components=300)
    X_train_npz = pca.fit_transform(X_train_scaled)
    X_val_npz = pca.transform(X_val_scaled)
    X_test_npz = pca.transform(X_test_scaled)

    params_npz = {
      'objective': 'binary',
      'metric': 'binary_logloss',
      'boosting_type': 'gbdt',
      'num_leaves': 31,
      'learning_rate': 0.1,
      'feature_fraction': 0.3,
    }
    num_round_npz = 105
    lgb_train_npz = lgb.Dataset(X_train_npz, label=y_train_npz)
    model_npz = lgb.train(params_npz, lgb_train_npz, num_round_npz)
    print("Deep features model trained successfully.\n")

    return model_npz, X_train_npz, y_train_npz, X_val_npz, y_val_npz, X_test_npz



def train_combined_model(X_train_emoticon, X_train_feat, X_train_seq, y_train_emoticon):
    print("Combining all three datasets....")
    X_train_final = pd.concat([pd.DataFrame(X_train_feat), pd.DataFrame(X_train_seq), pd.DataFrame(X_train_emoticon)], axis=1)
    X_val_final = pd.concat([pd.DataFrame(X_val_feat), pd.DataFrame(X_val_seq), pd.DataFrame(X_val_emoticon)], axis=1)
    X_test_final = pd.concat([pd.DataFrame(X_test_feat), pd.DataFrame(X_test_seq), pd.DataFrame(X_test_emoticon)], axis=1)

    column_names = [f"feature_{i}" for i in range(0, 350)]
    X_train_final.columns = column_names
    X_val_final.columns = column_names
    X_test_final.columns = column_names

    y_train_final = y_train_emoticon

    scaler = StandardScaler()
    X_train_final = scaler.fit_transform(X_train_final)
    X_val_final = scaler.transform(X_val_final)
    X_test_final = scaler.transform(X_test_final)

    # LGB
    params_final = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.3,
        'verbose': -1,
    }
    num_round_final = 105
    lgb_train_final = lgb.Dataset(X_train_final, label=y_train_final)
    model_final = lgb.train(params_final, lgb_train_final, num_round_final)
    print("Datasets combined and model trained successfully.\n")

    return model_final, X_test_final



if __name__ == '__main__':
    
    #read emoticon dataset
    train_emoticon_df = pd.read_csv("datasets/train/train_emoticon.csv")
    val_emoticon_df = pd.read_csv("datasets/valid/valid_emoticon.csv")
    test_emoticon_df = pd.read_csv("datasets/test/test_emoticon.csv")

    # read text sequence dataset
    train_seq_df = pd.read_csv("datasets/train/train_text_seq.csv")
    val_seq_df = pd.read_csv("datasets/valid/valid_text_seq.csv")
    test_seq_df= pd.read_csv("datasets/test/test_text_seq.csv")

    # read feature dataset
    train_feat = np.load("datasets/train/train_feature.npz", allow_pickle=True)
    val_feat = np.load("datasets/valid/valid_feature.npz", allow_pickle=True)
    test_feat = np.load("datasets/test/test_feature.npz", allow_pickle=True)
    
    # Training each model
    model_emoticon, X_train_emoticon, y_train_emoticon, X_val_emoticon, y_val_emoticon, X_test_emoticon = train_emoticon_model(train_emoticon_df, val_emoticon_df, test_emoticon_df)
    model_seq, X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq = train_text_seq_model(train_seq_df, val_seq_df, test_seq_df)
    model_feat, X_train_feat, y_train_feat, X_val_feat, y_val_feat, X_test_feat = train_feat_model(train_feat, val_feat, test_feat)
    
    # Training the combined model
    model_combined, X_test_combined = train_combined_model(X_train_emoticon, X_train_feat, X_train_seq, y_train_emoticon)

    # Predict from each model
    print("Ready to predict..")
    y_pred_emoticon = model_emoticon.predict(X_test_emoticon)
    y_pred_emoticon = [1 if x >= 0.5 else 0 for x in y_pred_emoticon]
    y_pred_textseq = model_seq.predict(X_test_seq)
    y_pred_textseq = [1 if x >= 0.5 else 0 for x in y_pred_textseq]
    y_pred_feat = model_feat.predict(X_test_feat)
    y_pred_feat = [1 if x >= 0.5 else 0 for x in y_pred_feat]
    y_pred_combined = model_combined.predict(X_test_combined)
    y_pred_combined = [1 if x >= 0.5 else 0 for x in y_pred_combined]
    
    print("Saving Predictions to .txt files...")
    with open("pred_emoticon.txt", "w") as f_emoticon:
        for pred in y_pred_emoticon:
            f_emoticon.write(f"{pred}\n")

    with open("pred_textseq.txt", "w") as f_textseq:
        for pred in y_pred_textseq:
            f_textseq.write(f"{pred}\n")

    with open("pred_deepfeat.txt", "w") as f_feat:
        for pred in y_pred_feat:
            f_feat.write(f"{pred}\n")

    with open("pred_combined.txt", "w") as f_combined:
        for pred in y_pred_combined:
            f_combined.write(f"{pred}\n")

    print("Predictions saved to text files.")