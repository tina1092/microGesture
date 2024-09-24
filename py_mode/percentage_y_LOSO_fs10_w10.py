#@title Preparing Data
import os
import numpy as np
from sklearn.model_selection import train_test_split
import glob
is_windowed = True #@param{type: "boolean"}
window_size = 10 #@param
is_normalization = False #@param{type: "boolean"}
exclusion_list = ['p24', 'p26', 'p36'] #@param
inclusion_list = ['p38'] #@param
is_inclusion = False #@param{type: "boolean"}
is_y_percentage = True #@param{type: "boolean"}

def normalize(raw_data):
  mean = np.mean(raw_data, axis=0)
  std = np.std(raw_data, axis=0)
  normalized_data = (raw_data - mean) / (std + 1e-6)
  return normalized_data


# Function to create windowed data
def create_windowed_data(X, y, window_size=10):
    num_samples = X.shape[0] - window_size + 1  # Number of windows we can create
    windowed_X = np.array([(normalize(X[i:i + window_size]) if is_normalization else X[i:i + window_size]) for i in range(num_samples)])
    max_y = max(np.max(y), 1e-6)
    windowed_y = np.array([np.mean(y[i + window_size - 1])/max_y if is_y_percentage else np.mean(y[i + window_size - 1]) for i in range(num_samples)])  # target at the end of the window
    return windowed_X, windowed_y, max_y


def load_and_preprocess_data(data_path, users_list=None):
    data = None
    labels = None
    users = []
    sessions = []
    max_force_dict = {}
    # Walk through the folder structure
    for participant_folder in os.listdir(data_path):
      participant_path = os.path.join(data_path, participant_folder)
      user = os.path.basename(participant_path)
      if users_list and user not in users_list:
        continue
      if (user in exclusion_list):
        continue
      for session_id in ['1', '2', '3']:
        file_pattern = os.path.join(participant_path, f'*-{session_id}.npy')
        real = np.array([])
        imag = np.array([])
        force = np.array([])
        for file in glob.glob(file_pattern):
          if 'real' in file:
            with open(file, 'rb') as f:
              real = np.load(f)
          elif 'imag' in file:
            with open(file, 'rb') as f:
              imag = np.load(f)
          elif 'force' in file:
            with open(file, 'rb') as f:
              force = np.load(f)
        num_entries = real.shape[0]
        new_data = np.concatenate([real, imag], axis=1)
        new_force = force
        max_force = 0
        if is_windowed:
          new_data, new_force, max_force = create_windowed_data(new_data, force, window_size=window_size)
        if data is not None:
          data = np.append(data, new_data, axis=0)
        else:
          data = new_data
        labels = np.append(labels, new_force, axis=0) if labels is not None else new_force
        users.extend([user] * new_data.shape[0])
        sessions.extend([session_id] * num_entries)
        max_force_dict[(user, session_id)] = max_force

    return data, labels, np.array(users), np.array(sessions), max_force_dict

# Path to the extracted dataset
data_path = '/home/ubuntu/paper_arm_model/EIT_Model_Input_Resource/pinch-pressure/batch-2/#processed-data' #@param{type: "string"}
print(os.listdir(data_path))
# Load and preprocess the data
data, labels, users, sessions, max_force_dict = load_and_preprocess_data(data_path)

print(data.shape)
print(labels.shape)
print(max_force_dict)

# label_set = np.unique(labels)
print(np.unique(users))
print(np.unique(sessions))


#@title CNN
import shutil

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import LeaveOneGroupOut
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

is_classification = False #@param{type: "boolean"}
is_LOSO = True #@param{type: "boolean"}
is_finegrained = False #@param{type: "boolean"}
is_fewshot_finetuning = True #@param{type: "boolean"} # CHANGE
num_epoch_train = 30 #@param
num_epoch_finetune = 100 #@param
start_sample_index_finetune = 3000 #@param
num_samples_finetune = 1000 #@param
is_finetune_whole_sesion = False #@param{type: "boolean"}
low_pressure_boundary = 4  if not is_y_percentage else 0.4 #@param
high_pressure_boundary = 7 if not is_y_percentage else 0.7 #@param
checkpoint_filepath = '/home/ubuntu/paper_arm_model/0909/tmp/cnn_regressor_best_model.keras' #@param{type: "string"}
user_list = [] #@param

target_names = [str(i) for i in range(10)] if is_finegrained else ['light', 'median', 'hard']
model_name = 'percentage_y_LOSO_fs10_w10' #@param{type: "string"} #percentage_y_LOSO_fs20_w10
model_dir = f'/home/ubuntu/paper_arm_model/0909/pinch-force-v2-models/{model_name}/'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
def classify_pressure(pressure):
  if is_finegrained:
    if pressure/0.1 >= 9.5:
      return 9
    else:
      return int(np.round(pressure/0.1))
  else:
    if pressure == 0:
      return 0
    elif pressure < low_pressure_boundary:
      return 1
    elif pressure < high_pressure_boundary:
      return 2
    else:
      return 3

# Convert labels to indices
labels_c = np.array([classify_pressure(y_i) for y_i in labels])
label_set = np.unique(labels_c)
labels_used = labels
if is_classification:
  print(label_set)
  labels_used = to_categorical(labels_c, num_classes=len(label_set))

logo = LeaveOneGroupOut()
logo_session = LeaveOneGroupOut()
fold_accuracies = []
LOSO_result = {}



def create_model():
  # Define the path to save the best model
  # Create a ModelCheckpoint callback to save the model with the lowest validation loss

  model = Sequential()
  # model.add(BatchNormalization())
  model.add(Conv1D(32, kernel_size=3, activation='relu'))
  model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(window_size,
  12)))
  # model.add(BatchNormalization())
  # model.add(MaxPooling1D(pool_size=2))
  model.add(Conv1D(128, kernel_size=3, activation='relu'))
  # model.add(BatchNormalization())
  # model.add(MaxPooling1D(pool_size=2))
  model.add(GlobalAveragePooling1D())
  model.add(Dense(64, activation='relu'))
  model.add(Dense(32, activation='relu'))
  model.add(Dense(16, activation='relu'))
  # model.add(BatchNormalization())
  # model.add(Dense(len(label_set), activation='softmax'))
  if is_classification:
    model.add(Dense(len(label_set), activation='softmax'))
  elif is_y_percentage:
    print("sigmoid")
    model.add(Dense(1, activation='sigmoid'))
  else:
    model.add(Dense(1, activation='linear'))
  model.compile(optimizer='adam', loss='mse' if not is_classification else 'categorical_crossentropy', metrics=['mae' if not is_classification else 'accuracy'])

  return model

if not is_LOSO:
  X_train, X_test, y_train, y_test = train_test_split(data, labels_used, test_size=0.2, random_state=42)
  model = create_model()
  model.compile(optimizer='adam', loss='mse' if not is_classification else 'categorical_crossentropy', metrics=['mae' if not is_classification else 'accuracy'])
  checkpoint = ModelCheckpoint(
      filepath=checkpoint_filepath,
      monitor='val_loss' if not is_classification else 'val_accuracy',  # You can also use 'val_accuracy' to monitor accuracy instead
      verbose=0,
      save_best_only=True,
      mode='min' if not is_classification else 'max'  # 'min' for minimizing loss, 'max' for maximizing accuracy
  )
  history = model.fit(X_train, y_train, epochs=num_epoch_train, batch_size=32, validation_data=(X_test, y_test), callbacks=[checkpoint])

else:
  for train_index, test_index in logo.split(data, labels_used, groups=users):
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = labels_used[train_index], labels_used[test_index]
    print(users[test_index])
    if user_list and users[test_index][0] not in user_list:
      continue
    model = create_model()
    checkpoint = ModelCheckpoint(
      filepath=checkpoint_filepath,
      monitor='val_loss' if not is_classification else 'val_accuracy',  # You can also use 'val_accuracy' to monitor accuracy instead
      verbose=0,
      save_best_only=True,
      mode='min' if not is_classification else 'max'  # 'min' for minimizing loss, 'max' for maximizing accuracy
      )
    history = model.fit(X_train, y_train, epochs=num_epoch_train, batch_size=512, validation_data=(X_test, y_test), callbacks=[checkpoint])

    if is_fewshot_finetuning:
      for fs_train_index, fs_test_index in logo_session.split(X_test, y_test, groups=sessions[test_index]):
        fs_X_train, fs_X_test = X_test[fs_train_index], X_test[fs_test_index]
        fs_y_train, fs_y_test = y_test[fs_train_index], y_test[fs_test_index]
        if not is_finetune_whole_sesion:
          used_sessions = sessions[test_index][fs_test_index]
          print(f"LOSO session: {len(used_sessions)}")
          used_idx = list(range(start_sample_index_finetune, min(start_sample_index_finetune + num_samples_finetune, len(X_test[fs_test_index]))))
          fs_X_train, fs_X_test = X_test[fs_train_index], X_test[fs_test_index][used_idx]
          fs_y_train, fs_y_test = y_test[fs_train_index], y_test[fs_test_index][used_idx]

        fs_history = model.fit(fs_X_test, fs_y_test, epochs=num_epoch_finetune, batch_size=32, validation_data=(X_test, y_test), callbacks=[checkpoint])
        break
    # Evaluate the model
    cur_best_model = tf.keras.models.load_model(checkpoint_filepath)
    test_loss, test_acc = cur_best_model.evaluate(X_test, y_test)

    model_path = os.path.join(model_dir, f'{users[test_index][0]}_cnn_{model_name}_regresser_best_model.keras')
    shutil.copy(checkpoint_filepath, model_path)
    LOSO_result[users[test_index][0]] = (test_loss, test_acc)
    fold_accuracies.append(test_loss if not is_classification else test_acc)

avg_accuracy = np.mean(fold_accuracies)
print(f"Average LOSO metric: {avg_accuracy}")


#@title CNN Evaluation
import joblib

from sklearn.metrics import mean_squared_error
import tensorflow as tf
import matplotlib.pyplot as plt

is_y_percentage = False #@param{type: "boolean"}
model_name = "p28_percentage_y_LOSO_fs_20_w20" #@param{type: "string"}
user = 'p28'  #@param{type: "string"}
saved_model_path = f'/home/ubuntu/paper_arm_model/0909/pinch-force-v2-models/{model_name}/{user}_cnn_{model_name}_regresser_best_model.keras' #@param
is_saved_to_csv = False #@param{type: "boolean"}
saved_csv_path = f'/home/ubuntu/paper_arm_model/0909/pinch-force-v2-models/{user}_result.csv' #@param

print(LOSO_result)
print(np.mean(fold_accuracies))
# with gfile.Open(saved_model_path, 'rb') as f:
loaded_model = tf.keras.models.load_model(saved_model_path)


# X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
user_data, user_labels, _, _, _ = load_and_preprocess_data(data_path, users_list=[user])


# X_test = X_test.reshape(X_test.shape[0], window_size * 12)
y_pred_loaded = loaded_model.predict(user_data)
mse = mean_squared_error(user_labels, y_pred_loaded)
print(mse)


data_to_plot_1 = y_pred_loaded
data_to_plot_2 = user_labels
num_samples = 10000 #@param
num_samples = min(num_samples, len(data_to_plot_1))
plt.figure(figsize=(20, 6))
x = range(len(data_to_plot_1))
print(data_to_plot_1.shape)
plt.plot(x[:num_samples], data_to_plot_1[:num_samples], marker='s', label='predictions')
plt.plot(x[:num_samples], data_to_plot_2[:num_samples], marker='o', label='groundtruth')
plt.legend()
plt.show()

if is_saved_to_csv:
  np_to_csv = np.vstack([data_to_plot_1, data_to_plot_2])
  print(np_to_csv.shape)
  with open(saved_csv_path, 'wb') as f:
    np.savetxt(f, np_to_csv, delimiter=",")