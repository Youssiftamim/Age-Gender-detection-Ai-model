import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Suppress oneDNN warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

DATASET_PATH = "D:/Age & gender/UTKFace/"
IMG_SIZE = (48, 48)
BATCH_SIZE = 32

def categorize_age(age):
    return min(age // 10, 6)

def process_image(file_entry):
    filename = file_entry.name
    try:
        parts = filename.split("_")
        if len(parts) < 3:
            return None
        age, gender = int(parts[0]), int(parts[1])
        age_category = categorize_age(age)
        img_path = os.path.join(DATASET_PATH, filename)
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = img / 255.0
        return img, age_category, gender
    except Exception:
        return None

def load_dataset():
    file_list = [entry for entry in os.scandir(DATASET_PATH) if entry.is_file()]
    dataset = tf.data.Dataset.from_generator(
        lambda: (process_image(f) for f in file_list),
        output_signature=(
            tf.TensorSpec(shape=(*IMG_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    dataset = dataset.filter(lambda x, y1, y2: x is not None)
    return dataset

dataset = load_dataset()
data = list(dataset.as_numpy_iterator())
X, y_age, y_gender = zip(*data)
X, y_age, y_gender = np.array(X), np.array(y_age), np.array(y_gender)

# Print gender distribution
gender_counts = np.bincount(y_gender)
print("Gender distribution (Male, Female):", gender_counts)

# Compute class weights for gender (for reference only)
gender_class_weights = compute_class_weight('balanced', classes=np.unique(y_gender), y=y_gender)
gender_class_weights_dict = {0: gender_class_weights[0], 1: gender_class_weights[1]}
print("Gender class weights:", gender_class_weights_dict)

# Compute class weights for age (for reference only)
age_class_weights = compute_class_weight('balanced', classes=np.arange(7), y=y_age)
age_class_weights_dict = {i: weight for i, weight in enumerate(age_class_weights)}
print("Age class weights:", age_class_weights_dict)

# Oversample the female class to balance the dataset
male_indices = np.where(y_gender == 0)[0]
female_indices = np.where(y_gender == 1)[0]
num_males = len(male_indices)
num_females = len(female_indices)
num_to_oversample = num_males - num_females  # Number of female samples to add

if num_to_oversample > 0:
    # Randomly sample female indices to oversample
    oversampled_female_indices = np.random.choice(female_indices, size=num_to_oversample, replace=True)
    # Concatenate the oversampled indices with the original indices
    all_indices = np.concatenate([np.arange(len(X)), oversampled_female_indices])
    X = X[all_indices]
    y_age = y_age[all_indices]
    y_gender = y_gender[all_indices]
    print("After oversampling - Gender distribution (Male, Female):", np.bincount(y_gender))

y_age = to_categorical(y_age, num_classes=7)
y_gender = to_categorical(y_gender, num_classes=2)

X_train, X_test, y_train_age, y_test_age, y_train_gender, y_test_gender = train_test_split(
    X, y_age, y_gender, test_size=0.2, random_state=42
)

def create_dataset(X, y_age, y_gender, training=False):
    dataset = tf.data.Dataset.from_tensor_slices((X, {'age_output': y_age, 'gender_output': y_gender}))
    if training:
        dataset = dataset.map(
            lambda x, y: (
                tf.image.random_flip_left_right(
                    tf.image.random_brightness(
                        tf.image.random_contrast(x, 0.9, 1.1), max_delta=0.1
                    )
                ),
                y
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    dataset = dataset.cache().shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = create_dataset(X_train, y_train_age, y_train_gender, training=True)
test_dataset = create_dataset(X_test, y_test_age, y_test_gender, training=False)

inputs = Input(shape=(*IMG_SIZE, 3))
x = SeparableConv2D(32, (3, 3), padding='same')(inputs)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = SeparableConv2D(64, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = SeparableConv2D(128, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)

age_output = Dense(7, activation='softmax', name='age_output')(x)
gender_output = Dense(2, activation='softmax', name='gender_output')(x)

model = Model(inputs=inputs, outputs=[age_output, gender_output])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss={'age_output': 'categorical_crossentropy', 'gender_output': 'categorical_crossentropy'},
    loss_weights={'age_output': 1.5, 'gender_output': 1.0},
    metrics=['accuracy']
)

early_stopping = EarlyStopping(monitor='val_gender_output_accuracy', patience=7, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)

history = model.fit(
    train_dataset,
    epochs=50,
    validation_data=test_dataset,
    callbacks=[early_stopping, lr_scheduler],
    verbose=1
    # Removed class_weight
)

test_results = model.evaluate(test_dataset, return_dict=True)
print(f"Test Accuracy - Age: {test_results['age_output_accuracy'] * 100:.2f}%, Gender: {test_results['gender_output_accuracy'] * 100:.2f}%")

model_path = "D:/calc/3.11/age_gender_classification_model_v4_cpu.keras"
model.save(model_path)
if os.path.exists(model_path):
    print(f"Model saved successfully to: {model_path}")
else:
    print("Error: Model was not saved!")