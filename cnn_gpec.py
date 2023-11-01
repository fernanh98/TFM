import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import imageio
import imgaug.augmenters as iaa

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# Import DataFrame
file_path = '../gpec_dataset.xlsx'
dataset = pd.read_excel(file_path)
dataset.head()

# Shuffle the dataset and create one-hot-encoding columns
dataset = dataset.sample(frac=1, random_state=42)
dataset['neg'] = [1 if val==0 else 0 for val in dataset['PDL1 label']]
dataset['pos'] = [1 if val==1 else 0 for val in dataset['PDL1 label']]

# Check proportion of Pdl1 Status on dataset
print('Percentage of PDL1 status', dataset['PDL1 label'].value_counts(normalize=True))
print('---------------------------------','\n','Number of images: ',len(dataset))

from sklearn.model_selection import StratifiedShuffleSplit

# Split data into train and test set
split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
for train_index, test_index in split.split(dataset, dataset['PDL1 label']):
    train_set1 = dataset.iloc[train_index]
    test_set = dataset.iloc[test_index]

# Split train data into train and validation
split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
for train_index, test_index in split.split(train_set1, train_set1['PDL1 label']):
    train_set = train_set1.iloc[train_index]
    val_set = train_set1.iloc[test_index]

print('Train data', train_set['PDL1 label'].value_counts(normalize=True))
print('---------------------------------','\n','Number of train images: ',len(train_set))
print('---------------------------------')
print('Test data', test_set['PDL1 label'].value_counts(normalize=True))
print('---------------------------------','\n','Number of test images: ',len(test_set))
print('---------------------------------')
print('Validation data', val_set['PDL1 label'].value_counts(normalize=True))
print('---------------------------------','\n','Number of validation images: ',len(val_set))


# Image Preprocess
def preprocess_image(path, target_size, gamma, saturation, std, sigma, plot, train):

    # Load image as a Numpy Matrix
    image = imageio.v2.imread(path)

    # Crop image
    image_crop = iaa.CenterCropToSquare()(images=[image])[0]

    # Resize image
    image_resize = iaa.Resize({"height": target_size,
                               "width": target_size})(images=[image_crop])[0]
    if train:
         # Random rotation
        if tf.random.uniform(()) > 0.5:
            angle = tf.random.uniform(shape=[], minval=0, maxval=361, dtype=tf.float32)
            image_rotated = iaa.Rot90(int(angle / 90))(images=[image_resize])[0]
        else:
            image_rotated = image_resize

        # Randomly flip
        image_flip = iaa.Flipud(0.5)(images=[image_rotated])[0]
        image_flip = iaa.Fliplr(0.5)(images=[image_flip])[0]

        # Gamma
        image_gamma = iaa.GammaContrast((0.0, gamma))(images=[image_flip])[0]

        # Saturation and Temperature
        image_sat = iaa.MultiplySaturation((0, saturation))(images=[image_gamma])[0]
        image_temp = iaa.ChangeColorTemperature((2000, 8000))(images=[image_sat])[0]

        # Gaussian Noise
        noisy_image = iaa.AdditiveGaussianNoise(scale=(0, std))(images=[image_temp])[0]

        # Blur
        image_blur = iaa.GaussianBlur(sigma=(0.0, sigma))(images=[noisy_image])[0]

    else:
        return image_resize


    # Plot images
    if plot:
        plt.figure(figsize=(20, 15))

        images = {'image':image, 'image_crop':image_crop, 'image_rotated':image_rotated,
                  'image_flipped':image_flip, 'image_gamma':image_gamma,
                  'image_saturation':image_sat, 'image_delta':image_temp,
                  'noisy_image':noisy_image, 'image_blur':image_blur}

        cols = 3
        rows = int(len(images) / cols)

        for index, (key, val) in enumerate(images.items()):
            plt.subplot(rows, cols, index + 1)
            plt.imshow(val)
            if index == 0:
                plt.title(f'{key}')
            else:
                plt.title(f'{key} [{np.min(val)} , {np.max(val)}]')
        plt.show()

    return image_blur

# Get data
def get_data(df, path = '../gpec_images/', target_size=224, gamma=0.0, saturation=0, std=0,
                    sigma=0.0, plot=False, train=False):

    images = df['path']
    images_data = []

    for img in images:
        # Get path
        img_folder = img.split('_')
        img_folder.pop()
        folder = '_'.join(img_folder)
        image_path = path + folder + '/' + img

        # Preprocess image
        image = preprocess_image(image_path, gamma=gamma, saturation=saturation, plot=plot,
                                train=train, std=std, sigma=sigma, target_size=target_size) / 255
        # Save image
        images_data.append(image)

    images = np.array(images_data)
    labels = np.array(df[['neg', 'pos']].astype(float))

    return images, labels

# Get train, test and validation sets
train_images, train_labels = get_data(train_set, train = True, gamma = 3, std = 20, saturation = 2.0, sigma = 2.0)# sigma=4
test_images, test_labels = get_data(test_set)
val_images, val_labels = get_data(val_set)

print('Preprocesamiento completado')

# Convolutional Neural Network
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2, VGG16, ResNet152V2, DenseNet121
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, SeparableConv2D
from tensorflow.keras.layers import BatchNormalization, Input, ReLU, GlobalAveragePooling2D, Add
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

base_model = DenseNet121(input_shape=(224,224,3), include_top=False, weights=None)
base_model.load_weights('../densenet121.h5')
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Flatten(),
    Dense(256, activation="relu", kernel_initializer='he_normal'),
    Dropout(0.3),
    Dense(2, activation='softmax')
])

# Set learning rate
learning_rate = 0.0001

# Define optimizer
adam_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)

# loss function
from tensorflow.keras import backend as K
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        pt = y_pred * y_true + (1 - y_pred) * (1 - y_true)
        pt = K.clip(pt, K.epsilon(), 1 - K.epsilon())
        return -K.sum(alpha * K.pow(1 - pt, gamma) * K.log(pt), axis=-1)
    return focal_loss_fixed

model.compile(optimizer=adam_optimizer, loss=focal_loss(gamma=2.0, alpha=0.25), metrics=['accuracy'])

# Set an early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Fit model
h_callback = model.fit(train_images, train_labels,
                       epochs = 100, validation_data=(val_images, val_labels),
                       verbose=1, batch_size=10, callbacks=[early_stopping])

# Save model
model.save('densenet_model.keras')

def plot_performance(train,validation,title):
    plt.figure()
    plt.plot(train)
    plt.plot(validation)
    plt.title(title)
    plt.ylabel(title)
    plt.xlabel('Épocas')
    plt.legend(['Entrenamiento', 'Validación'], loc='upper right')
    plt.show()
    plt.savefig('../results/' + title)

# Extract from the h_callback object loss and val_loss to plot the learning curve
plot_performance(h_callback.history['loss'], h_callback.history['val_loss'],'Pérdida')
plot_performance(h_callback.history['accuracy'], h_callback.history['val_accuracy'],'Precisión')

preds = model.predict(test_images)

print(f'Predictions: {preds}')
print(f'Test data: {test_labels}')

# RESULTS
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, multilabel_confusion_matrix

# Round the predictions to obtain binary values
rounded_preds = np.round(preds)

# Calculate accuracy
acc = accuracy_score(test_labels, rounded_preds) * 100

# Calculate precision, recall, and F1-score per label
precision = precision_score(test_labels, rounded_preds, average=None) * 100
recall = recall_score(test_labels, rounded_preds, average=None) * 100
f1 = f1_score(test_labels, rounded_preds, average=None)

# Calculate multilabel confusion matrix
cm = multilabel_confusion_matrix(test_labels, rounded_preds)

# Extract true negatives, false positives, false negatives, true positives from the confusion matrix
tn = cm[:, 0, 0]
fp = cm[:, 0, 1]
fn = cm[:, 1, 0]
tp = cm[:, 1, 1]

print('CONFUSION MATRIX ------------------')
print(cm)

print('\nTEST METRICS ----------------------')
print('Accuracy: {}%'.format(acc))
print('Precision: {}%'.format(precision))
print('Recall: {}%'.format(recall))
print('F1-score: {}'.format(f1))

print('\nTRAIN METRIC ----------------------')
print('Train acc: {}'.format(np.round((h_callback.history['accuracy'][-1]) * 100, 2)))


# Evaluate performance
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

# Calcular las probabilidades de predicción del modelo
pred_probs = model.predict(test_images)

# Binarizar las etiquetas si es necesario (dependiendo de la forma de tus etiquetas)
test_labels_bin = label_binarize(test_labels, classes=[0, 1])

# Calcular la curva ROC y el AUC para cada clase
n_classes = test_labels_bin.shape[1]
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(test_labels_bin[:, i], pred_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Configurar colores y estilos para las curvas ROC
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
lw = 2

# Graficar las curvas ROC para cada clase
plt.figure(figsize=(10, 6))


for i, color in zip(range(n_classes), colors):
    if i == 1:
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label=f'Curva ROC para caso PD-L1 positivo (AUC = {roc_auc[i]:.2f})')




# Configurar los detalles del gráfico
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Ratio Falsos Positivos')
plt.ylabel('Ratio Verdaderos Positivos')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
plt.savefig('../results/AUC.png')

# Excel predictions
real=[]
for i in test_labels:
    index = np.argmax(i)
    real.append(index)

predic=[]
for i in preds:
    index = np.argmax(i)
    predic.append(index)

result=pd.DataFrame()
result['real'] = real
result['pred'] = predic
result.to_excel('../Comparison.xlsx')

# Plot Bar diagram
def plot_bar(real_cases, pred_cases):
    real_0 = list(real_cases).count(0)
    real_1 = list(real_cases).count(1)
    pred_0 = list(pred_cases).count(0)
    pred_1 = list(pred_cases).count(1)
    labels = ['Negativo', 'Positivo']
    real_values = [real_0, real_1]
    pred_values = [pred_0, pred_1]

    plt.figure()

    # Real cases diagram
    ax1 = plt.subplot(121)
    plt.bar(labels, real_values)
    # Agregar etiquetas con el número de veces que aparece cada número
    for i, v in enumerate(real_values):
        plt.text(i, v, str(v), ha='center', va='bottom')

    plt.xlabel('PD-L1')
    plt.ylabel('Casos')
    plt.title('Conjunto de casos reales')


    # Predicted cases diagram
    ax2 = plt.subplot(122, sharey=ax1)
    plt.bar(labels, pred_values)
    # Agregar etiquetas con el número de veces que aparece cada número
    for i, v in enumerate(pred_values):
        plt.text(i, v, str(v), ha='center', va='bottom')

    plt.xlabel('PD-L1')
    plt.ylabel('Casos')
    plt.title('Conjunto de casos predecidos')

    plt.savefig('../Comparison.png')

plot_bar(real, predic)
