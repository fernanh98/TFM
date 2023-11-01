import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import imageio
import imgaug.augmenters as iaa

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)


# Load dataset
dataset_path ='../database.xlsx'
# Drop NA rows
dataset = pd.read_excel(dataset_path).dropna(subset=['PDL1-group'])

# Data Cleaning
def modify_path(path):
    'Adjust a path to a certain format'

    parts = path.split('/')
    parts = parts[2:]
    modified_parts = [part for part in parts]
    modified_path = '/'.join(modified_parts)
    return modified_path

# Apply modify_path()
dataset['Path'] = dataset['Path'].apply(lambda x: modify_path(x))
folder_path = "../cimes_images/"
dataset['Path'] = [folder_path + path for path in dataset['Path']]



# OneHotEncoder
dataset['<1'] = [1 if row == '<1' else 0 for row in dataset['PDL1-group']]
dataset['1-49'] = [1 if row == '1-49' else 0 for row in dataset['PDL1-group']]
dataset['>49'] = [1 if row == '>49' else 0 for row in dataset['PDL1-group']]

# Check proportion of Pdl1 Status on dataset
prop = dataset['PDL1-group'].value_counts(normalize=True)
print('Percentage of PDL1 status', prop)
print('---------------------------------','\n','Number of images: ',len(dataset))
print('\n','Positive >49: ',len(dataset) * prop[0])
print('\n','Positive 1-49: ',len(dataset) * prop[1])
print('\n','Negative PDL1: ',len(dataset) * prop[2])

# Neg - Pos Proportion
prop = dataset['PDL1-cat'].value_counts(normalize=True)
print('Percentage of PDL1 status', prop)


# Modify path extension
def modify_path(path):
    'Adjust a path to a certain format'
    parts = path.split('.')
    parts.pop()
    modified_path = '.'.join(parts)
    return modified_path + '.jpg'

# Apply modify_path()
dataset['Path'] = dataset['Path'].apply(lambda x: modify_path(x))
dataset.reset_index(inplace=True, drop=True)

# Drop non-existing images on dataset
def check_dataframe_images(df, path_column):

    count = 0
    indexes = []
    for index,path in df[path_column].items():
        try:
            img = imageio.v2.imread(path)
        except:
            count+=1
            indexes.append(index)
            #print(f'Dropped row {index}: {df.iloc[index][path_column]}')
            #print(count)
    df.drop(labels=indexes, axis=0, inplace=True)
    return df

dataset = check_dataframe_images(dataset, 'Path')

# Shuffle dataset
dataset = dataset.sample(frac=1, random_state=42)

# Check proportion of Pdl1 Status on dataset
prop = dataset['PDL1-group'].value_counts(normalize=True)
print('Percentage of PDL1 status', prop)
print('---------------------------------','\n','Number of images: ',len(dataset))
print('\n','Positive >49: ',len(dataset) * prop[0])
print('\n','Positive 1-49: ',len(dataset) * prop[1])
print('\n','Negative PDL1: ',len(dataset) * prop[2])

# Neg - Pos Proportion
prop = dataset['PDL1-cat'].value_counts(normalize=True)
print('Percentage of PDL1 status', prop)


from sklearn.model_selection import StratifiedShuffleSplit

# Split data into train and test set
split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
for train_index, test_index in split.split(dataset, dataset['PDL1-group']):
    train_set1 = dataset.iloc[train_index]
    test_set = dataset.iloc[test_index]

# Split train data into train and validation
split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
for train_index, val_index in split.split(train_set1, train_set1['PDL1-group']):
    train_set = train_set1.iloc[train_index]
    val_set = train_set1.iloc[val_index]

print('Train data', train_set['PDL1-group'].value_counts(normalize=True))
print('---------------------------------','\n','Number of train images: ',len(train_set))
print('---------------------------------')
print('Test data', test_set['PDL1-group'].value_counts(normalize=True))
print('---------------------------------','\n','Number of test images: ',len(test_set))
print('---------------------------------')
print('Validation data', val_set['PDL1-group'].value_counts(normalize=True))
print('---------------------------------','\n','Number of validation images: ',len(val_set))

# Image Preprocess
def preprocess_image(path, target_size, gamma, saturation, std, sigma, plot, train, angle,
                    flipud_num, fliplr_num):

    # Load image as a Numpy Matrix
    image = imageio.v2.imread(path)

    # Crop image
    image_crop = iaa.CenterCropToSquare()(images=[image])[0]

    # Resize image
    image_resize = iaa.Resize({"height": target_size,
                               "width": target_size})(images=[image])[0]
    if train:
        image_rotated = iaa.Rot90(angle)(images=[image_resize])[0]

        # Randomly flip
        image_flip = iaa.Flipud(flipud_num)(images=[image_rotated])[0]
        image_flip = iaa.Fliplr(fliplr_num)(images=[image_flip])[0]

        # Gamma
        image_gamma = iaa.GammaContrast((0.0, gamma))(images=[image_flip])[0]

        # Saturation and Temperature
        image_sat = iaa.MultiplySaturation((0, saturation))(images=[image_gamma])[0]
        image_temp = iaa.ChangeColorTemperature((2000, 8000))(images=[image_sat])[0]

        # Gaussian Noise
        noisy_image = iaa.AdditiveGaussianNoise(scale=(0, std))(images=[image_sat])[0]

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
def get_data(df, target_size = 224, gamma = 0.0, saturation = 0, std = 15,
                    sigma = 0.0, plot = False, train = False, angle=0, flipud_num=0,
             fliplr_num=0):

    images_data = []

    for image_path in df['Path']:

        # Preprocess image
        image = preprocess_image(image_path, gamma=gamma, saturation=saturation, plot=plot,
                                train=train, std=std, sigma=sigma, target_size=target_size,
                                 angle=angle, flipud_num=flipud_num, fliplr_num=fliplr_num) / 255
        # Save image
        images_data.append(image)

    images = np.array(images_data)
    labels = np.array(df[['<1','1-49','>49']]).astype(float)

    return images, labels


test_images, test_labels = get_data(test_set)
val_images, val_labels = get_data(val_set)
train_images, train_labels = get_data(train_set)


train_images_0, train_labels_0 = get_data(train_set, train = True, angle=0, std=20)
# Shuffle 1
train_set_1 = train_set.sample(frac=1, random_state=42)
train_images_1, train_labels_1 = get_data(train_set_1, train = True, angle=1, std=10)
# Shuffle 2
train_set_2 = train_set.sample(frac=1, random_state=42)
train_images_2, train_labels_2 = get_data(train_set_2, train = True, angle=2, std=25)
# Shuffle 3
train_set_3 = train_set.sample(frac=1, random_state=42)
train_images_3, train_labels_3 = get_data(train_set_3, train = True, angle=3, std=15)

train_images = np.concatenate((train_images_0, train_images_1, train_images_2, train_images_3))

train_labels = np.concatenate((train_labels_0, train_labels_1, train_labels_2, train_labels_3))

print('Preprocesamiento completado',train_images.shape)

# CNN
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2, DenseNet121
from tensorflow.keras.models import Sequential, Model, load_model
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
    Dense(3, activation='softmax')
])

# Set learning rate
learning_rate = 0.01

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


model.compile(optimizer=adam_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model_save = ModelCheckpoint('best_model.hdf5', save_best_only=True)

h_callback = model.fit(train_images, train_labels,
                       epochs = 50, validation_data=(val_images, val_labels),
                       verbose=1, batch_size=10, callbacks=[early_stopping])

def plot_performance(train,validation,title):
  plt.figure()
  plt.plot(train)
  plt.plot(validation)
  plt.title(title)
  plt.ylabel(title)
  plt.xlabel('Épocas')
  plt.legend(['Entrenamiento', 'Validación'], loc='upper right')
  plt.savefig('../results_cimes/' + title)

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


# ### AUC
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

# Calcular las probabilidades de predicción del modelo
pred_probs = np.round(model.predict(test_images))

# Binarizar las etiquetas si es necesario (dependiendo de la forma de tus etiquetas)
classes=['<1', '1-49', '>49']
#test_labels_bin = label_binarize(test_labels, classes=classes)

# Calcular la curva ROC y el AUC para cada clase
n_classes = test_labels.shape[1]
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(test_labels[:, i], pred_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Configurar colores y estilos para las curvas ROC
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
lw = 2

# Graficar las curvas ROC para cada clase
plt.figure(figsize=(10, 6))

for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label=f'Curva ROC (PD-L1 {classes[i]}%) (AUC = {roc_auc[i]:.2f})')

# Configurar los detalles del gráfico
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Ratio Falsos Positivos')
plt.ylabel('Ratio Verdaderos Positivos')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
plt.savefig('../AUC.png')


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
    real_2 = list(real_cases).count(2)
    pred_0 = list(pred_cases).count(0)
    pred_1 = list(pred_cases).count(1)
    pred_2 = list(pred_cases).count(2)
    labels = ['<1','1-49','>49']
    real_values = [real_0, real_1, real_2]
    pred_values = [pred_0, pred_1, pred_2]

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
