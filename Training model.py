
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout



# 第一步：数据收集
# 设置训练、验证和测试数据集的路径
train_dir = 'F:\\lunwen\\raw-img'
val_dir = 'F:\\lunwen\\raw-val'
test_dir = 'F:\\lunwen\\raw-test'

# 设置图片大小和批量?
img_size = (224, 224)
batch_size = 32

# 为训练、验证和测试数据集定义图像数据生成器
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

# # 使用 ImageDataGenerator 加载训练、验证和测试数据�?
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# 步骤二：数据预处理这个例子中使用了ImageDataGenerator类实现数据增强。体
# 具来说，训练数据集使用了旋转、水平、垂直随机平移和水平翻转等数据增强技术，而验证和测试数据集只进行了图像大小的缩放，以确保模型的泛化能力�?
# 定义基础模型
base_model = tf.keras.applications.MobileNetV2(
    input_shape=img_size + (3,),
    include_top=False,
    weights='imagenet'
)

# 冻结基础模型
base_model.trainable = False

# 定义新的分类
model = Sequential([
    base_model,
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
def precision(y_true, y_pred):
    """自定义精确度指标"""
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
def recall(y_true, y_pred):
    """自定义召回率指标"""
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

# 编译模型并加入评价指标
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', precision, recall])

# 定义周期数
epochs = 10

# 训练模型
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs
)
# 将模型保存到磁盘上
model.save('my_model.h5')