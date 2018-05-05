import tensorflow as tf
import time
import argh
from utils import load_images


def conv_net(input_data, input_shape, n_classes, dropout_rate, reuse, is_training):
    """ Задает форму (или как ее еще называют топологию) нейронной сети
        Входные параметры:
          input_data:   - слой входных данных. tf.placeholders. Все помнят, зачем он нужен?
          input_shape:  - истинная размерность входных данных (ширина x высота)
          n_classes:    - число предсказываемых классов
          dropout_prob: - вероятность, с которой каждая из связей последнего слоя будет выкинута при дропауте
          reuse:        - надо ли переиспользовать уже созданные переменные 
                          (передается в вызов variable_scope)
          is_training:  - будет ли данная сеть использоваться для обучения 
                          (передается в слой дропаут для его активации)
        На выход функция возвращает последний слой нейронной сети, который можно использовать
        для обучения и предсказания результатов
    """
    # Объявление набора переменных нужно для переиспользования их значений в двух сетях
    with tf.variable_scope('ConvNet', reuse=reuse):
        # Первый светочный слой. Делает свертки с окошком 5 на 5 (параметр kernel_size)
        # Глубина картинки на входе - 3 (количества цветов). На выходе - 32 элемента (параметр filters)
        conv1 = tf.layers.conv2d(input_data, 32, 5, activation=tf.nn.relu)
        # Max Pooling (семплирование).
        # Выбирает максимальные значения из участков 2x2 (pool_size), с шагом 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
        # Второй светрочный слой. Свертки с окошком 3x3 и выходной толщиной слоя 64
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        # Еще одно семплирование результатов
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
        # Превращение кубического терзора (с шириной, высотой и толщиной) в плоский
        fc1 = tf.contrib.layers.flatten(conv2)
        # Полносвязный слой (из классической нейросети)
        fc1 = tf.layers.dense(fc1, 1024)
        # Слой дропаута (применяется только на этапе обучения)
        fc1 = tf.layers.dropout(fc1, rate=dropout_rate, training=is_training)
        # Выходной слой
        out = tf.layers.dense(fc1, n_classes)
    return out


def create_model_function(learning_rate, dropout_rate, input_shape, num_classes = 2):
    def model_fn(features, labels, mode):
        """
        Функция задает Tensorflow модель (нейросеть + способ ее обучения + способ ее оценивания), 
        которую затем можно использовать в шаблоне TF Estimator.
        Так как все в тензорфлоу написано рептилоидами, выглядит эта функция очень странно.
        Входные параметры:
           features: словарь с фичами, вида "имя входной фичи" - "ее плейсхолдер"
           labels:   плейсхолдер для выходных значений
           mode:     перечисление ситуаций, когда функция может быть вызвана.
                     (ModeKeys.TRAIN, ModeKeys.EVAL, ModeKeys.PREDICT)
        На выходе она возвращает описание в виде объекта EstimatorSpec
        (даже не спрашивайте меня, почему он так называется)
        """

        # Задаем сети для обучения и применения. Благодаря флажку reuse=True
        # обе сети имеют одни и те же внутренние веса
        logits_train = conv_net(features["images"], input_shape, num_classes,
                                dropout_rate, reuse=False, is_training=True)
        logits_predict = conv_net(features["images"], input_shape, num_classes,
                                  dropout_rate, reuse=True, is_training=False)
        # Предсказанный класс объекта. Берется как номер выхода на последнем слое с максимальным значением
        predicted_classes = tf.argmax(logits_predict, axis=1)
        # В режиме предсказания задавать процесс обучения нам не надо - возвращаем результаты
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=predicted_classes)
        # Функция потерь - кроссэнтропия между реальными и предсказанными метками классов
        loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_train, labels=labels))
        # В качесте оптимизатора используем AdamOptimizer - градиентный спуск с небольшими эвристиками
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # Процесс обучения - это минимизация ошибки при помощи выбранного оптимизатора
        train_function = optimizer.minimize(loss_function, global_step=tf.train.get_global_step())
        # Метрикой качества является точность - какой процент раз мы угадали животное на картинке правильно
        accuracy = tf.metrics.accuracy(labels=tf.argmax(labels, 1), predictions=predicted_classes)
        # Для обучения и оценки точности мы задаем расширенную спецификацию эстиматора
        # с функцией потерь, функционалом обучения и метриками качества
        # the different ops for training, evaluating, ...
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predicted_classes,
            loss=loss_function,
            train_op=train_function,
            eval_metric_ops={'accuracy': accuracy})

    return model_fn


@argh.arg('path', type=str, help='path to directory where pictures of cats and dogs are stored')
@argh.arg('--shape-x', type=int, help='horisontal dimention of input image')
@argh.arg('--shape-y', type=int, help='vertical dimention of input image')
@argh.arg('--num-steps', type=int, help='number of epochs of model training')
@argh.arg('--batch-size', type=int, help='number of examples into each training batch')
@argh.arg('--learning-rate', type=float, help='gradient descend speed')
@argh.arg('--dropout-rate', type=float, help='probability that particular connection will be dropped off')
@argh.arg('--testing-percentage', type=float, help='fraction of input data to be used for testing')
def train_and_evaluate(path,
                       shape_x=128,
                       shape_y=128,
                       num_steps=2000,
                       batch_size=200,
                       learning_rate=0.001,
                       dropout_rate=0.25,
                       testing_percentage=30):
    """
    Основная точка входа в скрипт.
    Декораторы @argh.arg нужны, чтобы иметь возможность задавать значения из коммандной строки.
    Аргументы:
      path: путь в файловой системе, где лежат скачанные с каггла картинки. У меня это, например "~/Downloads/train/".
             В этой папке лежат файлы, названные "cat123.jpg, dog.124.jpg"
      shape_x: размер по горизонтали, к которому надо приводить все картинки
      shape_y: размер по вертикали, к которому надо приводить все картинки
      testing_percentage: процент данных, используемых для обучения
      num_steps: число шагов обучения
      batch_size: размер минибатча, на котором будет делаться каждый шаг обучения
      learning_rate: скорость обучения градиентного спуска
      dropout_rate: частота выкидывания связей между слоями в дропауте
    """
    tf.logging.set_verbosity(tf.logging.INFO)
    start = time.perf_counter()
    # Загрузка изображений
    tf.logging.info("Loading images")
    image_list = load_images(path,
                             testing_percentage=testing_percentage,
                             shape=(shape_x, shape_y))
    num_classes = image_list["train"]["one_hot_labels"].shape[0]
    tf.logging.info("Image loading is done in {:.03f}s".format(time.perf_counter() - start))

    # Обучение модели
    tf.logging.info("Training model")
    start = time.perf_counter()
    model = tf.estimator.Estimator(create_model_function(learning_rate, dropout_rate, (shape_x, shape_y), num_classes))
    # Задаем функцию, которая будет скармливать пачки данных нашей модели
    train_input = tf.estimator.inputs.numpy_input_fn(
                            x={'images': image_list["train"]["images"]},
                            y=image_list["train"]["one_hot_labels"],
                            batch_size=batch_size,
                            num_epochs=None,
                            shuffle=True)

    model.train(train_input, steps=num_steps)
    tf.logging.info("Model training done in {:.03f}s".format(time.perf_counter() - start))

    # Оценка качества модели на тестовой выборке
    testing_input = tf.estimator.inputs.numpy_input_fn(
                            x={'images': image_list["test"]["images"]},
                            y=image_list["test"]["one_hot_labels"],
                            batch_size=batch_size,
                            shuffle=False)
    testing_metrics = model.evaluate(testing_input)

    tf.logging.info("Testing Accuracy: {}".format(testing_metrics["accuracy"]))


if __name__ == "__main__":
    argh.dispatch_command(train_and_evaluate)
