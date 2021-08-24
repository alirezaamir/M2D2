import tensorflow as tf
import numpy as np
from utils.data23 import get_balanced_data, get_test_data
from sklearn.metrics import f1_score
import pathlib
import matplotlib.pyplot as plt


def tf_quantized_model(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model_quant = converter.convert()
    print(tflite_model_quant)
    return tflite_model_quant


def get_fxp_var(std, NUM_FRACTION, BIT_RANGE):
    var_inv = 1 / np.sqrt(np.add(std, 0.001))
    w_new = np.array(var_inv * (1 << NUM_FRACTION)).astype(np.int)
    w_new = np.clip(w_new, a_min=-(1 << BIT_RANGE) + 1, a_max=(1 << BIT_RANGE) - 1)
    w_new = np.where(w_new == 0, w_new+1, w_new)
    w_new = w_new/(1 << NUM_FRACTION)
    new_var = 1/(w_new ** 2) - 0.001
    return new_var


def my_quantized_model(model, num_fraction, bits):
    bit_range = bits - 1
    for layer in model.layers:
        # print(layer.name)
        if len(layer.get_weights()) > 0:
            new_weights = []
            for idx, w in enumerate(layer.get_weights()):
                # plt.figure()
                # plt.hist(np.reshape(w, (-1,1)), bins=20)
                # plt.title("{} {}".format(layer.name, idx))
                # plt.show()
                if idx == 3:
                    new_var = get_fxp_var(w, num_fraction, bit_range)
                    new_weights.append(new_var)
                    # new_weights.append(w)
                else:
                    w_new = np.array(w * (1 << num_fraction)).astype(np.int)
                    w_new = np.clip(w_new, a_min=-(1 << bit_range) + 1, a_max= (1 << bit_range) - 1)
                    new_weights.append(w_new/(1 << num_fraction))
            layer.set_weights(new_weights)

    # for layer in model.layers:
    #     print(layer.name)
    #     print(layer.get_weights())

    return model


def save_model(model, subdirname, test_patient):
    tflite_models_dir = pathlib.Path("{}/tflite_models/".format(subdirname))
    tflite_models_dir.mkdir(exist_ok=True, parents=True)

    # Save the quantized model:
    tflite_model_quant_file = tflite_models_dir / "test_{}.tflite".format(test_patient)
    tflite_model_quant_file.write_bytes(model)


def tf_inference(subdirname, test_patient:int):
    X_test, y_test = get_test_data(test_patient, root='../')
    X_test = np.clip(X_test, a_min=-250, a_max=250)

    # Initialize the interpreter
    tflite_file = "{}/tflite_models/test_{}.tflite".format(subdirname, test_patient)
    interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    predictions = np.zeros((len(X_test),), dtype=int)
    for i, test_image_index in enumerate(range(X_test.shape[0])):
        test_image = X_test[test_image_index]
        test_label = y_test[test_image_index]

        test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
        interpreter.set_tensor(input_details["index"], test_image)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])[0]

        predictions[i] = output.argmax()

    print(predictions)


def inference(model, test_patient:int):
    X_test, y_test = get_test_data(test_patient, root='../')
    X_test = np.clip(X_test, a_min=-250, a_max=250)
    predicted = model.predict(X_test)
    y_pred = np.argmax(predicted, axis=1)
    return f1_score(y_test, y_pred)


def get_results():
    arch = '23channel'
    subdirname = "../../temp/vae_mmd/integrated/1024/{}/FCN_v1".format(arch)
    f1_orig = []
    f1_qnt = []
    for pat_id in range(12, 24):
        print("PATIENT : {}".format(pat_id))
        # save_path = '{}/model/test_{}/saved_model/'.format(subdirname, pat_id)
        # trained_model = tf.keras.models.load_model(save_path)
        # f1_orig.append(inference(trained_model, test_patient=pat_id))

        save_path = '{}/model/q_8_test_{}/saved_model/'.format(subdirname, pat_id)
        trained_model = tf.keras.models.load_model(save_path)
        q_model = my_quantized_model(trained_model, 5, 8)
        f1_qnt.append(inference(q_model, test_patient=pat_id))

    # print("Original : {}".format(f1_orig))
    # print("Original : {}".format(np.mean(f1_orig)))
    print("Quantized8_retrained = {}".format(f1_qnt))
    print("Quantized : {}".format(np.mean(f1_qnt)))


if __name__ == '__main__':
    get_results()
