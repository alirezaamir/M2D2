import tensorflow as tf
import numpy as np
from utils.data23 import get_balanced_data, get_test_data, get_test_overlapped, get_non_seizure
from sklearn.metrics import f1_score, confusion_matrix
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
    w_new = np.round(np.array(var_inv * (1 << NUM_FRACTION))).astype(np.int)
    w_new = np.clip(w_new, a_min=-(1 << BIT_RANGE) + 1, a_max=(1 << BIT_RANGE) - 1)
    w_new = np.where(w_new == 0, w_new+1, w_new)
    w_new = w_new.astype(np.float)/(1 << NUM_FRACTION)
    new_var = 1.0/(w_new ** 2) - 0.001
    return new_var


def my_quantized_model(model, num_fraction_w, num_fraction_b, bits):
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
                    new_var = get_fxp_var(w, num_fraction_b, bit_range)
                    new_weights.append(new_var)
                    # new_weights.append(w)
                else:
                    num_fraction = num_fraction_b if (layer.name).startswith('batch') else num_fraction_w
                    w_new = np.round(np.array(w * (1 << num_fraction))).astype(np.int)
                    w_new = np.clip(w_new, a_min=-(1 << bit_range) + 1, a_max= (1 << bit_range) - 1)
                    new_weights.append(w_new.astype(np.float)/(1 << num_fraction))
            layer.set_weights(new_weights)

    # for layer in model.layers:
    #     print(layer.name)
    #     print(layer.get_weights())

    return model


def scaled_model(model:tf.keras.models.Model):
    scaled = model
    first_conv_name = [layer.name for layer in model.layers if layer.name.startswith('conv')][0]
    first_batch_name = [layer.name for layer in model.layers if layer.name.startswith('batch')][0]
    # print("First Conv : {}\nFirst Batch : {}".format(first_conv_name, first_batch_name))
    w =  model.get_layer(first_conv_name).get_weights()[0]
    bias = model.get_layer(first_conv_name).get_weights()[1]
    new_weight = [w, bias/250]
    scaled.get_layer(first_conv_name).set_weights(new_weight)

    gamma = model.get_layer(first_batch_name).get_weights()[0]
    beta = model.get_layer(first_batch_name).get_weights()[1]
    mean = model.get_layer(first_batch_name).get_weights()[2]
    var = model.get_layer(first_batch_name).get_weights()[3]
    new_weight = [gamma, beta, mean/250, var/(250*250)]
    scaled.get_layer(first_batch_name).set_weights(new_weight)
    return scaled


def save_model(model, subdirname, test_patient):
    tflite_models_dir = pathlib.Path("{}/tflite_models/".format(subdirname))
    tflite_models_dir.mkdir(exist_ok=True, parents=True)

    # Save the quantized model:
    tflite_model_quant_file = tflite_models_dir / "test_{}.tflite".format(test_patient)
    tflite_model_quant_file.write_bytes(model)


def save_c_data(test_pat):
    dirname = '../../output/C_data'
    arch = '23channel'
    subdirname = "../../temp/vae_mmd/integrated/1024/{}/FCN_v1".format(arch)

    save_path = '{}/model/q_8_test_{}/saved_model/'.format(subdirname, test_pat)
    trained_model = tf.keras.models.load_model(save_path)
    model = my_quantized_model(trained_model, 8, 5, 8)

    NUM_FRACTION = 8
    BN_NUM_FRACTION = 5
    BITS = 8
    BIT_RANGE = BITS - 1

    def write_conv(w0, layer):
        text_file.write("\nchar conv1d_{}_w[] = ".format(layer))
        text_file.write("{")
        w_fxp = w0 * (1 << NUM_FRACTION)
        w_fxd_clipped = np.clip(w_fxp, a_min=-(1 << BIT_RANGE) + 1, a_max=(1 << BIT_RANGE) - 1)
        print("Conv1d {} shape: {}".format(layer, w0.shape))
        for wn in range(w0.shape[2]):
            for wj in range(w0.shape[1]):
                for wi in range(w0.shape[0]):
                    w_int = int(np.round(w_fxd_clipped[wi, wj, wn]))
                    if w_int < 0: w_int += (1 << BITS)
                    text_file.write("{},".format(hex(w_int)))
        text_file.write("};\n")

    def write_bias(w0, layer, mode):
        text_file.write("\nchar {}_{}_b[] = ".format(mode, layer))
        text_file.write("{")
        w_fxp = w0 * (1 << NUM_FRACTION)
        w_fxd_clipped = np.clip(w_fxp, a_min=-(1 << BIT_RANGE) + 1, a_max=(1 << BIT_RANGE) - 1)
        print("{}, {} shape: {}".format(mode, layer, w0.shape))
        for wi in range(w0.shape[0]):
            w_int = int(np.round(w_fxd_clipped[wi]))
            if w_int < 0: w_int += (1 << BITS)
            text_file.write("{},".format(hex(w_int)))
        text_file.write("};\n")

    def write_bn(w0, layer, param):
        mode = 'batch_normalization'
        text_file.write("\nchar bn_{}_{}[] = ".format(layer, param))
        text_file.write("{")
        w_fxp = w0 * (1 << BN_NUM_FRACTION)
        print("{}: {}, {}".format(param, np.min(w_fxp), np.max(w_fxp)))
        w_fxd_clipped = np.clip(w_fxp, a_min=-(1 << BIT_RANGE) + 1, a_max=(1 << BIT_RANGE) - 1)
        print("{}, {} shape: {}".format(mode, layer, w0.shape))
        for wi in range(w0.shape[0]):
            w_int = int(np.round(w_fxd_clipped[wi]))
            if w_int < 0: w_int += (1 << BITS)
            text_file.write("{},".format(hex(w_int)))
        text_file.write("};\n")

    def write_fc(w0, layer):
        depths = [128, 100]
        text_file.write("\nchar dense_{}_w[] = ".format(layer))
        text_file.write("{")
        print("shape : {}".format(w0.shape))
        w_fxp = w0 * (1 << NUM_FRACTION)
        w_fxd_clipped = np.clip(w_fxp, a_min=-(1 << BIT_RANGE) + 1, a_max=(1 << BIT_RANGE) - 1)
        for wn in range(w0.shape[1]):
            for depth in range(depths[layer]):
                for wi in range(w0.shape[0] // depths[layer]):
                    w_int = int(np.round(w_fxd_clipped[wi * depths[layer] + depth, wn]))
                    if w_int < 0: w_int += (1 << BITS)
                    text_file.write("{},".format(hex(w_int)))
        text_file.write("};\n")

    with open("{}/pat{}_model.c".format(dirname, test_pat), "w") as text_file:
        ## Weights
        conv = 0
        bn = 0
        fc = 0
        for layer in model.layers:
            print("Name: {}".format(layer.name))
            if layer.name.startswith("conv"):
                write_conv(layer.get_weights()[0], conv)
                write_bias(layer.get_weights()[1], conv, 'conv1d')
                conv += 1
            if layer.name.startswith("batch"):
                write_bn(layer.get_weights()[0], bn, 'gamma')
                write_bn(layer.get_weights()[1], bn, 'betta')
                write_bn(layer.get_weights()[2], bn, 'mean')
                var_inv = 1.0 / np.sqrt(np.add(layer.get_weights()[3], 0.001))
                write_bn(var_inv, bn, 'var')
                bn+=1
            if layer.name.startswith("dense"):
                write_fc(layer.get_weights()[0], fc)
                write_bias(layer.get_weights()[1], fc, 'dense')
                fc+=1


        ## Fully Connected weights
        # depths = [128, 100]
        # for layer in range(2):
        #


        # text_file.write("\nint* A_w[] = {A_w0, A_w1, A_w2, A_w3, A_w4, A_w5, A_w6};\n")
        text_file.write("\nchar* conv1d_w[] = {conv1d_0_w, conv1d_1_w, conv1d_2_w};\n")
        text_file.write("\nchar* conv1d_b[] = {conv1d_0_b, conv1d_1_b, conv1d_2_b};\n")
        text_file.write("\nchar* dense_w[] =  {dense_0_w, dense_1_w};\n")
        text_file.write("\nchar* dense_b[] =  {dense_0_b, dense_1_b};\n")
        text_file.write("\nchar* bn[] =  {bn_0_gamma, bn_0_betta, bn_0_mean, bn_0_var,"
                        "bn_1_gamma, bn_1_betta, bn_1_mean, bn_1_var,"
                        "bn_2_gamma, bn_2_betta, bn_2_mean, bn_2_var};\n")

def save_input_data(test_patient):
    IN_NUM_FRACTION = 10
    IN_BIT_RANGE = 15
    IN_BITS= 16
    X_test, y_test = get_non_seizure(test_patient, root='../..')
    # if X_test.shape[0] > 30000:
    #     print("Pat {}: Too many samples".format(test_patient))
    #     return
    X_test = np.clip(X_test, a_min=-250, a_max=250)
    X_test = X_test/250
    # np.random.seed(137)
    # permutation_list = np.random.permutation(range(X_test.shape[0]))
    # X_test = X_test[permutation_list]
    # y_test = y_test[permutation_list]
    dirname = '../../output/C_data/non_ictal'
    print(X_test.shape)
    # np.save("{}/input".format(dirname), X_test[:2000])
    for iter in range(X_test.shape[0]//10000 +1):
        with open("{}/pat{}{}.txt".format(dirname, test_patient, iter), "w") as text_file:
            input_fxp = X_test[10000*iter:10000*(iter+1)] * (1 << IN_NUM_FRACTION)
            # input_fxp = X_test * (1 << IN_NUM_FRACTION)
            input_fxp_clipped = np.clip(input_fxp, a_min=-(1 << IN_BIT_RANGE) + 1, a_max=(1 << IN_BIT_RANGE) - 1)
            for sample in range(input_fxp_clipped.shape[0]):
                for wj in range(input_fxp_clipped.shape[2]):
                    for wi in range(input_fxp_clipped.shape[1]):
                        sample_int = int(np.round(input_fxp_clipped[sample, wi, wj]))
                        if sample_int < 0: sample_int += (1 << IN_BITS)
                        # text_file.write("{},".format(hex(sample_int)))
                        text_file.write("{}\n".format(hex(sample_int)))


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
    X_test, y_test = get_non_seizure(test_patient, root='../..')
    X_test = np.clip(X_test, a_min=-250, a_max=250)
    X_test = X_test/250
    # np.random.seed(137)
    # permutation_list = np.random.permutation(range(X_test.shape[0]))
    # X_test = X_test[permutation_list]
    # y_test = y_test[permutation_list]
    predicted = model.predict(X_test)
    y_pred = np.argmax(predicted, axis=1)
    # print(np.where(y_test == 1)[0].shape)
    with open('../../output/C_data/non_ictal/out{}.txt'.format(test_patient)) as f:
        out_lines = f.readlines()
        c_out = [int(a[0]) for a in out_lines]
    print("Pat {}, Floating Point: {:.2f}, C out: {:.2f}".format(test_patient,
                                                         f1_score(y_test, y_pred) * 100,
                                                         f1_score(y_test, c_out) * 100))
    # print(f1_score(y_test, y_pred))
    return confusion_matrix(y_test, y_pred), confusion_matrix(y_test, c_out)
    # return f1_score(y_test, y_pred)
#

def get_results():
    arch = '23channel'
    subdirname = "../../temp/vae_mmd/integrated/1024/{}/FCN_v1".format(arch)
    f1_orig = []
    f1_qnt = []
    f1_c = []
    mat = []
    for pat_id in range(22, 23):
        # if pat_id == 4:
        #     continue
        print("PATIENT : {}".format(pat_id))
        save_path = '{}/model/test_{}/saved_model/'.format(subdirname, pat_id)
        trained_model = tf.keras.models.load_model(save_path)
        scaled_model(trained_model)
        orig, c_out = inference(trained_model, test_patient=pat_id)
        f1_orig.append(orig)
        f1_c.append(c_out)

        # save_path = '{}/model/q_8_test_{}/saved_model/'.format(subdirname, pat_id)
        # trained_model = tf.keras.models.load_model(save_path)
        # q_model = my_quantized_model(trained_model, 8, 5, 8)
        # f1_qnt.append(inference(q_model, test_patient=pat_id)[0])
        # mat += inference(q_model, test_patient=pat_id)
        # save_c_data(q_model)
        # save_input()
        # _data(test_patient=pat_id)
    print("MATrix : {}".format(mat))

    print("Original = {}".format(f1_orig))
    # print("Original : {}".format(np.mean(f1_orig)))
    print("Quantized8_retrained = {}".format(f1_qnt))
    print("C_out = {}".format(f1_c))
    # print("Quantized : {}".format(np.mean(f1_qnt)))


def get_data_number():
    total_shape = 0
    for test_patient in range(1, 24):
        X_test, y_test = get_test_data(test_patient, root='../')
        print("Pat: {}, Shape: {}".format(test_patient, X_test.shape[0]))
        total_shape += X_test.shape[0]
    print("Total : {}".format(total_shape))


if __name__ == '__main__':
    # tf.config.experimental.set_visible_devices([], 'GPU')
    get_results()
    # get_daAta_number()
    # for pat in range(1, 4):
    #     save_input_data(pat)
    #     save_c_data(pat)

