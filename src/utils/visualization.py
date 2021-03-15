import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import utils.vae_model as vae_model
import pickle

SF = 256
SEG_LENGTH = 1024
arch = 'epilepsiae'


def visualize_inout(test_patient):
    # all_filename = get_epilepsy_files()
    # random_file = np.random.choice(all_filename[test_patient])
    # print("Random file: {}".format(random_file))
    random_file = get_epilepsiae_files(test_patient)

    with open(random_file, "rb") as pickle_file:
        # name = random_file.split('/')[-1][:8]
        data = pickle.load(pickle_file)
        X_total = np.array(data['X'])
        y_total = np.array(data['y'])

        model = get_model(test_patient)

        epilepsy_samples = np.where(y_total != 0 )[0]
        epilepsy_rand_sample = int(np.median(epilepsy_samples))
        middle_sample = np.expand_dims(X_total[epilepsy_rand_sample], 0)
        epilepsy_predict = model.predict(middle_sample)

        sample = np.random.randint(0, X_total.shape[0])
        sample_in = np.expand_dims(X_total[sample], 0)
        predict = model.predict(sample_in)

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.3)
        plt.subplot(221)
        plt.plot(X_total[sample][:, 0])
        plt.title("Raw sample {}".format(sample))
        plt.subplot(223)
        plt.title("Recons. sample {}".format(sample))
        plt.plot(predict[0, :, 0])
        plt.subplot(222)
        plt.plot(X_total[epilepsy_rand_sample][:, 0])
        plt.title("Raw Ep. sample {}".format(epilepsy_rand_sample))
        plt.subplot(224)
        plt.title("Recons. Ep. sample {}".format(epilepsy_rand_sample))
        plt.plot(epilepsy_predict[0, :, 0])


        plt.show()


def get_model(test_patient):
    beta = 1e-05
    latent_dim = 16
    lr = 1e-05
    decay = 0.5
    gamma = 0.0

    root = "../../output/vae/{}/".format(arch)
    stub = "seg_n_{}/beta_{}/latent_dim_{}/lr_{}/decay_{}/gamma_{}/test_{}/saved_model/"
    build_model = vae_model.build_model
    build_model_args = {
        "input_shape": (SEG_LENGTH, 2,),
        "enc_dimension": latent_dim,
        "beta": beta,
        "gamma": 0,
        "optim": tf.keras.optimizers.Adam(lr),
        "FS": SF
    }

    model, _ = build_model(**build_model_args)
    dirname = root + stub.format(SEG_LENGTH, beta, latent_dim, lr, decay, gamma, test_patient)
    if not os.path.exists(dirname):
        print("Model does not exist in {}".format(dirname))
        return None
    model.load_weights(dirname)

    return model


def generate_z_space(all_filename, test_patient):
    # all_filename = get_epilepsy_files()

    model = get_model(test_patient)
    if model is None:
        return

    dirname = "../../temp/z_norm/{}".format(arch)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    subdirname = "{}/{}/".format(dirname, test_patient)
    if not os.path.exists(subdirname):
        os.makedirs(subdirname)

    intermediate_model = tf.keras.models.Model(inputs=model.inputs, outputs=model.layers[1].output)

    z_train = {}
    for pat in range(1, 25):
        if pat == test_patient:
            continue

        for filename in all_filename[pat]:
            with open(filename, "rb") as pickle_file:
                name = filename.split('/')[-1][:8]
                data = pickle.load(pickle_file)
                X_total = np.array(data['X'])
                y_total = np.array(data['y'])
                # print("X shape: {}".format(X_total.shape))
                latent = intermediate_model.predict(X_total)[2]
                z_train[name] = {'Z': latent, 'y': y_total}

    z_filename = subdirname + "train.pickle"
    with open(z_filename, 'wb') as pickle_file:
        pickle.dump(z_train, pickle_file)

    z_test = {}
    for filename in all_filename[test_patient]:
        with open(filename, "rb") as pickle_file:
            name = filename.split('/')[-1][:8]
            data = pickle.load(pickle_file)
            X_total = np.array(data['X'])
            y_total = np.array(data['y'])
            # print("X shape: {}".format(X_total.shape))
            latent = intermediate_model.predict(X_total)[2]
            z_test[name] = {'Z': latent, 'y': y_total}

    z_filename = subdirname + "test.pickle"
    with open(z_filename, 'wb') as pickle_file:
        pickle.dump(z_test, pickle_file)


def get_epilepsy_files():
    all_filenames = {}

    for test_patient in range(1, 25):
        all_filenames[test_patient] = []
        for mode in ['train', 'valid']:
            dirname = "../../temp/vae_mmd_data/{}/full_normal/{}".format(SEG_LENGTH, mode)
            filenames = ["{}/{}".format(dirname, x) for x in os.listdir(dirname) if
                         x.startswith("chb{:02d}".format(test_patient))]
            for filename in filenames:
                with open(filename, "rb") as pickle_file:
                    data = pickle.load(pickle_file)
                    y = np.array(data["y"])
                    print("Number of Epilepsy in {}: {}".format(filename, np.sum(y)))
                    if np.sum(y) != 0:
                        all_filenames[test_patient].append(filename)
    return all_filenames


def get_epilepsiae_files(test_patient):
    dirname = "../../temp/vae_mmd_data/{}/epilepsia_normal/{}".format(SEG_LENGTH, 'train')
    filename = "{}/{}.pickle".format(dirname, test_patient)
    return filename


if __name__ == '__main__':
    tf.config.experimental.set_visible_devices([], 'GPU')
    visualize_inout(test_patient="pat_102")
    # all_filename = get_epilepsy_files()
    # for patient in range(1,25):
    #     print("Model {}".format(patient))
    #     generate_z_space(all_filename, test_patient=patient)
