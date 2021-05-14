import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils.data import dataset_training, get_non_seizure_signal, get_epilepsiae_seizures, get_epilepsiae_test, \
    get_new_conv_w, get_epilepsiae_non_seizure, build_dataset_pickle as test_dataset
from vae_mmd import plot_mmd
from scipy.signal import savgol_filter


STATE_LEN = 899


def load_model(test_patient):
    arch = 'vae_free'
    subdirname = "../../temp/vae_mmd/integrated/{}/{}/z_minus1_v52".format(1024, arch)
    save_path = '{}/model/test_{}/saved_model/'.format(subdirname, test_patient)
    trained_model = tf.keras.models.load_model(save_path)
    intermediate_model = tf.keras.models.Model(inputs= trained_model.input,
                                               outputs=trained_model.get_layer('conv_interval').output)
    return intermediate_model


def predict_(test_patient, model):
    sessions = get_epilepsiae_test(test_patient, root='../../')
    for node in sessions.keys():
        X = sessions[node]['data']
        y_true = sessions[node]['label']

        if np.sum(y_true) == 0:
            continue

        X_section = X
        y_true_section = y_true

        X_section = np.expand_dims(X_section, 0)
        X_edge = get_epilepsiae_non_seizure(test_patient, state_len=STATE_LEN, root='../..')
        X_section = np.concatenate((X_edge, X_section, X_edge), axis=1)

        mmd_predicted = model.predict(X_section)
        for idx in range(9):
            subdirname = "../../output/Conv/"
            mmd_edge_free = mmd_predicted[0, STATE_LEN:-STATE_LEN, idx]
            mmd_maximum = [np.argmax(mmd_edge_free)]
            name = "{}_{}".format(node, idx)
            plot_mmd(mmd_edge_free, mmd_maximum, y_true_section, name, subdirname)


def plot_loss():
    train_prop = [0.07395225, 0.06909066, 0.068582386, 0.06630091, 0.063985184, 0.06426628, 0.06658931, 0.059268847,
             0.05781046, 0.05774517, 0.055948384, 0.054986846, 0.055713568, 0.057495274, 0.057696924, 0.05247025,
             0.05077579, 0.057280157, 0.05039756, 0.052063618, 0.050503626, 0.04881094, 0.0491899, 0.058053352,
             0.04965144, 0.046025872, 0.049840588, 0.05534634, 0.05308451, 0.043698166, 0.04635534, 0.045784563,
             0.04253897, 0.04106991, 0.042335063, 0.04617808, 0.04012497, 0.040421426, 0.039329275, 0.050366]
    val_prop = [0.04254984, 0.04109845, 0.0414726, 0.040244382, 0.0377069, 0.033698905, 0.035142835, 0.03358663, 0.030590137,
           0.034239404, 0.030521324, 0.031657856, 0.033846598, 0.032659728, 0.031011075, 0.029501917, 0.028027873,
           0.031753477, 0.03017194, 0.029849041, 0.028522674, 0.032253243, 0.036512762, 0.03662708, 0.027777221,
           0.02935879, 0.032486405, 0.02919231, 0.026753126, 0.03454768, 0.0330078, 0.03620705, 0.036402293,
           0.033403143, 0.033009045, 0.02890453, 0.030994445, 0.033234607, 0.035094086, 0.030933877]

    train_base = [0.10975403, 0.11720013, 0.091526434, 0.091829374, 0.08983434, 0.0875075, 0.08235574, 0.09672947,
             0.10174493, 0.09686926, 0.084549375, 0.08622021, 0.08586749, 0.079288684, 0.08279883, 0.08586897,
             0.08446617, 0.084386416, 0.08927809, 0.082750276, 0.08369371, 0.08425126, 0.08487934, 0.078737736,
             0.07523212, 0.07964456, 0.08513628, 0.115592316, 0.078626126, 0.08264745, 0.09221768, 0.07549098,
             0.07830557, 0.08071485, 0.08965241, 0.07758921, 0.07661322, 0.07744231, 0.08329227, 0.081882514,
             0.076377645, 0.073281564, 0.08288187, 0.07910564, 0.0751621, 0.075463034, 0.078345194, 0.08086707,
             0.07576258, 0.089513674, 0.077933274, 0.07593552, 0.07781791, 0.076989725, 0.072942436, 0.0730275,
             0.07687733, 0.075997606, 0.07530323, 0.07843173, 0.078272864, 0.07589647, 0.09461279, 0.07720455,
             0.07697463, 0.0717944, 0.07385931, 0.07322432, 0.07533424, 0.07342027, 0.07072813, 0.073467135, 0.06997508,
             0.071640626, 0.081630416, 0.072745375, 0.07181584, 0.0724938, 0.06980331, 0.075619645, 0.06936413,
             0.071611024, 0.114292674, 0.079954006, 0.06908138, 0.09315644, 0.068982325, 0.0708083, 0.06836273,
             0.076493405, 0.06915832, 0.06709041, 0.06791162, 0.07499703, 0.07689925, 0.07244946, 0.073226295,
             0.066079885, 0.06751405, 0.07191193]
    val_base = [0.087705106, 0.07553537, 0.06587043, 0.066178404, 0.057555385, 0.06095126, 0.052580968, 0.12242991,
           0.055422843, 0.05366252, 0.05210326, 0.044242185, 0.04446043, 0.048425306, 0.05687013, 0.046370376,
           0.0530493, 0.058534782, 0.06575366, 0.050307978, 0.04833868, 0.042863775, 0.047523517, 0.051563077,
           0.04094267, 0.046149686, 0.053233847, 0.058986444, 0.050997153, 0.04933599, 0.08645082, 0.044921685,
           0.06390143, 0.044864155, 0.055939823, 0.041849043, 0.040725697, 0.046840947, 0.048032545, 0.05747249,
           0.04564674, 0.03990192, 0.06566421, 0.04039668, 0.041121025, 0.037081763, 0.054052036, 0.041028157,
           0.043728232, 0.048646513, 0.03953632, 0.04976702, 0.043569084, 0.042898554, 0.04760137, 0.044751894,
           0.03898823, 0.041361734, 0.04650889, 0.042640094, 0.039624043, 0.044144258, 0.06663338, 0.045898166,
           0.043672547, 0.048585653, 0.042066645, 0.051326517, 0.04347228, 0.04194375, 0.0429822, 0.0390038,
           0.042480793, 0.05124293, 0.05053661, 0.046064176, 0.04035553, 0.036429852, 0.04134134, 0.04718546, 0.0406843,
           0.057932973, 0.06662242, 0.04116848, 0.035855103, 0.054905087, 0.046357166, 0.039973732, 0.039566986,
           0.041460633, 0.04260156, 0.046209365, 0.053320237, 0.047437526, 0.051537026, 0.04676223, 0.054507323,
           0.043914992, 0.046382703, 0.047643375]

    plt.figure(figsize=(10,6))

    plt.plot(savgol_filter(train_prop, 39, 2), 'r')
    plt.plot(train_prop, '--r')

    plt.plot(savgol_filter(val_prop, 39, 2), 'g')
    plt.plot(val_prop, '--g')

    plt.plot(savgol_filter(train_base, 99, 2), 'b')
    plt.plot(train_base, '--b')

    plt.plot(savgol_filter(val_base, 99, 2), 'c')
    plt.plot(val_base, '--c')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    tf.config.experimental.set_visible_devices([], 'GPU')
    # test_pat = 'pat_102'
    # model = load_model(1)
    # predict_(test_pat, model)
    plot_loss()
