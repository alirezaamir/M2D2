import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
from utils.data import dataset_training, get_non_seizure_signal, get_epilepsiae_seizures, get_epilepsiae_test, \
    get_new_conv_w, get_epilepsiae_non_seizure, build_dataset_pickle as test_dataset
from vae_mmd import plot_mmd


STATE_LEN = 899


def get_PCA(X, y, name):
    # df = px.data.iris()
    # X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    print(X)
    pca = PCA(n_components=3)
    components = pca.fit_transform(X)
    fig = px.scatter_3d(
        components, title=name, color=y, x=0, y=1, z=2,
        labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
    )
    fig.show()


def load_model(test_patient):
    arch = 'vae_free'
    subdirname = "../../temp/vae_mmd/integrated/{}/{}/Anthony_v53".format(1024, arch)
    save_path = '{}/model/test_{}/saved_model/'.format(subdirname, test_patient)
    trained_model = tf.keras.models.load_model(save_path)
    print(trained_model.summary())
    intermediate_model = tf.keras.models.Model(inputs= trained_model.input,
                                               outputs=[
                                                   trained_model.output,
                                               trained_model.get_layer('tf_op_layer_AddV2').output])
    return intermediate_model


def predict_(test_patient, model):
    sessions = test_dataset(test_patient, root='../../')
    for node in sessions.keys():
        X = sessions[node]['data']
        y_true = sessions[node]['label']

        if np.sum(y_true) == 0:
            continue

        X_section = X
        y_true_section = y_true

        X_section = np.expand_dims(X_section, 0)
        X_edge = get_non_seizure_signal(test_patient, state_len=STATE_LEN, root='../..')
        X_section = np.concatenate((X_edge, X_section, X_edge), axis=1)

        out, z = model.predict(X_section)
        print("z shape :{}".format(z[0,STATE_LEN:-STATE_LEN,:].shape))
        get_PCA(z[0,STATE_LEN:-STATE_LEN,:], y_true_section, node)
        # for idx in range(9):
        #     subdirname = "../../output/Conv/"
        #     mmd_edge_free = mmd_predicted[0, STATE_LEN:-STATE_LEN, idx]
        #     mmd_maximum = [np.argmax(mmd_edge_free)]
        #     name = "{}_{}".format(node, idx)
        #     plot_mmd(mmd_edge_free, mmd_maximum, y_true_section, name, subdirname)



if __name__ == "__main__":
    tf.config.experimental.set_visible_devices([], 'GPU')
    test_pat = 3
    model = load_model(test_pat)
    predict_(test_pat, model)
