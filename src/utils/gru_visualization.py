import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from utils.data import dataset_training, get_non_seizure_signal, get_epilepsiae_seizures, get_epilepsiae_test, \
    get_new_conv_w, get_epilepsiae_non_seizure, build_dataset_pickle as test_dataset
from vae_mmd import plot_mmd
from sklearn.metrics import roc_curve, auc

STATE_LEN = 899


def get_PCA_plotly(X, y, name):
    # df = px.data.iris()
    # X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    fig = px.scatter(components, x=0, y=1, color=y, title=name)
    fig.show()


def get_PCA(X, y, mmd_max, name):
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    plt.figure()
    plt.title(name)
    plt.scatter(x=components[:, 0], y= components[:, 1], c=y)
    plt.scatter(components[mmd_max, 0], components[mmd_max, 1], s=80, facecolors='none', edgecolors='r')
    print(mmd_max, components[mmd_max,:])
    plt.show()


def get_accuracy(y_predict , y_true):
    fpr, tpr, thresholds = roc_curve(y_true, y_predict)
    return auc(fpr, tpr)


def load_model(test_patient):
    arch = 'vae_free'
    subdirname = "../../temp/vae_mmd/integrated/{}/{}/z_minus1_v52".format(1024, arch)
    save_path = '{}/model/test_{}/saved_model/'.format(subdirname, test_patient)
    trained_model = tf.keras.models.load_model(save_path)
    print(trained_model.summary())
    intermediate_model = tf.keras.models.Model(inputs=trained_model.input,
                                               outputs=[
                                                   trained_model.output,
                                                   trained_model.get_layer('tf_op_layer_AddV2').output])
    return intermediate_model


def predict_(test_patient, model):
    sessions = test_dataset(test_patient, root='../../')
    out_list = np.zeros(0)
    true_list = np.zeros(0)
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
        print("out shape: {}".format(out.shape))
        print("true shape: {}".format(y_true.shape))
        mmd_argmax = np.argmax(out[0, STATE_LEN:-STATE_LEN, :])
        # print("z shape :{}".format(z[0, STATE_LEN:-STATE_LEN, :].shape))
        out_list = np.concatenate((out_list, out[0,STATE_LEN:-STATE_LEN,0]))
        true_list = np.concatenate((true_list, y_true))
        get_PCA(z[0, STATE_LEN:-STATE_LEN, :], y_true_section, mmd_argmax, node)
        # for idx in range(9):
        #     subdirname = "../../output/Conv/"
        #     mmd_edge_free = mmd_predicted[0, STATE_LEN:-STATE_LEN, idx]
        #     mmd_maximum = [np.argmax(mmd_edge_free)]
        #     name = "{}_{}".format(node, idx)
        #     plot_mmd(mmd_edge_free, mmd_maximum, y_true_section, name, subdirname)
    return get_accuracy(out_list, true_list)


def plot_AUCs():
    AUC_proposed = [0.9738982044813771, 0.186912814731304, 0.995100804791544, 0.9206983409133493, 0.9979496606334841,
                    0.46454173177176655, 0.9962129546886764, 0.7794261007708556, 0.8037654007438184, 0.9984635360015542,
                    0.9895854572331533, 0.6217678717413025, 0.5828515181611804, 0.5064780556681352, 0.7519378499336598,
                    0.37998509131569136, 1.0, 0.6640755794261384, 0.9358475894245722, 0.9606332583514519,
                    0.9643451311184434, 0.9997316082857339, 0.9958944579219429]

    AUC_baseline = [0.9575619528119899, 0.42455843667794063, 0.9689870117196433, 0.8264724606946361, 0.911825766145619,
                    0.5964214892224181, 0.9784243197164356, 0.8469607070287662, 0.9679046015155182, 0.9585288117783716,
                    0.9229300472355318, 0.6082874239258789, 0.7270177118804388, 0.42601387127560825, 0.6652350678229395,
                    0.7779471274160055, 0.9379094699225731, 0.8087869550491121, 0.9363116440074248, 0.780882378756304,
                    0.7923359837082622, 0.9821760374372032, 0.9473443175906155]
    x = np.arange(1,24)
    plt.bar(x- 0.1, AUC_proposed, width=0.15, label='proposed')
    plt.bar(x+0.1, AUC_baseline, width=0.15, label='baseline')
    plt.legend(loc=4)
    plt.show()


if __name__ == "__main__":
    tf.config.experimental.set_visible_devices([], 'GPU')
    auc_list = []
    for test_pat in [15]: #range(1, 24):
        model = load_model(test_pat)
        auc_predict = predict_(test_pat, model)
        print("AUC : {}".format(auc_predict))
        auc_list.append(auc_predict)
    print(auc_list)
    # plot_AUCs()

