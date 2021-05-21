import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from utils.data import dataset_training, get_non_seizure_signal, get_epilepsiae_seizures, get_epilepsiae_test, \
    get_new_conv_w, get_epilepsiae_non_seizure, build_dataset_pickle as test_dataset
from vae_mmd import plot_mmd
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score
from utils.params import pat_list
import json

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
    plt.scatter(x=components[:, 0], y=components[:, 1], c=y)
    plt.scatter(components[mmd_max, 0], components[mmd_max, 1], s=80, facecolors='none', edgecolors='r')
    print(mmd_max, components[mmd_max, :])
    plt.show()


def get_accuracy(y_predict, y_true):
    fpr, tpr, thresholds = roc_curve(y_true, y_predict)
    return auc(fpr, tpr)
    # y_pred = y_predict > 0.5
    # return accuracy_score(y_true, y_pred)


def plot_roc():
    plt.figure()
    for method in ['baseline_unseen', 'proposed_unseen']:
        data = json.load(open('{}.json'.format(method), 'r'))
        y_true = data['true']
        y_predict = data['predict']
        fpr, tpr, thresholds = roc_curve(y_true, y_predict)
        plt.plot(fpr, tpr, marker='.')
    plt.savefig()


def get_within_between(z, y_true):
    y_seizure = np.where(y_true != 0)[0]
    y_non_seizure = np.where(y_true == 0)[0]
    z_seizure = z[y_seizure]
    z_non_seizure = z[y_non_seizure]
    m_s = np.mean(z_seizure, axis=0)
    m_n = np.mean(z_non_seizure, axis=0)
    m = np.mean(z, axis=0)
    S_b = z_seizure.shape[0] * np.dot(m_s - m, m_s - m) + z_non_seizure.shape[0] * np.dot(m_n - m, m_n - m)

    w_s = np.sum(np.square(z_seizure - m_s))
    w_n = np.sum(np.square(z_non_seizure - m_n))
    S_w = w_n + w_s
    print("Mean Class\nSeizure: {}\nNon-Seizure: {}".format(m_s, m_n))
    print("Sb: {}".format(S_b))
    print("Sw : {}".format(S_w))
    return S_b, S_w, S_b / S_w


def load_model(test_patient):
    arch = 'vae_free'
    subdirname = "../../temp/vae_mmd/integrated/{}/{}/Anthony_v53".format(1024, arch)
    # subdirname = "../../output/vae/vae_supervised/seg_n_1024/beta_1e-05/latent_dim_16/lr_0.0001/decay_0.5/gamma_0.0"
    save_path = '{}/model/test_{}/saved_model/'.format(subdirname, test_patient)
    # save_path = '{}/test_{}/'.format(subdirname, test_patient)
    trained_model = tf.keras.models.load_model(save_path)
    print(trained_model.summary())
    intermediate_model = tf.keras.models.Model(inputs=trained_model.input,
                                               outputs=[
                                                   trained_model.output,
                                                   trained_model.get_layer('dense1').input])
    return intermediate_model


def predict_(test_patient, model):
    sessions = get_epilepsiae_test(test_patient, root='../../')
    out_list = np.zeros(0)
    true_list = np.zeros(0)
    J_dict = {}
    for node in sessions.keys():
        X = sessions[node]['data']
        y_true = sessions[node]['label']

        if np.sum(y_true) == 0:
            continue

        X_section = X
        length = X.shape[0]
        y_true_section = y_true

        X_section = np.expand_dims(X_section, 0)
        X_edge = get_epilepsiae_non_seizure(test_patient, state_len=STATE_LEN, root='../..')
        X_section = np.concatenate((X_edge, X_section, X_edge), axis=1)

        out, z = model.predict(X_section)
        z = z[0, STATE_LEN:-STATE_LEN, :]
        Sb, Sw, J = get_within_between(z, y_true)
        # print("Sw: {}, Sb: {}, J: {}".format(Sw, Sb, J))
        J_dict[node] = J
        # mmd_argmax = np.argmax(out[0, STATE_LEN:-STATE_LEN, :])
        out = out[0, STATE_LEN:-STATE_LEN, 0]
        # plt.plot(out, 'r')
        out_list = np.concatenate((out_list, out))
        true_list = np.concatenate((true_list, y_true))
        # get_PCA(z[0, STATE_LEN:-STATE_LEN, 0, :], y_true_section, mmd_argmax, node)
        # for idx in range(9):
        #     subdirname = "../../output/Conv/"
        #     mmd_edge_free = mmd_predicted[0, STATE_LEN:-STATE_LEN, idx]
        #     mmd_maximum = [np.argmax(mmd_edge_free)]
        #     name = "{}_{}".format(node, idx)
        #     plot_mmd(mmd_edge_free, mmd_maximum, y_true_section, name, subdirname)
    # auc_pat = get_accuracy(out_list, true_list)
    return out_list, true_list, J_dict


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
    x = np.arange(1, 24)
    plt.bar(x - 0.1, AUC_proposed, width=0.15, label='proposed')
    plt.bar(x + 0.1, AUC_baseline, width=0.15, label='baseline')
    plt.legend(loc=4)
    plt.show()


def plot_loss():
    train_base = [0.5208320, 0.12061878, 0.114064395, 0.108487554, 0.10401606, 0.10320567, 0.09876294, 0.101990275, 0.096678995,
             0.09617278, 0.094597794, 0.11623233, 0.097215466, 0.09235345, 0.09451152, 0.09299788, 0.09301158,
             0.09191418, 0.095877595, 0.091506325, 0.092040524, 0.09099137, 0.092499524, 0.08912101, 0.090801306,
             0.0888222, 0.088640615, 0.08819486, 0.08695013, 0.088359125, 0.088425696, 0.08699258, 0.08805715,
             0.08746431, 0.08492977, 0.08507419, 0.08405146, 0.08277092, 0.08779592, 0.08344959, 0.08203691,
             0.082594715, 0.08239346, 0.08572996, 0.0868499, 0.08177531, 0.08135989, 0.079726815, 0.079694614,
             0.07995395, 0.09173624, 0.08089297, 0.07733864, 0.078912795, 0.07739585, 0.07949289, 0.08847409,
             0.07522238, 0.07532276, 0.07577384, 0.07433823, 0.07611246, 0.07425541, 0.08835441, 0.07451748,
             0.080530874, 0.07466169, 0.07482684, 0.07377973, 0.07277359, 0.073005006, 0.074152716, 0.07668282,
             0.07116853, 0.07127991, 0.07130532, 0.07463388, 0.070745006, 0.07382759, 0.06964074, 0.07152174,
             0.07037546, 0.0726581, 0.0693998, 0.072314024, 0.069680706, 0.06826116, 0.06940761, 0.0685298, 0.068792894,
             0.06738007, 0.06987364, 0.0702186, 0.06628634, 0.07138748, 0.06705796, 0.06593372, 0.067826666, 0.06690713,
             0.07181488, 0.070307195, 0.064575486, 0.067048095, 0.07040514, 0.06519824, 0.06459863, 0.06711179,
             0.06572854, 0.06425247, 0.06506234, 0.072436586, 0.06333936, 0.064436436, 0.06806895, 0.066828474,
             0.0639176, 0.0639935, 0.062286958, 0.062183276, 0.063628316, 0.06446373, 0.061470333, 0.06342898,
             0.065777354, 0.063479185, 0.06133291, 0.06642751, 0.063455224, 0.07120049, 0.061455097, 0.06807866,
             0.06338489, 0.061225273, 0.06171488, 0.060288798, 0.060865056, 0.068118066, 0.06213498, 0.059647627,
             0.05918764, 0.059846748, 0.059312113, 0.059809137, 0.06188498, 0.06348806, 0.059059124, 0.0610048,
             0.059005186, 0.058855012, 0.059645027, 0.078964286, 0.064386554, 0.057967026, 0.058337715, 0.059039596,
             0.057708703, 0.05892552, 0.05645577, 0.058195997, 0.058719933, 0.057316698, 0.058127258, 0.05689173,
             0.056266595, 0.057502758, 0.057116654, 0.057434242, 0.05671209, 0.058260065, 0.0579633, 0.05815178,
             0.056657232, 0.056319755, 0.056790818, 0.057397246, 0.05941951, 0.056166876, 0.058032595, 0.05608712,
             0.05584224, 0.061570153, 0.056142464, 0.056343522, 0.05900562, 0.06124948, 0.05520105, 0.05631957,
             0.05540107, 0.055211086, 0.056433514, 0.055972144, 0.05699303, 0.055672675, 0.05851944, 0.0538344,
             0.062326852, 0.054765765, 0.058906626, 0.057791743, 0.06255012, 0.05443941]
    val_base = [0.5020043, 0.09053746, 0.08586736, 0.07014158, 0.06718455, 0.072716504, 0.0667405, 0.06575075, 0.06307745, 0.06283447,
           0.058039676, 0.070541024, 0.05948223, 0.05532927, 0.0582478, 0.054326426, 0.054049633, 0.053677205,
           0.061710414, 0.052727576, 0.059398536, 0.05565812, 0.050945293, 0.05383401, 0.054033373, 0.050629742,
           0.05474617, 0.0508901, 0.05084675, 0.04825178, 0.048283402, 0.050078746, 0.046737053, 0.052911837,
           0.049616788, 0.053928502, 0.051904667, 0.04867808, 0.04903507, 0.04580939, 0.047801707, 0.046318244,
           0.04876499, 0.057849027, 0.048851095, 0.04762938, 0.053120952, 0.049340095, 0.050407697, 0.05255597,
           0.06913991, 0.055938356, 0.05057871, 0.04699445, 0.05060215, 0.049512204, 0.06947014, 0.047697835,
           0.04737008, 0.051197574, 0.046336118, 0.046455566, 0.047754444, 0.05956154, 0.05396718, 0.055455673,
           0.050453044, 0.049826458, 0.05190908, 0.053366356, 0.05068318, 0.049700048, 0.051643383, 0.050233893,
           0.051677685, 0.052059893, 0.052289214, 0.050322577, 0.062153116, 0.05296635, 0.05332456, 0.05114777,
           0.0508571, 0.055848837, 0.05455623, 0.053134162, 0.054865208, 0.056501184, 0.054237604, 0.059271622,
           0.06269153, 0.058899086, 0.055610128, 0.059389446, 0.054673724, 0.05626967, 0.058903832, 0.059404403,
           0.06518126, 0.07304633, 0.06749336, 0.061445586, 0.06559925, 0.057174377, 0.057622254, 0.058110703,
           0.06745416, 0.061808996, 0.05952898, 0.059035506, 0.061204795, 0.06401205, 0.06343734, 0.07195949, 0.0639838,
           0.060260225, 0.06561228, 0.065066114, 0.06533865, 0.06699535, 0.06188521, 0.06644715, 0.06525127,
           0.062727556, 0.06361665, 0.0687621, 0.06235404, 0.06878112, 0.09276538, 0.06555204, 0.06767696, 0.06284831,
           0.06260198, 0.06847393, 0.07421159, 0.06432828, 0.06872222, 0.06711638, 0.07045207, 0.07012524, 0.06839242,
           0.07055096, 0.0708506, 0.06896302, 0.08976114, 0.077666834, 0.07380697, 0.0732713, 0.07464951, 0.08431944,
           0.06928722, 0.06953187, 0.07445921, 0.07684821, 0.070367984, 0.0752387, 0.07286769, 0.08021419, 0.08697529,
           0.07152231, 0.07604594, 0.06886061, 0.07320593, 0.07815247, 0.07605521, 0.075487345, 0.07474963, 0.071386315,
           0.074843, 0.07925126, 0.07183477, 0.08077211, 0.08689828, 0.07642361, 0.07836349, 0.07876528, 0.0772546,
           0.07540582, 0.07983153, 0.077923276, 0.07505929, 0.07621439, 0.077860095, 0.07340562, 0.1032433, 0.083039105,
           0.07452954, 0.08504611, 0.08019695, 0.07707134, 0.08228986, 0.075702086, 0.07555824, 0.075577594,
           0.083352976, 0.075245745, 0.08436415, 0.07947133, 0.07996021, 0.08183285, 0.08836517]

    train_prop = [0.8796004, 0.0809545, 0.07647995, 0.07604479, 0.072958544, 0.075662024, 0.07787212, 0.067573205,
             0.064818874, 0.06494394, 0.0641031, 0.062163066, 0.059288155, 0.060104027, 0.057992592, 0.05650213,
             0.054881584, 0.05395069, 0.05246209, 0.053017557, 0.0528928, 0.052119866, 0.049731348, 0.049219,
             0.052162573, 0.049560238, 0.05465308, 0.05358653, 0.05578692, 0.045820266, 0.04565361, 0.047829397,
             0.045987286, 0.042451568, 0.043405265, 0.045626085, 0.044464123, 0.043412685, 0.044035856, 0.038122382,
             0.039056025, 0.037668027, 0.036392838, 0.04262462, 0.039514877, 0.0377689, 0.03299426, 0.037715055,
             0.037809093, 0.03344347, 0.035811946, 0.030611752, 0.032286108, 0.029965727, 0.027825665, 0.027238501,
             0.029654924, 0.025974387, 0.031404838, 0.02829731, 0.029840223, 0.028835885, 0.02615743, 0.025364969,
             0.025306422, 0.026650673, 0.024867028, 0.025736704, 0.023153247, 0.030482255, 0.02253643, 0.024337549,
             0.022431321, 0.021905439, 0.022943672, 0.02097403, 0.021511706, 0.020532142, 0.018494733, 0.020203685,
             0.018770598, 0.018152233, 0.020264518, 0.018889466, 0.017469773, 0.018668188, 0.017795404, 0.020173809,
             0.017986668, 0.023010252, 0.015530022, 0.016002307, 0.015992943, 0.015496324, 0.016542073, 0.0126962885,
             0.016875362, 0.017885143, 0.014072292, 0.013811122]
    val_prop = [0.8988607, 0.04725663, 0.047616683, 0.04201163, 0.03861454, 0.03898997, 0.035857093, 0.03631827, 0.038912307,
           0.038024373, 0.036908988, 0.03622091, 0.036981277, 0.033508655, 0.033961274, 0.039850798, 0.035303537,
           0.03453926, 0.037372097, 0.037358798, 0.035172798, 0.034923837, 0.03623529, 0.030455502, 0.03339322,
           0.045322224, 0.048679985, 0.047383565, 0.053870283, 0.029856918, 0.032955848, 0.037770815, 0.03380523,
           0.031185966, 0.040142715, 0.031566832, 0.039467596, 0.04369635, 0.050239883, 0.04090422, 0.0347345,
           0.044359855, 0.04937221, 0.062331304, 0.04701491, 0.029479519, 0.0406843, 0.036671646, 0.051451802,
           0.040146742, 0.03919628, 0.03722184, 0.045277804, 0.03989306, 0.043224346, 0.045969784, 0.046433337,
           0.047182415, 0.03235215, 0.047196757, 0.029580034, 0.040883288, 0.053958885, 0.050859947, 0.056697533,
           0.044222467, 0.054805603, 0.04425262, 0.06074646, 0.040459104, 0.057377055, 0.05484994, 0.057438783,
           0.052543726, 0.035548784, 0.059269354, 0.052352253, 0.05408903, 0.06323591, 0.064431734, 0.053326353,
           0.054221712, 0.05479148, 0.06754122, 0.073611334, 0.070545785, 0.06146686, 0.06218679, 0.06352546,
           0.050786473, 0.05676218, 0.06318102, 0.06781284, 0.056076918, 0.069646314, 0.06519431, 0.0696569,
           0.058675777, 0.07144404, 0.0555296]

    k_ascending = 10
    proposed_i = len(val_prop)
    base_i = len(val_base)
    for i in range(len(val_prop) - k_ascending):
        filtered = gaussian_filter1d(val_prop[:i+k_ascending], sigma=5)
        if list(filtered[i:i+k_ascending]) == list(sorted(filtered[i:i+k_ascending])):
            print("i: {}".format(i))
            proposed_i = i+k_ascending
            break
    for i in range(len(val_base)- k_ascending):
        filtered = gaussian_filter1d(val_base[:i + k_ascending], sigma=5)
        if list(filtered[i:i + k_ascending]) == list(sorted(filtered[i:i + k_ascending])):
            print("i: {}".format(i))
            base_i = i+k_ascending
            break

    plt.figure(figsize=(12, 6))

    plt.plot(gaussian_filter1d(train_prop, sigma=8)[:proposed_i+1], 'r', label='Proposed Training')
    plt.plot(proposed_i, gaussian_filter1d(train_prop, sigma=8)[proposed_i], '*k')
    # plt.plot(train_prop, '--r')

    plt.plot(gaussian_filter1d(val_prop, sigma=8)[:proposed_i+1], 'g', label='Proposed Validation')
    plt.plot(proposed_i, gaussian_filter1d(val_prop, sigma=8)[proposed_i], '*k')
    # plt.plot(val_prop, '--g')

    plt.plot(gaussian_filter1d(train_base, sigma=8)[:base_i+1], 'b', label='C-CNN Training')
    # plt.plot(train_base, '--b')

    plt.plot(gaussian_filter1d(val_base, sigma=8)[:base_i+1], 'c', label='C-CNN Validation')
    # plt.plot(val_base, '--c')
    plt.legend(fontsize=12)
    plt.xlim([0,base_i])
    plt.xlabel('Epochs', fontsize=16)
    plt.xticks(np.arange(0, base_i+ 1, 10), fontsize=12)
    plt.grid(which='both')
    plt.ylabel('Loss', fontsize=16)
    plt.yticks(np.arange(0, 0.16, 0.02), fontsize=11)
    plt.savefig('../../output/images/loss')


if __name__ == "__main__":
    tf.config.experimental.set_visible_devices([], 'GPU')
    # test_pat = 'pat_102'
    # model = load_model(1)
    # predict_(test_pat, model)
    # auc_list = []
    # out_all = np.zeros(0)
    # true_all = np.zeros(0)
    # length = []
    # J_list = {}
    # model = load_model(-1)
    # for test_pat in pat_list:
    #     out, true, J = predict_(test_pat, model)
    #     out_all = np.concatenate((out_all, out))
    #     true_all = np.concatenate((true_all, true))
    #     J_list.update(J)
    #     # print(auc_list)
    #     # print(length)
    #     # auc_list.append(auc_pat)
    # auc_total = get_accuracy(out_all, true_all)
    # dict_out = {'predict': out_all.tolist(), 'true': true_all.tolist()}
    # json.dump(dict_out, open('baseline_unseen.json', 'w'))
    plot_roc()
    # print("Total AUC: {}".format(auc_total))
    #
    # print("J : {}".format(J_list))
    # plot_AUCs()
    # plot_loss()
