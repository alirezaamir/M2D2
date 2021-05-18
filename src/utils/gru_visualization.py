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
    sessions = test_dataset(test_patient, root='../../')
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
        X_edge = get_non_seizure_signal(test_patient, state_len=STATE_LEN, root='../..')
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
    train_base = [0.114879616, 0.107278965, 0.10444485, 0.10254134, 0.10885842, 0.11081395, 0.10317421, 0.100420974,
                  0.098123536, 0.10425612, 0.09434872, 0.10829816, 0.09290178, 0.09835106, 0.101057105, 0.11547324,
                  0.09437633, 0.10342342, 0.096436545, 0.09994636, 0.111055784, 0.09458033, 0.103790514, 0.10127968,
                  0.094952576, 0.099042594, 0.10139243, 0.09873186, 0.09965446, 0.1012714, 0.10808665, 0.09593195,
                  0.094999656, 0.0964515, 0.09156221, 0.09251862, 0.093612, 0.091434486, 0.091626175, 0.10155304,
                  0.09314952, 0.09382901, 0.09097023, 0.09260536, 0.08890163, 0.09110128, 0.09591684, 0.09797197,
                  0.09272714, 0.11030274, 0.10093895, 0.09042642, 0.089018926, 0.08951192, 0.09413556, 0.091330916,
                  0.08828767, 0.09287061, 0.107926995, 0.08909056, 0.09453289, 0.0875374, 0.087889984, 0.09534812,
                  0.092772044, 0.08551638, 0.0922854, 0.10635716, 0.094906546, 0.09169916, 0.08631686, 0.09116414,
                  0.09133782, 0.0874106, 0.09000756, 0.09209926, 0.08375025, 0.08784057, 0.08704919, 0.08734523,
                  0.08699805, 0.084864266, 0.08377412, 0.08470347, 0.09622674, 0.09095345, 0.08750315, 0.084677145,
                  0.084696144, 0.08905664, 0.10697954, 0.09351126, 0.08812659, 0.082045816, 0.10312486, 0.08433531,
                  0.08161334, 0.08599776, 0.08207344, 0.08681356, 0.100244306, 0.08523705, 0.09490935, 0.08631344,
                  0.08466678, 0.08356826, 0.09225825, 0.085485786, 0.084931925, 0.094430864, 0.095605955, 0.09105059,
                  0.085830346, 0.086021915, 0.087280184, 0.09193634, 0.08402268, 0.07968612, 0.08857901, 0.07940532,
                  0.08452076, 0.086159945, 0.09045677, 0.08413133, 0.08905556, 0.0819871, 0.08866317, 0.08265982,
                  0.08964157, 0.08683931, 0.07923411, 0.0804829, 0.080432504, 0.08579106, 0.0800323, 0.07954132,
                  0.084073685, 0.085508145, 0.16271298, 0.07851243, 0.08316745, 0.08724444, 0.08653795, 0.08193126,
                  0.08242713, 0.08753, 0.08099361, 0.083801284, 0.0781318, 0.08235223, 0.08046249, 0.08228921,
                  0.08183326, 0.08069019, 0.08266254, 0.08722024, 0.07782189, 0.07691247, 0.07556646, 0.074100606,
                  0.08297451, 0.08291033, 0.09631123, 0.079868644, 0.07776102, 0.07686608, 0.13170889, 0.07921742,
                  0.07534597, 0.07723164, 0.08075483, 0.07731443, 0.08607863, 0.07894633, 0.080153294, 0.077706635,
                  0.08691954, 0.07893083, 0.081457324, 0.07898775, 0.077745005, 0.08012711, 0.08060459, 0.08248443,
                  0.08313771, 0.08468673, 0.08265309, 0.07405262, 0.0829077, 0.08233521, 0.08175606, 0.08153338,
                  0.08059495, 0.07687554, 0.081093214, 0.08505147, 0.08392958, 0.08492152, 0.07447414, 0.07888174]
    val_base = [0.09991706, 0.0961624, 0.10325032, 0.09597794, 0.10110531, 0.09828942, 0.083303474, 0.084511474,
                0.09414271, 0.09606542, 0.09049848, 0.099779725, 0.08493221, 0.091798745, 0.08544467, 0.095061034,
                0.08867508, 0.090762675, 0.08310804, 0.08742963, 0.10896893, 0.08168366, 0.08916236, 0.08024325,
                0.08522293, 0.08864944, 0.08584675, 0.104135886, 0.09022265, 0.08808702, 0.10124971, 0.0920497,
                0.093806304, 0.12778413, 0.10460161, 0.07649121, 0.08818212, 0.08315223, 0.08713903, 0.0916338,
                0.083020076, 0.09144743, 0.08488615, 0.11124948, 0.08934724, 0.08811841, 0.0894029, 0.08810265,
                0.084107436, 0.11830309, 0.13037404, 0.08589151, 0.08046632, 0.08420219, 0.08340391, 0.08672674,
                0.08317313, 0.08483741, 0.10173912, 0.082936436, 0.087723024, 0.086721234, 0.080978066, 0.09795986,
                0.08472202, 0.08408828, 0.0852692, 0.09328748, 0.08736597, 0.08989051, 0.09629722, 0.086397104,
                0.08478857, 0.08719018, 0.08755637, 0.08606708, 0.08815031, 0.0883621, 0.08311005, 0.092780516,
                0.07806347, 0.08541822, 0.094325095, 0.08936081, 0.08284243, 0.09940675, 0.07911638, 0.09736012,
                0.08478639, 0.09071399, 0.11434852, 0.08460361, 0.10529117, 0.08366225, 0.12986983, 0.09181473,
                0.087146424, 0.10493255, 0.089121014, 0.08857711, 0.08431386, 0.08486363, 0.0888386, 0.08080483,
                0.09249929, 0.082950324, 0.105526835, 0.09020754, 0.081770964, 0.09336423, 0.08770881, 0.08545587,
                0.08036543, 0.08034243, 0.08393748, 0.088484876, 0.08636041, 0.091404095, 0.08527219, 0.08373651,
                0.099428855, 0.09075035, 0.08914225, 0.08060853, 0.08861554, 0.08189408, 0.0863594, 0.08094394,
                0.117897324, 0.11401948, 0.08267516, 0.08644626, 0.082081735, 0.08740857, 0.08797805, 0.08257659,
                0.086056486, 0.08924369, 0.18376127, 0.091138594, 0.08304439, 0.08253578, 0.08493695, 0.08405023,
                0.085530266, 0.087808006, 0.09052508, 0.08650216, 0.08493129, 0.094521426, 0.09435177, 0.0873112,
                0.08260586, 0.08621031, 0.08884082, 0.08622752, 0.10524772, 0.09015063, 0.086996555, 0.09052586,
                0.0942563, 0.09431229, 0.1298396, 0.08916094, 0.0955913, 0.08557007, 0.19355777, 0.094834805,
                0.10188458, 0.08740301, 0.085568614, 0.09322231, 0.09179729, 0.08805706, 0.089192815, 0.090177596,
                0.09415495, 0.09038047, 0.09612787, 0.08686837, 0.08559256, 0.09519824, 0.12283238, 0.08853301,
                0.09819088, 0.09788862, 0.10088797, 0.0870884, 0.088644356, 0.106706046, 0.08778971, 0.09689851,
                0.0866846, 0.08447019, 0.09868926, 0.09489682, 0.10689733, 0.10029785, 0.09266517, 0.0854351]

    train_prop = [0.08691157, 0.08266329, 0.07848644, 0.08130687, 0.07499847, 0.07567355, 0.08183796, 0.071556225,
                  0.07028796, 0.06748302, 0.06748672, 0.064914145, 0.06632703, 0.062515065, 0.068416886, 0.06038705,
                  0.06583887, 0.05876884, 0.057993755, 0.06271187, 0.05589436, 0.05483376, 0.055886336, 0.05801319,
                  0.056222554, 0.054858673, 0.051101547, 0.049334817, 0.067900196, 0.04863401, 0.04866196, 0.05084523,
                  0.04586129, 0.04806399, 0.04546932, 0.04845136, 0.045145575, 0.041045677, 0.040565263, 0.04303995,
                  0.043068945, 0.04586151, 0.04879127, 0.038278297, 0.03561922, 0.049554866, 0.0336102, 0.035924193,
                  0.032459654, 0.03302303, 0.045124605, 0.03395535, 0.035147287, 0.028081521, 0.03200286, 0.027875016,
                  0.02676873, 0.02550395, 0.023925664, 0.025806611]

    val_prop = [0.07349116, 0.07365676, 0.06822809, 0.06685868, 0.061658017, 0.0626298, 0.07258935, 0.064475276,
                0.06348247, 0.057596616, 0.05949148, 0.057993464, 0.060733277, 0.058187507, 0.064713374, 0.056577455,
                0.058978207, 0.060202874, 0.058975734, 0.069730796, 0.06265193, 0.062029757, 0.059897386, 0.05726696,
                0.06819255, 0.062682584, 0.07168895, 0.07008407, 0.06364006, 0.06267165, 0.06388435, 0.07055283,
                0.07036804, 0.070330136, 0.06297536, 0.07884774, 0.06788459, 0.070985906, 0.07179245, 0.07981455,
                0.078957364, 0.07622244, 0.092719406, 0.07983903, 0.08141426, 0.06580017, 0.08225299, 0.07725382,
                0.08102153, 0.085061595, 0.06575966, 0.07814225, 0.09124658, 0.08279137, 0.08061912, 0.08506591,
                0.0826574, 0.08561981, 0.08335278, 0.08300086]

    plt.figure(figsize=(10, 6))

    plt.plot(gaussian_filter1d(train_prop, sigma=5), 'r', label='Train Proposed')
    # plt.plot(train_prop, '--r')

    plt.plot(gaussian_filter1d(val_prop, sigma=5), 'g', label='Val Proposed')
    # plt.plot(val_prop, '--g')

    plt.plot(gaussian_filter1d(train_base, sigma=5), 'b', label='Train baseline')
    # plt.plot(train_base, '--b')

    plt.plot(gaussian_filter1d(val_base, sigma=5), 'c', label='Val baseline')
    # plt.plot(val_base, '--c')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    tf.config.experimental.set_visible_devices([], 'GPU')
    # test_pat = 'pat_102'
    # model = load_model(1)
    # predict_(test_pat, model)
    auc_list = []
    out_all = np.zeros(0)
    true_all = np.zeros(0)
    length = []
    J_list = {}
    for test_pat in range(1,24):
        model = load_model(test_pat)
        out, true, J = predict_(test_pat, model)
        out_all = np.concatenate((out_all, out))
        true_all = np.concatenate((true_all, true))
        J_list.update(J)
        # print(auc_list)
        # print(length)
        # auc_list.append(auc_pat)
    auc_total = get_accuracy(out_all, true_all)
    print("Total AUC: {}".format(auc_total))

    print("J : {}".format(J_list))
    # plot_AUCs()
