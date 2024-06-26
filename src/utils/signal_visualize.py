import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from utils.data import build_dataset_pickle as test_dataset
from utils.data import get_non_seizure_signal, get_epilepsiae_test
from utils.params import pat_list
from scipy.stats import wilcoxon


def load_model(test_patient, model_name):
    arch = 'vae_free'
    subdirname = "../../temp/vae_mmd/integrated/{}/{}/{}".format(1024, arch, model_name)
    save_path = '{}/model/test_{}/saved_model/'.format(subdirname, test_patient)
    trained_model = tf.keras.models.load_model(save_path)
    return trained_model


def predict_(test_patient, model_proposed, model_baseline):
    sessions = test_dataset(test_patient, root='../../')
    for node in sessions.keys():
        STATE_LEN = 899

        X = sessions[node]['data']
        y_true = sessions[node]['label']
        print(y_true.shape)
        minutes = (y_true.shape[0] + 1) // 15

        if np.sum(y_true) == 0:
            continue

        # X_section = np.expand_dims(X, 0)
        # X_edge = get_non_seizure_signal(test_patient, state_len=STATE_LEN, root='../..')
        # X_section = np.concatenate((X_edge, X_section, X_edge), axis=1)
        #
        # out = model_proposed.predict(X_section)[0, STATE_LEN:STATE_LEN + minutes * 15, :]
        # print(np.where(out > 0.5))
        # out_baseline = model_baseline.predict(X_section)[0, STATE_LEN:STATE_LEN + minutes * 15, :]
        # print(np.where(out_baseline > 0.5))
        # t_out = np.linspace(0, minutes, minutes * 15)

        X_20min = np.reshape(X, newshape=(-1, 2))[:256 * minutes * 60]
        t = np.linspace(0, minutes, X_20min.shape[0])

        y_non_zero = np.where(y_true > 0, 1, 0)
        y_diff = np.diff(y_non_zero)
        start_points = np.where(y_diff > 0)[0]
        stop_points = np.where(y_diff < 0)[0]

        plt.figure(figsize=(15, 6))
        plt.plot(t, 0.35 * (X_20min[:, 0]) + 10, linewidth=0.3, c='dimgray')
        plt.plot(t, 0.35 * (X_20min[:, 1]) + 20, linewidth=0.3, c='dimgray')
        for seizure_start, seizure_stop in zip(start_points, stop_points):
            plt.axvspan(t[seizure_start * 1024], t[seizure_stop * 1024], color='r', alpha=0.5)
        # plt.plot(t_out, out * 10 + 34, c='k', marker='o', markersize=1.2)
        # plt.plot(t_out, out_baseline * 10 + 50, c='k', marker='o', markersize=1.2)

        plt.grid(b=True, c='r', which='major', lw=0.5, axis='x')
        plt.grid(b=True, c='r', which='major', lw=0.2, axis='y')
        plt.grid(b=True, c='r', which='minor', lw=0.2)
        plt.xticks(ticks=np.arange(0, minutes + 1, step=5),
                   labels=["{}:00".format(str(i)) for i in np.arange(0, minutes + 1, step=5)], fontsize=12)
        plt.xlim([0, minutes])
        plt.yticks(ticks=[10, 20, 34, 38, 44, 50, 54, 60],
                   labels=['F8-T8', 'F7-T7', '0', 'Proposed', '1', '0', 'Baseline', '1'], fontsize=14)
        plt.ylim([0, 30])
        plt.minorticks_on()
        plt.xlabel('Time (min)', fontsize=14)

        # plt.show()
        plt.savefig("../../output/signals/{}".format(node))


proposed = np.dot(
    [0, 0, 0, 0, 0, 0, 0, 94, 11, 0, 0, 0, 0, 0, 0, 0, 0, 113, 0, 0, 0, 0, 0, 0, 1590, 380, 3156, 1024, 659, 340, 927,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2752, 3, 0, 0, 0, 0, 0, 2, 0, 3, 0, 71, 0, 0, 1, 153, 0, 6, 243, 1, 28, 0, 0, 1, 25,
     20, 634, 161, 456, 618, 0, 173, 593, 238, 297, 0, 630, 67, 77, 113, 2, 0, 254, 69, 0, 0, 262, 0, 594, 5, 0, 325, 0,
     38, 219, 579, 300, 474, 296, 0, 0, 219, 0, 495, 667, 223, 0, 181, 0, 0, 0, 505, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 246,
     0, 0], 4)

proposed_129 = np.dot(
    [0, 0, 0, 0, 0, 0, 0, 94, 60, 11, 0, 0, 0, 0, 0, 0, 0, 0, 113, 0, 0, 0, 0, 0, 0, 1590, 380, 3156, 1024, 659, 340, 927,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2752, 3, 0, 0, 0, 0, 0, 2, 0, 3, 0, 71, 0, 0, 1, 153, 0, 6, 243, 1, 28, 0, 0, 1, 25,
     20, 634, 161, 456, 618, 0, 173, 593, 238, 297, 0, 630, 67, 77, 113, 2, 0, 254, 69, 0, 0, 262, 0, 594, 5, 0, 325, 0,
     38, 219, 579, 300, 474, 296, 0, 0, 0, 0, 219, 0, 495, 667, 223, 0, 181, 0, 0, 0, 505, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 246,
     0, 0], 4)
baseline = np.dot([0, 0, 0, 0, 0, 0, 0, 89, 38, 0, 0, 348, 0, 0, 0, 0, 1943, 106, 0, 0, 0, 0, 0, 0, 1659, 421, 680,
                   292, 101, 2927, 237, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2709, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 337, 0, 0,
                   0, 783, 0, 25, 243, 1, 50, 0, 426, 0, 140, 27, 634, 102, 0, 737, 0, 441, 538, 185, 105, 134, 211,
                   212, 18, 63, 249, 33, 427, 192, 0, 30, 316, 240, 193, 66, 80, 267, 0, 20, 28, 149, 303, 6, 137,
                   0,
                   0, 222, 57, 497, 129, 462, 0, 0, 0, 0, 0, 0, 0, 0, 0, 240, 0, 0, 48, 0, 0, 0, 258, 0, 0], 4)
manual = np.dot([0, 0, 0, 0, 0, 0, 0, 86, 14, 11, 17, 339, 578, 9, 5, 0, 346, 104, 0, 0, 0, 0, 35, 0, 54, 419, 3105,
                 362, 653, 824, 1903, 46, 0, 0, 0, 0, 0, 0, 0, 0, 51, 2724, 366, 791, 460, 0, 184, 673, 1230, 0, 5,
                 647, 452, 6, 85, 139, 739, 19, 7, 6, 1, 93, 680, 184, 86, 178, 29, 90, 9, 0, 675, 0, 442, 565, 129,
                 247, 6, 543, 211, 615, 124, 324, 113, 137, 70, 131, 30, 285, 171, 95, 90, 423, 205, 150, 73, 29,
                 101,
                 463, 501, 185, 0, 0, 222, 13, 11, 129, 596, 0, 0, 0, 63, 460, 0, 7, 220, 0, 87, 280, 135, 459, 0,
                 0,
                 243, 254, 0, 0], 4)

eglass = np.dot(
    [0, 0, 0, 0, 0, 0, 3, 0, 87, 13, 0, 0, 0, 0, 0, 0, 0, 306, 107, 0, 0, 0, 0, 0, 0, 61, 405, 2631, 1032, 94, 14, 185,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2927, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 436, 432, 0, 0, 160, 0, 97, 0, 49, 50, 0, 3, 0,
     140, 26, 32, 102, 282, 619, 0, 110, 225, 185, 280, 269, 211, 243, 55, 149, 0, 2, 0, 23, 2, 95, 314, 0, 193, 0, 0,
     405, 0, 20, 183, 43, 303, 8, 183, 0, 0, 184, 7, 224, 0, 230, 9, 460, 0, 0, 0, 0, 0, 0, 0, 0, 0, 151, 0, 0, 463, 0,
     0, 0, 252, 0, 0], 4)
eglass_epilepsiae = np.dot(
    [3, 314, 166, 573, 52, 12, 40, 0, 238, 45, 0, 39, 0, 227, 107, 42, 196, 169, 143, 1, 0, 55, 265, 640, 59, 70,
     0, 0, 0, 0, 658, 40, 0, 209, 413, 107, 338, 670, 217, 156, 2, 195, 132, 307, 181, 0, 141, 158, 275, 402,
     594, 352, 0, 80, 58, 214, 0, 7, 25, 0, 298, 3, 13, 1, 624, 0, 543, 0, 509, 511, 6, 339, 8, 124, 254, 142,
     344, 22, 552, 97, 0, 0, 0, 9, 296, 32, 0, 0, 0, 0, 0, 303, 454, 102, 14, 23, 31, 56, 38, 84, 241, 523, 656,
     250, 0, 225, 0, 69, 279, 0, 0, 137, 341, 416, 0, 406, 298, 158, 0, 147, 111, 0, 0, 0, 0, 24, 0, 0, 450, 306,
     0, 0, 0, 532, 148, 279, 544, 29, 25, 0, 61, 14, 0, 98, 208, 0, 4, 600, 0, 9, 0, 0, 242, 128, 241, 133, 0, 2,
     136, 0, 0, 0, 78, 0, 0, 647, 463, 0, 801, 508, 6, 0, 345, 468, 14, 0, 449, 0, 157, 46, 379, 174, 274, 85,
     371, 0, 482, 226, 140, 0, 0, 375, 0, 0, 0, 0, 107, 43, 332, 593, 0, 505, 268, 105, 142, 0, 0, 466, 0, 0, 0,
     0, 466, 270, 93, 0, 27, 428, 256, 84, 140, 164, 363, 95, 540, 782, 0, 3, 395, 266, 436, 17, 0, 0, 10, 7, 0,
     107, 0, 0, 0, 0, 222, 61, 773, 207, 0, 0, 398, 0, 456, 201, 33, 0, 639, 40, 0, 154, 0, 389, 0, 0], 4)

proposed_epilepsiae = np.dot(
    [10, 314, 166, 0, 0, 2, 574, 2, 12, 138, 0, 0, 0, 0, 177, 9, 195, 289, 0, 3, 0, 0, 264, 10, 151, 0, 0, 0, 0, 0,
     0, 0, 0, 302, 413, 71, 103, 52, 128, 392, 62, 203, 233, 254, 5, 108, 77, 52, 107, 401, 647, 84, 11, 209, 163,
     208, 18, 102, 23, 42, 106, 4, 0, 7, 625, 0, 220, 0, 508, 0, 283, 115, 149, 124, 827, 22, 329, 438, 552, 206, 0,
     0, 0, 233, 6, 9, 0, 0, 0, 0, 0, 0, 397, 161, 76, 17, 34, 61, 39, 24, 786, 2, 38, 250, 0, 82, 0, 0, 303, 0, 36,
     334, 341, 445, 0, 292, 9, 158, 0, 509, 312, 0, 0, 0, 0, 12, 0, 0, 78, 4, 0, 0, 433, 202, 148, 330, 544, 328,
     319, 0, 60, 18, 0, 0, 0, 0, 23, 148, 0, 0, 0, 0, 0, 0, 246, 282, 0, 0, 2, 0, 0, 0, 0, 0, 0, 556, 463, 0, 0,
     508, 171, 0, 345, 0, 0, 0, 505, 0, 611, 8, 351, 156, 88, 100, 239, 141, 7, 189, 638, 0, 0, 125, 0, 0, 0, 0,
     440, 235, 295, 0, 0, 31, 0, 0, 284, 0, 0, 465, 0, 0, 0, 0, 57, 277, 217, 0, 220, 243, 256, 423, 397, 0, 0, 120,
     103, 526, 234, 0, 199, 489, 432, 17, 0, 0, 136, 5, 0, 0, 0, 135, 0, 0, 201, 850, 726, 0, 0, 0, 513, 0, 0, 89,
     0, 0, 0, 0, 0, 0, 0, 601, 0, 0], 4)
baseline_epilepsiae = np.dot(
    [11, 313, 0, 515, 450, 9, 9, 13, 235, 8, 176, 145, 37, 138, 170, 222, 196, 111, 55, 265, 0, 19, 264, 717, 206,
     0, 52, 730, 71, 771, 658, 0, 525, 209, 20, 85, 249, 132, 258, 117, 3, 198, 132, 9, 10, 78, 123, 32, 266, 402,
     265, 100, 8, 56, 202, 337, 0, 211, 23, 41, 104, 3, 200, 6, 624, 0, 537, 482, 208, 502, 348, 84, 33, 124, 822,
     146, 336, 429, 532, 148, 0, 217, 0, 1, 41, 0, 71, 0, 78, 53, 0, 76, 138, 162, 76, 590, 24, 54, 38, 148, 3, 2,
     304, 250, 491, 324, 0, 69, 315, 0, 0, 133, 244, 443, 198, 482, 0, 0, 472, 631, 312, 30, 0, 0, 0, 24, 0, 480,
     666, 287, 10, 0, 664, 326, 155, 224, 379, 45, 491, 26, 62, 18, 0, 99, 91, 0, 35, 703, 203, 9, 3, 162, 244, 127,
     241, 411, 43, 477, 138, 302, 13, 0, 89, 397, 0, 221, 290, 686, 0, 506, 32, 0, 734, 25, 52, 557, 535, 363, 209,
     7, 383, 237, 605, 86, 227, 0, 21, 24, 546, 0, 88, 362, 680, 324, 132, 0, 100, 427, 332, 777, 227, 30, 453, 0,
     142, 6, 46, 465, 0, 0, 352, 36, 562, 285, 110, 0, 96, 255, 292, 22, 401, 109, 0, 112, 504, 516, 121, 135, 173,
     384, 435, 17, 453, 596, 0, 8, 378, 347, 8, 19, 191, 0, 221, 61, 725, 208, 0, 660, 553, 0, 7, 84, 86, 0, 544,
     26, 0, 124, 421, 8, 599, 93], 4)

manual_epilepsiae = np.dot(
    [19, 310, 78, 2, 12, 107, 59, 5, 231, 135, 166, 305, 524, 7, 167, 3, 199, 255, 55, 260, 0, 15, 256, 721, 151, 0,
     57, 0, 80, 0, 661, 17, 456, 207, 417, 395, 98, 125, 255, 160, 31, 198, 136, 22, 8, 44, 408, 152, 179, 398, 262,
     182, 0, 55, 173, 333, 0, 307, 21, 42, 99, 0, 14, 1, 207, 348, 535, 486, 522, 516, 70, 303, 133, 113, 103, 112,
     569, 434, 549, 256, 43, 492, 14, 8, 0, 0, 79, 0, 77, 22, 0, 299, 451, 159, 16, 270, 29, 63, 20, 88, 7, 228,
     326, 251, 493, 585, 0, 563, 44, 25, 2, 136, 345, 402, 164, 1, 5, 162, 242, 509, 115, 35, 200, 14, 236, 21, 277,
     455, 312, 448, 800, 296, 668, 111, 0, 322, 542, 361, 21, 152, 172, 29, 202, 219, 90, 17, 11, 51, 79, 0, 693,
     264, 335, 72, 791, 410, 578, 481, 2, 298, 237, 0, 86, 184, 227, 224, 280, 681, 791, 502, 36, 242, 27, 30, 12,
     560, 450, 370, 169, 383, 340, 471, 276, 232, 0, 19, 222, 572, 189, 171, 364, 89, 100, 309, 11, 111, 36, 110,
     752, 0, 28, 450, 124, 143, 4, 41, 543, 568, 14, 0, 45, 462, 662, 107, 191, 19, 204, 292, 17, 250, 104, 245, 7,
     765, 676, 266, 219, 173, 380, 20, 15, 0, 238, 148, 154, 9, 250, 0, 43, 188, 0, 335, 84, 771, 208, 402, 422,
     541, 477, 269, 233, 5, 40, 548, 23, 107, 178, 225, 419, 785, 0, 0], 4)

FCN = np.dot(
    [0, 0, 0, 0, 0, 0, 0, 91, 13, 0, 0, 0, 0, 0, 0, 0, 1943, 1889, 0, 0, 0, 0, 0, 0, 49, 2299, 121, 2332, 418, 3115,
     1939, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2709, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 0, 161, 0, 80, 117, 47, 50, 0,
     449, 0, 140, 25, 634, 47, 180, 737, 0, 603, 589, 185, 24, 336, 343, 138, 512, 46, 133, 541, 398, 243, 0, 30, 227,
     0, 0, 0, 0, 387, 242, 0, 0, 580, 111, 6, 219, 0, 0, 232, 0, 235, 70, 466, 0, 0, 0, 0, 0, 0, 0, 0, 0, 118, 0, 0,
     461, 0, 0, 0, 456, 64, 483]
    , 4)
FCN_epilepsiae = np.dot(
    [24, 313, 81, 589, 0, 4, 587, 13, 0, 137, 0, 0, 36, 138, 170, 195, 294, 446, 0, 157, 0, 51, 264, 717, 165, 0, 0, 0,
     130, 0, 0, 0, 10, 411, 358, 292, 330, 636, 258, 117, 62, 199, 119, 9, 181, 0, 106, 131, 348, 406, 631, 373, 8, 81,
     202, 337, 0, 331, 22, 41, 0, 19, 735, 6, 624, 0, 538, 0, 0, 502, 228, 63, 33, 133, 822, 135, 344, 438, 468, 94, 0,
     0, 0, 7, 0, 638, 0, 0, 708, 0, 0, 0, 447, 0, 76, 590, 20, 64, 23, 89, 3, 2, 304, 142, 491, 588, 0, 636, 282, 0, 14,
     462, 18, 443, 0, 427, 0, 15, 0, 548, 312, 0, 0, 0, 0, 547, 0, 0, 334, 223, 12, 0, 664, 112, 148, 224, 682, 336,
     147, 26, 61, 185, 0, 0, 0, 0, 16, 669, 0, 0, 0, 0, 0, 0, 52, 0, 2, 70, 5, 302, 0, 0, 0, 0, 0, 742, 510, 0, 0, 506,
     698, 0, 98, 5, 14, 55, 382, 363, 598, 7, 0, 280, 605, 8, 240, 106, 21, 24, 545, 0, 0, 101, 681, 0, 0, 0, 212, 39,
     114, 0, 28, 30, 268, 0, 35, 6, 0, 516, 0, 0, 0, 0, 305, 285, 695, 0, 27, 198, 0, 332, 401, 109, 0, 0, 504, 29, 121,
     3, 199, 265, 436, 17, 419, 0, 0, 9, 253, 347, 0, 0, 89, 0, 222, 109, 724, 597, 0, 0, 0, 0, 0, 84, 0, 0, 110, 0, 0,
     527, 0, 305, 0, 0]
    , 4)


def plot_box():
    # fig, ax = plt.subplots(figsize=(8, 6))
    # colors = ['pink', 'lightblue', 'lightgreen']
    # bplot = ax.boxplot([proposed, baseline, manual], notch=False, whis=[5, 95], labels=['Proposed', 'C-CNN', 'Manual MMD'])
    # for patch, color in zip(bplot['boxes'], colors):
    #     patch.set_facecolor(color)

    # plt.boxplot([proposed_epilepsiae, baseline_epilepsiae, manual_epilepsiae], whis=[5, 95])
    # ax.set_xticks(ticks=[1,2,3], fontsize=16)
    # plt.ylim([-20, 3600])
    # plt.yticks(ticks=np.arange(0, 3601, step=600), labels=[" {}".format(str(i)) for i in np.arange(0, 61, step=10)],
    #            fontsize=14)
    # plt.ylabel('Time (min)', fontsize=16)

    boxprops = dict(linewidth=3, color='black')
    capprops = dict(linewidth=3)
    flierprops = dict(marker='x', markerfacecolor='red', markersize=8,
                      markeredgecolor='red')
    medianprops = dict(linewidth=3.5, color='firebrick')

    all_data = [proposed_epilepsiae, baseline_epilepsiae, manual_epilepsiae, eglass_epilepsiae, FCN_epilepsiae]
    # all_data = [proposed, baseline, manual, eglass, FCN]
    labels = ['Proposed', 'B-VIB', 'B-MMD', 'B-FET', 'B-FCN']

    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

    # rectangular box plot
    bplot1 = ax1.boxplot(all_data,
                         whis=[5, 95],
                         patch_artist=True,  # fill with color
                         boxprops=boxprops, medianprops=medianprops, capprops=capprops, flierprops=flierprops,
                         whiskerprops=capprops)  # will be used to label x-ticks
    # ax1.set_title('Rectangular box plot')
    ax1.set_ylim([-20, 3600])
    ax1.set_ylabel('Time (min)', fontsize=16)
    ax1.set_yticks(ticks=np.arange(0, 3601, step=600))
    ax1.set_yticklabels(labels=[" {}".format(str(i)) for i in np.arange(0, 61, step=10)],
                        fontsize=14)
    ax1.set_xticklabels(labels=labels, fontsize=14)
    plt.grid(axis='y')

    # notch shape box plot
    # bplot2 = ax1.boxplot(all_data,
    #                      notch=True,  # notch shape
    #                      vert=True,  # vertical box alignment
    #                      patch_artist=True,  # fill with color
    #                      labels=labels)  # will be used to label x-ticks
    # ax1.set_title('Notched box plot')

    # fill with colors
    colors = ['pink', 'lightblue', 'lightgreen', 'navajowhite', 'b']
    for bplot in (bplot1, bplot1):
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
    # plt.show()
    plt.savefig('../../output/images/Unseen.pdf', format='pdf')


def time_table():
    for time_expected in [0, 15, 30, 60, 150, 300]:
        for result, name in zip([proposed_epilepsiae, baseline_epilepsiae, manual_epilepsiae],
                                ["proposed", "CCNN", "MMD"]):
            cnt = np.count_nonzero(result <= time_expected)
            print("{} Time: {} = {},{:.2f}%".format(name, time_expected, cnt, cnt / 262.))


def statistics():
    print(proposed_epilepsiae.shape, manual_epilepsiae.shape)
    _, pvalue_loocv = wilcoxon(proposed, y=manual, alternative='less')
    _, pvalue_unseen = wilcoxon(proposed_epilepsiae, y=manual_epilepsiae, alternative='less')
    print(pvalue_loocv, pvalue_unseen)


if __name__ == '__main__':
    tf.config.experimental.set_visible_devices([], 'GPU')
    # test_pat = 3
    # model_proposed = load_model(test_patient=test_pat, model_name='z_minus1_v52')
    # model_baseline = load_model(test_patient=test_pat, model_name='Anthony_v53')
    # for test_pat in range(1, 24):
    #     predict_(test_patient=test_pat, model_proposed=model_proposed, model_baseline=model_baseline)
    # plot_box()
    # time_table()
    statistics()
