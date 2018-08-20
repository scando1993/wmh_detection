from evaluation import *
import matplotlib
matplotlib.use("agg")

import matplotlib.pyplot as plt
import numpy as np

resultDir = '/media/fimcp/DATA/WMH/Results/Singapore'
dataDir = '/media/fimcp/DATA/WMH/Models/'


def main():
    dsc = []
    h95 = []
    avd = []
    recall = []
    f1 = []

    # dirs = os.listdir(resultDir)
    # for folder in dirs:
    #     dsc_, h95_, avd_, recall_, f1_ = do(os.path.join(resultDir,folder), os.path.join(resultDir,folder), True)
    #     dsc.append(dsc_)
    #     h95.append(h95_)
    #     avd.append(avd_)
    #     recall.append(recall_)
    #     f1.append(f1_)
    #
    # print 'Data Finished!'
    # # print "-"*30
    #
    # dsc_n = np.array(dsc)
    # h95_n = np.array(h95)
    # avd_n = np.array(avd)
    # recall_n = np.array(recall)
    # f1_n = np.array(f1)

    dsc_ut = np.load(dataDir + 'dsc_ut.npy')
    h95_ut = np.load(dataDir + 'h95_ut.npy')
    avd_ut = np.load(dataDir + 'avd_ut.npy')
    recall_ut = np.load(dataDir + 'recall_ut.npy')
    f1_ut = np.load(dataDir + 'f1_ut.npy')

    dsc_ge3t = np.load(dataDir + 'dsc_ge3t.npy')
    h95_ge3t = np.load(dataDir + 'h95_ge3t.npy')
    avd_ge3t = np.load(dataDir + 'avd_ge3t.npy')
    recall_ge3t = np.load(dataDir + 'recall_ge3t.npy')
    f1_ge3t = np.load(dataDir + 'f1_ge3t.npy')

    dsc_si = np.load(dataDir + 'dsc_si.npy')
    h95_si = np.load(dataDir + 'h95_si.npy')
    avd_si = np.load(dataDir + 'avd_si.npy')
    recall_si = np.load(dataDir + 'recall_si.npy')
    f1_si = np.load(dataDir + 'f1_si.npy')

    # np.save(dataDir + 'dsc_si.npy', dsc_n)
    # np.save(dataDir + 'h95_si.npy', h95_n)
    # np.save(dataDir + 'avd_si.npy', avd_n)
    # np.save(dataDir + 'recall_si.npy', recall_n)
    # np.save(dataDir + 'f1_si.npy', f1_n)

    plt.figure()
    plt.boxplot([dsc_ut, dsc_ge3t, dsc_si])
    plt.xticks([1, 2, 3], ['Utrecht', 'GE3T', 'Singapore'])
    plt.title("Coeficiente de similitud DICE")
    plt.savefig('dsc_3.png')

    plt.figure()
    plt.boxplot([h95_ut, h95_ge3t, h95_si])
    plt.xticks([1, 2, 3], ['Utrecht', 'GE3T', 'Singapore'])
    plt.title("Distancia de Hausdorff (95p) (mm)")
    plt.savefig('h95_3.png')

    plt.figure()
    plt.boxplot([avd_ut, avd_ge3t, avd_si])
    plt.xticks([1, 2, 3], ['Utrecht', 'GE3T', 'Singapore'])
    plt.title("Diferencia promedio de volumen (%)")
    plt.savefig('avd_3.png')

    plt.figure()
    plt.boxplot([recall_ut, recall_ge3t, recall_si])
    plt.xticks([1, 2, 3], ['Utrecht', 'GE3T', 'Singapore'])
    plt.title("Lesion Recall")
    plt.savefig('recall_3.png')

    plt.figure()
    plt.boxplot([f1_ut, f1_ge3t, f1_si])
    plt.xticks([1, 2, 3], ['Utrecht', 'GE3T', 'Singapore'])
    plt.title("Lesion F1")
    plt.savefig('f1_3.png')

    # plt.figure(1, figsize=(9, 6))
    # plt.boxplot([dsc_n, h95_n, avd_n, recall_n, f1_n])
    # plt.xticks([1, 2, 3, 4, 5], ['dsc', 'h95', 'avd', 'recall', 'f1'])
    # plt.title(["Overall performance of the network"])
    # plt.savefig('all.png')

    plt.show()


if "__main__" == __name__:
    main()
