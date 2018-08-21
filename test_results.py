from evaluation import *
import matplotlib
matplotlib.use("agg")

import matplotlib.pyplot as plt
import numpy as np

'''
-----------------------------------------------------------------------------------------------------
Se carga los directorios en donde se encuentran los datos y en donde se van a guardar los datos
generados por cada metrica obtenida, en el result dir deben estar los nombres de los centros usados,
o todos los sujetos en una carpeta
-----------------------------------------------------------------------------------------------------
'''

resultDir = '/media/fimcp/DATA/WMH/Results/'
dataDir = '/media/fimcp/DATA/WMH/Models/'
imagesDir = 'images/'
'''
-----------------------------------------------------------------------------------------------------
Se llama a la funcion main, la cual si se le pone el parametro verbose en True muestra por cada imagen
en el dataset los valores de los coeficientes obtenidos, si load es False, este vuelve a generar los 
coeficientes en lugar de cargarlos, tambien se debe espeficar que centro se usa es decir como un arre
glo de debe ir ['Utretch','Singapore','GE3T'] puede ser uno o todos estos generan en el datadir los
archivos por cada coeficiente, si se usa la bandera plot_ind se hace un grafico por cada uno de los 
centros o si se usa en plot_all se imprime solo el acumulado de todos los centros. 
-----------------------------------------------------------------------------------------------------
'''


def main_test(load=True, verbose=False, centers=None, activation_maps=False, plot_ind=False, plot_all=True):
    dsc = []
    h95 = []
    avd = []
    recall = []
    f1 = []

    '''
    -----------------------------------------------------------------------------------------------------
    Se llama a la funcion do, del script evaluation en el cual se obtienen los valores de los indices
    de dsc, h95, avd, recall y f1. 
    -----------------------------------------------------------------------------------------------------
    '''
    try:
        if centers is None:
            if not load:
                dirs = os.listdir(resultDir)
                for folder in dirs:
                    dsc_, h95_, avd_, recall_, f1_ = do(os.path.join(resultDir, folder),
                                                        os.path.join(resultDir, folder),
                                                        verbose)
                    dsc.append(dsc_)
                    h95.append(h95_)
                    avd.append(avd_)
                    recall.append(recall_)
                    f1.append(f1_)
        
                print 'Data Finished!'
                print "-" * 30
        
                dsc_n = np.array(dsc)
                h95_n = np.array(h95)
                avd_n = np.array(avd)
                recall_n = np.array(recall)
                f1_n = np.array(f1)
        
                np.save(dataDir + 'dsc_all.npy', dsc_n)
                np.save(dataDir + 'h95_all.npy', h95_n)
                np.save(dataDir + 'avd_all.npy', avd_n)
                np.save(dataDir + 'recall_all.npy', recall_n)
                np.save(dataDir + 'f1_all.npy', f1_n)
            else:
                dsc_n = np.load(dataDir + 'dsc_all.npy')
                h95_n = np.load(dataDir + 'h95_all.npy')
                avd_n = np.load(dataDir + 'avd_all.npy')
                recall_n = np.load(dataDir + 'recall_all.npy')
                f1_n = np.load(dataDir + 'f1_all.npy')
    
                plt.figure(1, figsize=(9, 6))
                plt.boxplot([dsc_n, h95_n, avd_n, recall_n, f1_n])
                plt.xticks([1, 2, 3, 4, 5], ['dsc', 'h95', 'avd', 'recall', 'f1'])
                plt.title(["Overall performance of the network"])
                plt.savefig(imagesDir + 'all.png')
        else:    
            if not load:
                if len(centers) > 3:
                    print "Mas centros de los originales"
                    return
                else:
                    for center in centers:
                        if "Utretch" == center:
                            dirs = os.listdir(resultDir, "Utretch")
                            for folder in dirs:
                                dsc_, h95_, avd_, recall_, f1_ = do(os.path.join(resultDir, folder),
                                                                    os.path.join(resultDir, folder),
                                                                    verbose)
                                dsc.append(dsc_)
                                h95.append(h95_)
                                avd.append(avd_)
                                recall.append(recall_)
                                f1.append(f1_)
        
                            print 'Data Finished!'
                            print "-"*30
        
                            dsc_n = np.array(dsc)
                            h95_n = np.array(h95)
                            avd_n = np.array(avd)
                            recall_n = np.array(recall)
                            f1_n = np.array(f1)
        
                            np.save(dataDir + 'dsc_ut.npy', dsc_n)
                            np.save(dataDir + 'h95_ut.npy', h95_n)
                            np.save(dataDir + 'avd_ut.npy', avd_n)
                            np.save(dataDir + 'recall_ut.npy', recall_n)
                            np.save(dataDir + 'f1_ut.npy', f1_n)
                        elif "Singapore" == center:
                            dirs = os.listdir(resultDir, "Singapore")
                            for folder in dirs:
                                dsc_, h95_, avd_, recall_, f1_ = do(os.path.join(resultDir, folder),
                                                                    os.path.join(resultDir, folder),
                                                                    verbose)
                                dsc.append(dsc_)
                                h95.append(h95_)
                                avd.append(avd_)
                                recall.append(recall_)
                                f1.append(f1_)
        
                            print 'Data Finished!'
                            print "-"*30
        
                            dsc_n = np.array(dsc)
                            h95_n = np.array(h95)
                            avd_n = np.array(avd)
                            recall_n = np.array(recall)
                            f1_n = np.array(f1)
        
                            np.save(dataDir + 'dsc_si.npy', dsc_n)
                            np.save(dataDir + 'h95_si.npy', h95_n)
                            np.save(dataDir + 'avd_si.npy', avd_n)
                            np.save(dataDir + 'recall_si.npy', recall_n)
                            np.save(dataDir + 'f1_si.npy', f1_n)
                        elif "GE3T" == center:
                            dirs = os.listdir(resultDir, "GE3T")
                            for folder in dirs:
                                dsc_, h95_, avd_, recall_, f1_ = do(os.path.join(resultDir, folder),
                                                                    os.path.join(resultDir, folder),
                                                                    verbose)
                                dsc.append(dsc_)
                                h95.append(h95_)
                                avd.append(avd_)
                                recall.append(recall_)
                                f1.append(f1_)
        
                            print 'Data Finished!'
                            print "-"*30
        
                            dsc_n = np.array(dsc)
                            h95_n = np.array(h95)
                            avd_n = np.array(avd)
                            recall_n = np.array(recall)
                            f1_n = np.array(f1)
        
                            np.save(dataDir + 'dsc_ge3t.npy', dsc_n)
                            np.save(dataDir + 'h95_ge3t.npy', h95_n)
                            np.save(dataDir + 'avd_ge3t.npy', avd_n)
                            np.save(dataDir + 'recall_ge3t.npy', recall_n)
                            np.save(dataDir + 'f1_ge3t.npy', f1_n)
            else:
                if len(centers) > 3:
                    print "Mas centros de los originales"
                    return
                else:
                    if len(centers) == 3:

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

                        if plot_all:
                            plt.figure()
                            plt.boxplot([dsc_ut, dsc_ge3t, dsc_si])
                            plt.xticks([1, 2, 3], ['Utrecht', 'GE3T', 'Singapore'])
                            plt.title("Coeficiente de similitud DICE")
                            plt.savefig(imagesDir + 'dsc_3.png')

                            plt.figure()
                            plt.boxplot([h95_ut, h95_ge3t, h95_si])
                            plt.xticks([1, 2, 3], ['Utrecht', 'GE3T', 'Singapore'])
                            plt.title("Distancia de Hausdorff (95p) (mm)")
                            plt.savefig(imagesDir + 'h95_3.png')

                            plt.figure()
                            plt.boxplot([avd_ut, avd_ge3t, avd_si])
                            plt.xticks([1, 2, 3], ['Utrecht', 'GE3T', 'Singapore'])
                            plt.title("Diferencia promedio de volumen (%)")
                            plt.savefig(imagesDir + 'avd_3.png')

                            plt.figure()
                            plt.boxplot([recall_ut, recall_ge3t, recall_si])
                            plt.xticks([1, 2, 3], ['Utrecht', 'GE3T', 'Singapore'])
                            plt.title("Lesion Recall")
                            plt.savefig(imagesDir + 'recall_3.png')

                            plt.figure()
                            plt.boxplot([f1_ut, f1_ge3t, f1_si])
                            plt.xticks([1, 2, 3], ['Utrecht', 'GE3T', 'Singapore'])
                            plt.title("Lesion F1")
                            plt.savefig(imagesDir + 'f1_3.png')
                    for center in centers:
                        if "Utretch" == center:
                            dsc_ut = np.load(dataDir + 'dsc_ut.npy')
                            h95_ut = np.load(dataDir + 'h95_ut.npy')
                            avd_ut = np.load(dataDir + 'avd_ut.npy')
                            recall_ut = np.load(dataDir + 'recall_ut.npy')
                            f1_ut = np.load(dataDir + 'f1_ut.npy')
        
                            if plot_ind:
                                plt.figure()
                                plt.boxplot([dsc_ut])
                                plt.xticks([1], ['Utrecht'])
                                plt.title("Coeficiente de similitud DICE")
                                plt.savefig(imagesDir + 'dsc_ut.png')
    
                                plt.figure()
                                plt.boxplot([h95_ut])
                                plt.xticks([1], ['Utrecht'])
                                plt.title("Distancia de Hausdorff (95p) (mm)")
                                plt.savefig(imagesDir + 'h95_ut.png')
    
                                plt.figure()
                                plt.boxplot([avd_ut])
                                plt.xticks([1], ['Utrecht'])
                                plt.title("Diferencia promedio de volumen (%)")
                                plt.savefig(imagesDir + 'avd_ut.png')
    
                                plt.figure()
                                plt.boxplot([recall_ut])
                                plt.xticks([1], ['Utrecht'])
                                plt.title("Lesion Recall")
                                plt.savefig(imagesDir + 'recall_ut.png')
    
                                plt.figure()
                                plt.boxplot([f1_ut])
                                plt.xticks([1], ['Utrecht'])
                                plt.title("Lesion F1")
                                plt.savefig(imagesDir + 'f1_ut.png')
                        elif "Singapore" == center:
                            dsc_si = np.load(dataDir + 'dsc_si.npy')
                            h95_si = np.load(dataDir + 'h95_si.npy')
                            avd_si = np.load(dataDir + 'avd_si.npy')
                            recall_si = np.load(dataDir + 'recall_si.npy')
                            f1_si = np.load(dataDir + 'f1_si.npy')
    
                            if plot_ind:
                                plt.figure()
                                plt.boxplot([dsc_si])
                                plt.xticks([1], ['Singapore'])
                                plt.title("Coeficiente de similitud DICE")
                                plt.savefig(imagesDir + 'dsc_si.png')
    
                                plt.figure()
                                plt.boxplot([h95_si])
                                plt.xticks([1], ['Singapore'])
                                plt.title("Distancia de Hausdorff (95p) (mm)")
                                plt.savefig(imagesDir + 'h95_si.png')
    
                                plt.figure()
                                plt.boxplot([avd_si])
                                plt.xticks([1], ['Singapore'])
                                plt.title("Diferencia promedio de volumen (%)")
                                plt.savefig(imagesDir + 'avd_si.png')
    
                                plt.figure()
                                plt.boxplot([recall_si])
                                plt.xticks([1], ['Singapore'])
                                plt.title("Lesion Recall")
                                plt.savefig(imagesDir + 'recall_si.png')
    
                                plt.figure()
                                plt.boxplot([f1_si])
                                plt.xticks([1], ['Singapore'])
                                plt.title("Lesion F1")
                                plt.savefig(imagesDir + 'f1_si.png')
                        elif "GE3T" == center:
                            dsc_ge3t = np.load(dataDir + 'dsc_ge3t.npy')
                            h95_ge3t = np.load(dataDir + 'h95_ge3t.npy')
                            avd_ge3t = np.load(dataDir + 'avd_ge3t.npy')
                            recall_ge3t = np.load(dataDir + 'recall_ge3t.npy')
                            f1_ge3t = np.load(dataDir + 'f1_ge3t.npy')
                            if plot_ind:
                                plt.figure()
                                plt.boxplot([dsc_ge3t])
                                plt.xticks([1], ['GE3T'])
                                plt.title("Coeficiente de similitud DICE")
                                plt.savefig(imagesDir + 'dsc_ge3t.png')
    
                                plt.figure()
                                plt.boxplot([h95_ge3t])
                                plt.xticks([1], ['GE3T'])
                                plt.title("Distancia de Hausdorff (95p) (mm)")
                                plt.savefig(imagesDir + 'h95_ge3t.png')
    
                                plt.figure()
                                plt.boxplot([avd_ge3t])
                                plt.xticks([1], ['GE3T'])
                                plt.title("Diferencia promedio de volumen (%)")
                                plt.savefig(imagesDir + 'avd_ge3t.png')
    
                                plt.figure()
                                plt.boxplot([recall_ge3t])
                                plt.xticks([1], ['GE3T'])
                                plt.title("Lesion Recall")
                                plt.savefig(imagesDir + 'recall_ge3t.png')
    
                                plt.figure()
                                plt.boxplot([f1_ge3t])
                                plt.xticks([1], ['GE3T'])
                                plt.title("Lesion F1")
                                plt.savefig(imagesDir + 'f1_ge3t.png')
    except Exception, e:
        print e.message
        
    if activation_maps:
        activation_maps = np.load(dataDir + 'activation_maps_down.npy')
        columns = len(activation_maps)
        rows = 1
        fig = plt.figure(figsize=(columns, rows))
        for i in range(1, columns * rows + 1):
            fig.add_subplot(rows, columns, i)
            plt.imshow(activation_maps[i-1])
        plt.title("Deformacion de las imagen a lo largo del modelo de aprendizaje")
        plt.savefig(imagesDir + 'activation_maps.png')

    plt.show()


if "__main__" == __name__:
    main_test(centers=['Utretch', 'Singapore', 'GE3T'], plot_ind=True)
