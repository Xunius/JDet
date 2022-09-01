import os
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from jdet.models.boxes.box_ops import poly_to_rotated_box_np

FOLDER = '/root/autodl-tmp/JDet/work_dirs/s2anet_r50_fpn_1x_fair1m_1_5'
CLASS = 'Ship'

def get_data(filename):

    with open(filename, 'rb') as fin:
        data = pickle.load(fin)

    print('load data from', filename)
    print('data.shape', data.shape)

    poly = data[:, 1:9]
    bbox = poly_to_rotated_box_np(poly)

    xc, yc, w, h, ang = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3], bbox[:, 4]
    ang = (ang / np.pi * 180) % 180
    num = len(data)
    ratio = h/w
    res = {'num': num,
           'xc': xc,
           'yc': yc,
           'w': w,
           'h': h,
           'ang': ang,
           'ratio': ratio,
           'score': data[:,-1],
           }

    return res

if __name__ == '__main__':

    tp_file = os.path.join(FOLDER, 'tp_detections_%s' %CLASS, 'tp_dets.dat')
    fp_file = os.path.join(FOLDER, 'fp_detections_%s' %CLASS, 'fp_dets.dat')

    tp_data = get_data(tp_file)
    fp_data = get_data(fp_file)


    figure, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 6))
    titles = ['num', 'xc', 'yc', 'w', 'h', 'ang', 'ratio', 'score']

    for ii, titleii in enumerate(titles):

        ax = axes.flatten()[ii]
        tpii = tp_data[titleii]
        fpii = fp_data[titleii]

        if titleii == 'num':
            ax.bar(0, tpii, width=0.5)
            ax.bar(1, fpii, width=0.5)
        else:
            ax.boxplot(tpii, sym='.', positions=[0,], widths=0.5,
                    showmeans=True, showfliers=True)
            ax.boxplot(fpii, sym='.', positions=[1,], widths=0.5,
                    showmeans=True, showfliers=True)
            '''
            ax.violinplot(tpii, positions=[0,], widths=0.5,
                    showmeans=True, showextrema=True)
            ax.violinplot(fpii, positions=[1,], widths=0.5,
                    showmeans=True, showextrema=True)
            '''

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['TP', 'FP'])
        ax.grid(True, axis='y')

    figure.tight_layout()
    outfilename = os.path.join(FOLDER, 'tp_fp_comp_%s.png' %CLASS)
    print('Save plot to', outfilename)
    figure.savefig(outfilename, dpi=100, bbox_inches='tight')








