from jdet.models.boxes.box_ops import rotated_box_to_poly_single
from jdet.utils.registry import DATASETS
from jdet.config.constant import FAIR_CLASSES_, FAIR1M_1_5_CLASSES
from jdet.models.boxes.box_ops import rotated_box_to_poly_single
from jdet.data.dota import DOTADataset
import os
import numpy as np

@DATASETS.register_module()
class FAIRDataset(DOTADataset):
    CLASSES = FAIR_CLASSES_
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.CLASSES = FAIR_CLASSES_

        img_infos = []
        for img_info in self.img_infos:
            boxes = img_info["ann"]["bboxes"]
            w,h = boxes[:,2],boxes[:,3]
            area = w*h
            index = area>1.
            img_info["ann"]["bboxes"] = img_info["ann"]["bboxes"][index]
            img_info["ann"]["labels"] = img_info["ann"]["labels"][index]


    def _balance_categories(self):
        img_infos = self.img_infos
        cate_dict = {}
        for idx,img_info in enumerate(img_infos):
            unique_labels = np.unique(img_info["ann"]["labels"])
            for label in unique_labels:
                if label not in cate_dict:
                    cate_dict[label]=[]
                cate_dict[label].append(idx)

        # fair
        # labels: [ 5, 10, 27, 36, 26, 12, 19, 33, 29, 28, 22, 37, 34,  9, 31, 16,  3, 15,  8,  4,  2, 20,  7, 32, 17, 30,  1, 14,  6, 35, 13, 23, 18, 11, 24, 25, 21]
        # counts: [ 644, 733,   1051,   2173,   2199,   2413,   2511,   2984, 3178,   3398,   3890,   3980,   3997,   4274,   4838,   5940,  6064,   6364,   6371,   6403,   6498,   9033,  10413,  11187, 11515,  11554,  16014,  23120,  24386,  24875,  30084,  35516, 37878,  40175,  97638, 503094, 544446]
        # category: []
        # ['C919',
        # 'ARJ21',
        # 'Tractor',
        # 'Roundabout',
        # 'Trailer',
        # 'Passenger Ship',
        # 'Warship',
        # 'Football Field',
        # 'Truck Tractor',
        # 'Excavator',
        # 'Bus',
        # 'Bridge',
        # 'Baseball Field',
        # 'A350',
        # 'Basketball Court',
        # 'Engineering Ship',
        # 'Boeing777',
        # 'Tugboat',
        # 'A330',
        # 'Boeing787',
        # 'Boeing747',
        # 'other-ship',
        # 'A321',
        # 'Tennis Court',
        # 'Liquid Cargo Ship',
        # 'other-vehicle',
        # 'Boeing737',
        # 'Fishing Boat',
        # 'A220',
        # 'Intersection',
        # 'Motorboat',
        # 'Cargo Truck',
        # 'Dry Cargo Ship',
        # 'other-airplane',
        # 'Dump Truck',
        # 'Van',
        # 'Small Car']
        new_idx = []
        balance_dict={
            "C919":(8,0),
            "ARJ21":(7,0),
            "Tractor":(5,0)
        }

        for k,d in cate_dict.items():
            classname = FAIR_CLASSES_[k-1]
            l1,l2 = balance_dict.get(classname,(1,0))
            new_d = d*l1+d[:l2]
            new_idx.extend(new_d)
        img_infos = [self.img_infos[idx] for idx in new_idx]
        return img_infos

@DATASETS.register_module()
class FAIR1M_1_5_Dataset(DOTADataset):
    CLASSES = FAIR1M_1_5_CLASSES
    def __init__(self,**kwargs):
        kwargs_copy = kwargs.copy()
        kwargs_copy['balance_category'] = False
        super().__init__(**kwargs_copy)
        self.CLASSES = FAIR1M_1_5_CLASSES

        for img_info in self.img_infos:
            boxes = img_info["ann"]["bboxes"]
            if boxes.shape[0] == 0:
                continue
            w,h = boxes[:,2],boxes[:,3]
            area = w*h
            index = area>1.
            img_info["ann"]["bboxes"] = img_info["ann"]["bboxes"][index]
            img_info["ann"]["labels"] = img_info["ann"]["labels"][index]

        self.balance_dict = kwargs.get('balance_category', False)
        if self.balance_dict:
            self.img_infos = self._balance_categories()
        self.total_len = len(self.img_infos)

    def _balance_categories(self):
        img_infos = self.img_infos
        cate_dict = {}
        for idx,img_info in enumerate(img_infos):
            unique_labels = np.unique(img_info["ann"]["labels"])
            for label in unique_labels:
                if label not in cate_dict:
                    cate_dict[label]=[]
                cate_dict[label].append(idx)
        new_idx = []

        total = 0
        for k,d in cate_dict.items():
            classname = self.CLASSES[k-1]
            total += len(d)
            print(f'{classname}: NO: {len(d)}')
        print('### Before adjusting :', total)

        '''
        Original FAIR1m-1.5:

        Vehicle: NO: 1489
        Intersection: NO: 855
        Ship: NO: 1492
        Bridge: NO: 198
        Airplane: NO: 2188
        Basketball_Court: NO: 214
        Baseball_Field: NO: 192
        Tennis_Court: NO: 211
        Roundabout: NO: 98
        Football_Field: NO: 184

        After including DOTAv1:

        Vehicle: NO: 4101
        Intersection: NO: 855
        Ship: NO: 3327
        Bridge: NO: 1600
        Airplane: NO: 4150
        Basketball_Court: NO: 477
        Baseball_Field: NO: 532
        Tennis_Court: NO: 887
        Roundabout: NO: 627
        Football_Field: NO: 418
        '''

        for k,d in cate_dict.items():
            classname = self.CLASSES[k-1]
            scale = self.balance_dict.get(classname, 1.0)
            s_int, s_frac = int(scale), scale - int(scale)
            new_d = d*s_int+d[:int(len(d)*s_frac)]
            new_idx.extend(new_d)
            print('clsname', classname, len(new_d))
        img_infos = [self.img_infos[idx] for idx in new_idx]

        print('### After adjusting :', len(img_infos))
        return img_infos
