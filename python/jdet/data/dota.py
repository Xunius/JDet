from numpy.lib import save
from jdet.data.devkits.voc_eval import voc_eval_dota
from jdet.models.boxes.box_ops import rotated_box_to_poly_np, rotated_box_to_poly_single
from jdet.utils.general import check_dir
from jdet.utils.registry import DATASETS
from jdet.config.constant import get_classes_by_name
from jdet.data.custom import CustomDataset
from jdet.ops.nms_poly import iou_poly
from jdet.utils.visualization import visualize_results, visualize_results_with_gt
import os
import jittor as jt
import numpy as np
from tqdm import tqdm
import pickle

def s2anet_post(result):
    dets,labels = result 
    labels = labels+1 
    scores = dets[:,5]
    dets = dets[:,:5]
    polys = rotated_box_to_poly_np(dets)
    return polys,scores,labels

@DATASETS.register_module()
class DOTADataset(CustomDataset):

    def __init__(self,*arg,balance_category=False,version='1',**kwargs):
        assert version in ['1', '1_5', '2']
        self.CLASSES = get_classes_by_name('DOTA'+version)
        super().__init__(*arg,**kwargs)
        if balance_category:
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
        balance_dict={
            "storage-tank":(1,526),
            "baseball-diamond":(2,202),
            "ground-track-field":(1,575),
            "swimming-pool":(2,104),
            "soccer-ball-field":(1,962),
            "roundabout":(1,711),
            "tennis-court":(1,655),
            "basketball-court":(4,0),
            "helicopter":(8,0),
            "container-crane":(50,0)
        }

        for k,d in cate_dict.items():
            classname = self.CLASSES[k-1]
            l1,l2 = balance_dict.get(classname,(1,0))
            new_d = d*l1+d[:l2]
            new_idx.extend(new_d)
        img_infos = [self.img_infos[idx] for idx in new_idx]
        return img_infos
    
    def parse_result(self,results,save_path):
        check_dir(save_path)
        data = {}
        for (dets,labels),img_name in results:
            img_name = os.path.splitext(img_name)[0]
            for det,label in zip(dets,labels):
                bbox = det[:5]
                score = det[5]
                classname = self.CLASSES[label]
                bbox = rotated_box_to_poly_single(bbox)
                temp_txt = '{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n'.format(
                            img_name, score, bbox[0], bbox[1], bbox[2], bbox[3], bbox[4],
                            bbox[5], bbox[6], bbox[7])
                if classname not in data:
                    data[classname] = []
                data[classname].append(temp_txt)
        for classname,lines in data.items():
            f_out = open(os.path.join(save_path, classname + '.txt'), 'w')
            f_out.writelines(lines)
            f_out.close()

    def evaluate(self,results,work_dir,epoch,logger=None,save=True):
        print("Calculating mAP......")
        if save:
            save_path = os.path.join(work_dir,f"detections/val_{epoch}")
            check_dir(save_path)
            jt.save(results,save_path+"/val.pkl")
        dets = []
        gts = []
        diffcult_polys = {}
        for img_idx,(result,target) in enumerate(results):
            det_polys,det_scores,det_labels =  result
            det_labels += 1
            if det_polys.size>0:
                idx1 = np.ones((det_labels.shape[0],1))*img_idx
                det = np.concatenate([idx1,det_polys,det_scores.reshape(-1,1),det_labels.reshape(-1,1)],axis=1) # [n_dets_in_img, 11]
                dets.append(det)
            
            scale_factor = target["scale_factor"]
            gt_polys = target["polys"]  # [n_gt_boxes, 8]
            gt_polys /= scale_factor

            if gt_polys.size>0:
                gt_labels = target["labels"].reshape(-1,1)
                idx2 = np.ones((gt_labels.shape[0],1))*img_idx
                gt = np.concatenate([idx2,gt_polys,gt_labels],axis=1) # [n_gt_boxes, 10]
                gts.append(gt)
            diffcult_polys[img_idx] = target["polys_ignore"]/scale_factor
        if len(dets) == 0:
            aps = {}
            for i,classname in tqdm(enumerate(self.CLASSES),total=len(self.CLASSES)):
                aps["eval/"+str(i+1)+"_"+classname+"_AP"]=0 
            map = sum(list(aps.values()))/len(aps)
            aps["eval/0_meanAP"]=map
            return aps
        dets = np.concatenate(dets)
        gts = np.concatenate(gts)
        aps = {}
        all_pr = {}

        examine_class = 'Vehicle'

        for i,classname in tqdm(enumerate(self.CLASSES),total=len(self.CLASSES)):
            c_dets = dets[dets[:,-1]==(i+1)][:,:-1] # [n_dets, 10]
            c_gts = gts[gts[:,-1]==(i+1)][:,:-1]  # [n_gt_boxes, 9]
            img_idx = gts[:,0].copy()
            classname_gts = {}
            for idx in np.unique(img_idx):  # loop through images
                g = c_gts[c_gts[:,0]==idx,:][:,1:]  # [n_gt_boxes, 8]
                dg = diffcult_polys[idx].copy().reshape(-1,8) # this is never > 1
                diffculty = np.zeros(g.shape[0]+dg.shape[0])
                diffculty[int(g.shape[0]):]=1
                diffculty = diffculty.astype(bool)
                g = np.concatenate([g,dg])
                classname_gts[idx] = {"box":g.copy(),"det":[False for i in range(len(g))],'difficult':diffculty.copy()}
            rec, prec, ap, tp, fp = voc_eval_dota(c_dets,classname_gts,iou_func=iou_poly)


            # get true positives
            if len(tp) > 0 and classname == examine_class:
                save_dir = os.path.join(work_dir, 'tp_detections_%s' %classname)
                os.makedirs(save_dir, exist_ok=True)
                tp_idx = np.where(tp==1)[0]
                tp_img_idx = np.unique([int(c_dets[jj][0]) for jj in tp_idx])
                tp_dets = c_dets[tp_idx]

                # group by imgs
                tp_det_imgs = []
                tp_gts = []
                for jj in tp_img_idx:
                    idxjj = np.where(tp_dets[:,0]==jj)[0]
                    tp_det_imgs.append([tp_dets[idxjj, 1:9], tp_dets[idxjj, 9], np.ones(len(idxjj), dtype='int')*i])
                    tp_gts.append(results[jj][1])

                visualize_results_with_gt(tp_det_imgs, tp_gts, self.CLASSES, save_dir)
                with open(os.path.join(save_dir, 'tp_dets.dat'), 'wb') as fout:
                    pickle.dump(tp_dets, fout)


            # get false positives
            if len(fp) > 0 and classname == examine_class:
                save_dir = os.path.join(work_dir, 'fp_detections_%s' %classname)
                os.makedirs(save_dir, exist_ok=True)

                fp_idx = np.where(fp==1)[0]
                fp_img_idx = np.unique([int(c_dets[jj][0]) for jj in fp_idx])
                fp_dets = c_dets[fp_idx]
                # group by imgs
                fp_det_imgs = []
                fp_gts = []
                for jj in fp_img_idx:
                    idxjj = np.where(fp_dets[:,0]==jj)[0]
                    fp_det_imgs.append([fp_dets[idxjj, 1:9], fp_dets[idxjj, 9], np.ones(len(idxjj), dtype='int')*i])
                    fp_gts.append(results[jj][1])

                visualize_results_with_gt(fp_det_imgs, fp_gts, self.CLASSES, save_dir)
                with open(os.path.join(save_dir, 'fp_dets.dat'), 'wb') as fout:
                    pickle.dump(fp_dets, fout)


            aps["eval/"+str(i+1)+"_"+classname+"_AP"]=ap 
            all_pr[classname] = {'prec': prec, 'rec': rec, 'AP': ap}
        map = sum(list(aps.values()))/len(aps)
        aps["eval/0_meanAP"]=map

        return aps, all_pr
            

def test_eval():
    results= jt.load("projects/s2anet/work_dirs/s2anet_r50_fpn_1x_dota/detections/val_0/val.pkl")
    results = jt.load("projects/s2anet/work_dirs/s2anet_r50_fpn_1x_dota/detections/val_rotate_balance/val.pkl")
    # results = results
    dataset = DOTADataset(annotations_file='/mnt/disk/lxl/dataset/DOTA_1024/trainval_split/trainval1024.pkl',
        images_dir='/mnt/disk/lxl/dataset/DOTA_1024/trainval_split/images/')
    dataset.evaluate(results,None,None,save=False)
    
    # data = []
    # for result,target in results:
    #     img_name = target["filename"]
    #     data.append((result,img_name))

    # dataset.parse_result(data,"test_")



if __name__ == "__main__":
    test_eval()
