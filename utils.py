import os
from collections import OrderedDict
import subprocess
import numpy as np

class buildTestData:
    def __init__(self, img_path, gt_path, eval_func):
        self.img_path = img_path
        self.gt_path = gt_path
        self.eval_func = eval_func

        self.build()
    def build(self):
        gt_files = np.sort(os.listdir(self.gt_path))
        ## get the image names without the extension
        self.img_names = [img[:-4] for img in np.sort(os.listdir(self.img_path))]

        self.relevant = {}
        self.non_relevant = {}
        self.junk = {}

        self.name_to_file = OrderedDict()
        for f in gt_files:
            if f.endswith('_query.txt'):
                q_name = f[:-len('_query.txt')]
                q_data = file("{}/{}".format(self.gt_path, f)).readline().split(' ')
                q_imgname = q_data[0][5:] if q_data[0].startswith('oxc1') else q_data[0]
                self.name_to_file[q_name] = q_imgname
                good = set([e.strip() for e in file("{}/{}_ok.txt".format(self.gt_path, q_name))])
                good = good.union(set([e.strip() for e in file("{}/{}_good.txt".format(self.gt_path, q_name))]))
                junk = set([e.strip() for e in file("{}/{}_junk.txt".format(self.gt_path, q_name))])
                good_plus_junk = good.union(junk)
                self.relevant[q_name] = [i for i in range(len(self.img_names)) if self.img_names[i] in good]
                self.non_relevant[q_name] = [i for i in range(len(self.img_names)) if self.img_names[i] not in good_plus_junk]
                self.junk[q_name] = [i for i in range(len(self.img_names)) if self.img_names[i] in junk]

        self.q_names = self.name_to_file.keys()
        self.q_index = np.array([self.img_names.index(self.name_to_file[q]) for q in self.q_names])
        self.img_num = len(self.img_names)
        self.q_num = len(self.q_index)

    def evalRetrieval(self, similarity, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        ranks = np.argsort(similarity, axis=1)[:,::-1]
        APs = [self.eval_q(i, ranks[q,:], save_path) for (i, q) in enumerate(self.q_index)]
        # for q in range(self.q_num):
        #     print "{}: {:.4f}".format(self.q_names[q], APs[q])
        return np.mean(APs)

    def eval_q(self, q, rank, save_path):
        rank_list = np.array(self.img_names)[rank]
        with open("{}/{}.rnkl".format(save_path, self.q_names[q]), 'w') as fw:
            fw.write("\n".join(rank_list)+"\n")
        command = "{0} {1}/{2} {3}/{4}.rnkl".format(self.eval_func, self.gt_path, self.q_names[q], save_path, self.q_names[q])
        sp = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        AP = float(sp.stdout.readlines()[0])
        sp.wait()
        return AP
