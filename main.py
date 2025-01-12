from PACK import *
from torch.optim.lr_scheduler import StepLR

from model import GAT
from utils import buildTestData
from collect_graph import removeIsolated, collectGraph_train, collectGraph_test

import numpy as np
import math
import time
import random
import visdom
from tqdm import tqdm

import argparse
import ast

eval_func = '/path/to/compute_ap'
retrieval_result = '/path/to/retrieval'
test_dataset = {
    'oxf': {
        'node_num': 5063,
        'img_testpath': '/path/to/images',
        'feature_path': '/path/to/feature',
        'gt_path': '/path/to/oxford5k_groundTruth',
    },
    'par': {
        'node_num': 6392,
        'img_testpath': '/path/to/images',
        'feature_path': '/path/to/feature',
        'gt_path': '/path/to/paris_groundTruth',
    }
}
building_oxf = buildTestData(img_path=test_dataset['oxf']['img_testpath'], gt_path=test_dataset['oxf']['gt_path'], eval_func=eval_func)
building_par = buildTestData(img_path=test_dataset['par']['img_testpath'], gt_path=test_dataset['par']['gt_path'], eval_func=eval_func)
building = {
    'oxf': building_oxf,
    'par': building_par,
}


def train(args):
    ## load training data
    print "loading training data ......"
    node_num, class_num = removeIsolated(args.suffix)
    label, feature_map, adj_lists = collectGraph_train(node_num, class_num, args.feat_dim, args.num_sample, args.suffix)
    label = torch.LongTensor(label)
    feature_map = torch.FloatTensor(feature_map)

    model = GAT(args.feat_dim, args.embed_dim, class_num, args.alpha, args.dropout, args.nheads, args.use_cuda)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.learning_rate_decay)

    ## train
    np.random.seed(2)
    random.seed(2)
    rand_indices = np.random.permutation(node_num)
    train_nodes = rand_indices[:args.train_num]
    val_nodes = rand_indices[args.train_num:]

    if args.use_cuda:
        model.cuda()
        label = label.cuda()
        feature_map = feature_map.cuda()

    epoch_num = args.epoch_num
    batch_size = args.batch_size
    iter_num = int(math.ceil(args.train_num / float(batch_size)))
    check_loss = []
    val_accuracy = []
    check_step = args.check_step
    train_loss = 0.0
    iter_cnt = 0
    for e in range(epoch_num):
        model.train()
        scheduler.step()

        random.shuffle(train_nodes)
        for batch in range(iter_num):
            batch_nodes = train_nodes[batch*batch_size: (batch+1)*batch_size]
            batch_label = label[batch_nodes].squeeze()
            batch_neighbors = [adj_lists[node] for node in batch_nodes]
            _, logit = model(feature_map, batch_nodes, batch_neighbors)
            loss = F.nll_loss(logit, batch_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_cnt += 1
            train_loss += loss.cpu().item()
            if iter_cnt % check_step == 0:
                check_loss.append(train_loss/check_step)
                print time.strftime('%Y-%m-%d %H:%M:%S'), "epoch: {}, iter: {}, loss:{:.4f}".format(e, iter_cnt, train_loss/check_step)
                train_loss = 0.0

        ## validation
        model.eval()

        group = int(math.ceil(len(val_nodes)/float(batch_size)))
        val_cnt = 0
        for batch in range(group):
            batch_nodes = val_nodes[batch*batch_size: (batch+1)*batch_size]
            batch_label = label[batch_nodes].squeeze()
            batch_neighbors = [adj_lists[node] for node in batch_nodes]
            _, logit = model(feature_map, batch_nodes, batch_neighbors)
            batch_predict = np.argmax(logit.cpu().detach().numpy(), axis=1)
            val_cnt += np.sum(batch_predict == batch_label.cpu().numpy())
        val_accuracy.append(val_cnt/float(len(val_nodes)))
        print time.strftime('%Y-%m-%d %H:%M:%S'), "Epoch: {}, Validation Accuracy: {:.4f}".format(e, val_cnt/float(len(val_nodes)))
        print "******" * 10

    checkpoint_path = 'checkpoint/checkpoint_{}.pth'.format(time.strftime('%Y%m%d%H%M'))
    torch.save({
            'train_num': args.train_num,
            'epoch_num': args.epoch_num,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'embed_dim': args.embed_dim,
            'num_sample': args.num_sample,
            'graph_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            },
            checkpoint_path)

    vis = visdom.Visdom(env='GraphAttention', port='8099')
    vis.line(
            X = np.arange(1, len(check_loss)+1, 1) * check_step,
            Y = np.array(check_loss),
            opts = dict(
                title=time.strftime('%Y-%m-%d %H:%M:%S'),
                xlabel='itr.',
                ylabel='loss'
            )
    )
    vis.line(
            X = np.arange(1, len(val_accuracy)+1, 1),
            Y = np.array(val_accuracy),
            opts = dict(
                title=time.strftime('%Y-%m-%d %H:%M:%S'),
                xlabel='epoch',
                ylabel='accuracy'
            )
    )

    return checkpoint_path, class_num

def test(checkpoint_path, class_num, args):

    model = GAT(args.feat_dim, args.embed_dim, class_num, args.alpha, args.dropout, args.nheads, args.use_cuda)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['graph_state_dict'])
    if args.use_cuda:
        model.cuda()
    model.eval()

    for key in building.keys():
        node_num = test_dataset[key]['node_num']
        old_feature_map, adj_lists = collectGraph_test(test_dataset[key]['feature_path'], node_num, args.feat_dim, args.num_sample, args.suffix)
        old_feature_map = torch.FloatTensor(old_feature_map)
        if args.use_cuda:
            old_feature_map = old_feature_map.cuda()

        batch_num = int(math.ceil(node_num/float(args.batch_size)))
        new_feature_map = torch.FloatTensor()
        for batch in tqdm(range(batch_num)):
            start_node = batch*args.batch_size
            end_node = min((batch+1)*args.batch_size, node_num)
            batch_nodes = range(start_node, end_node)
            batch_neighbors = [adj_lists[node] for node in batch_nodes]
            new_feature, _ = model(old_feature_map, batch_nodes, batch_neighbors)
            new_feature = F.normalize(new_feature, p=2, dim=1)
            new_feature_map = torch.cat((new_feature_map, new_feature.cpu().detach()), dim=0)
        new_feature_map = new_feature_map.numpy()
        old_similarity = np.dot(old_feature_map.cpu().numpy(), old_feature_map.cpu().numpy().T)
        new_similarity = np.dot(new_feature_map, new_feature_map.T)
        mAP_old = building[key].evalRetrieval(old_similarity, retrieval_result)
        mAP_new = building[key].evalRetrieval(new_similarity, retrieval_result)
        print time.strftime('%Y-%m-%d %H:%M:%S'), 'eval {}'.format(key)
        print 'base feature: {}, new feature: {}'.format(old_feature_map.size(), new_feature_map.shape)
        print 'base mAP: {:.4f}, new mAP: {:.4f}, improve: {:.4f}'.format(mAP_old, mAP_new, mAP_new-mAP_old)

        ## directly update node's features by mean pooling features of its neighbors.
        meanAggregator = model.attentions[0]
        mean_feature_map = torch.FloatTensor()
        for batch in tqdm(range(batch_num)):
            start_node = batch*args.batch_size
            end_node = min((batch+1)*args.batch_size, node_num)
            batch_nodes = range(start_node, end_node)
            batch_neighbors = [adj_lists[node] for node in batch_nodes]
            mean_feature = meanAggregator.meanAggregate(old_feature_map, batch_nodes, batch_neighbors)
            mean_feature = F.normalize(mean_feature, p=2, dim=1)
            mean_feature_map = torch.cat((mean_feature_map, mean_feature.cpu().detach()), dim=0)
        mean_feature_map = mean_feature_map.numpy()
        mean_similarity = np.dot(mean_feature_map, mean_feature_map.T)
        mAP_mean = building[key].evalRetrieval(mean_similarity, retrieval_result)
        print 'mean aggregation mAP: {:.4f}'.format(mAP_mean)
        print ""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Graph Attention Network, train on Landmark_clean, test on Oxford5k and Paris6k.')
    parser.add_argument('-E', '--epoch_num', type=int, default=160, required=False, help='training epoch number.')
    parser.add_argument('-R', '--step_size', type=int, default=50, required=False, help='learning rate decay step_size.')
    parser.add_argument('-G', '--learning_rate_decay', type=float, default=0.5, required=False, help='learning rate decay factor.')
    parser.add_argument('-B', '--batch_size', type=int, default=64, required=False, help='training batch size.')
    parser.add_argument('-S', '--check_step', type=int, default=100, required=False, help='loss check step.')
    parser.add_argument('-C', '--use_cuda', type=ast.literal_eval, default=True, required=False, help='whether to use gpu (True) or not (False).')
    parser.add_argument('-L', '--learning_rate', type=float, default=0.005, required=False, help='training learning rate.')
    parser.add_argument('-W', '--weight_decay', type=float, default=5e-6, required=False, help='weight decay (L2 regularization).')
    parser.add_argument('-N', '--num_sample', type=int, default=10, required=False, help='number of neighbors to aggregate.')
    parser.add_argument('-x', '--suffix', type=str, default='.frmac.npy', required=False, help='feature type, \'f\' for vggnet (512-d), \'fr\' for resnet (2048-d), \'frmac\' for vgg16_rmac (512-d).')
    parser.add_argument('-f', '--feat_dim', type=int, default=512, required=False, help='input feature dim of node.')
    parser.add_argument('-D', '--embed_dim', type=int, default=512, required=False, help='embedded feature dim of encoder.')
    parser.add_argument('-A', '--alpha', type=float, default=0.2, required=False, help='alpha for LeakyReLU.')
    parser.add_argument('-P', '--dropout', type=float, default=0.3, required=False, help='dropout rate (1 - keep_probability).')
    parser.add_argument('-H', '--nheads', type=int, default=1, required=False, help='number of attention heads.')
    parser.add_argument('-T', '--train_num', type=int, default=25000, required=False, help='number of training nodes (less than 36460). Left for validation.')
    args, _ = parser.parse_known_args()
    print "< < < < < < < < < < < GraphAttentionNetWork > > > > > > > > > >"
    print "= = = = = = = = = = = PARAMETERS SETTING = = = = = = = = = = ="
    for k, v in vars(args).items():
        print k, ":", v
    print "= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = ="

    print "training ......"
    checkpoint_path, class_num = train(args)

    print "testing ......"
    test(checkpoint_path, class_num, args)
