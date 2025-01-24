from datetime import datetime
import math
import os
import random
import sys
from time import time
from tqdm import tqdm
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd

import copy

from utility.parser import parse_args
from Models import Tie
from utility.batch_test import *
from utility.logging import Logger
from torch.utils.tensorboard import SummaryWriter
from modules.rqvae import RqVae

args = parse_args()

class Trainer(object):
    def __init__(self, data_config):
        self.task_name = "%s_%s_%s" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), args.dataset, args.cf_model,)
        self.logger = Logger(filename=self.task_name, is_debug=args.debug)
        self.logger.logging("PID: %d" % os.getpid())
        self.logger.logging(str(args))

        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
 
        self.image_feats = np.load(args.data_path + '{}/image_feat.npy'.format(args.dataset)) # [[], [], ...]
        self.text_feats = np.load(args.data_path + '{}/text_feat.npy'.format(args.dataset)) # [[], [], ...]

        self.cluster_ids = np.load(args.data_path + '{}/cluster/cluster_id_{}.npy'.format(args.dataset, args.c_num)) # [0, 1, ...]
        self.vv_ids = np.load(args.data_path + '{}/vv/vv_id_{}.npy'.format(args.dataset, args.vv_num)) # [14, 13, ...]

        self.u2i = np.load(args.data_path + '{}/u2i.npy'.format(args.dataset), allow_pickle = True).item()
        self.i2u = np.load(args.data_path + '{}/i2u.npy'.format(args.dataset), allow_pickle = True).item()

        if args.dataset == 'tiktok':
            self.audio_feats = np.load(args.data_path + '{}/audio_feat.npy'.format(args.dataset))
        else:
            self.audio_feats = None

        if args.dataset == 'tiktok':
            self.n_users = 9319
            self.n_items = 6710
            self.seq_len = 6
            self.cluster_num = args.c_num
            self.vv_num = args.vv_num
        elif args.dataset == 'baby':
            self.n_users = 19445
            self.n_items = 7050
            self.seq_len = 6
            self.cluster_num = args.c_num
            self.vv_num = args.vv_num
        elif args.dataset == 'sports':
            self.n_users = 35598
            self.n_items = 18357
            self.seq_len = 6
            self.cluster_num = args.c_num
            self.vv_num = args.vv_num
        elif args.dataset == 'netflix':
            self.n_users = 13187
            self.n_items = 8413
            self.seq_len = 6
            self.cluster_num = args.c_num
            self.vv_num = args.vv_num
        
        self.model = Tie(self.n_users, self.n_items, self.emb_dim, args.dataset, self.image_feats, self.text_feats, self.audio_feats, self.cluster_num, self.vv_num)
        # self.model = STB(self.n_users, self.n_items, self.emb_dim, args.dataset, self.image_feats, self.text_feats, self.audio_feats, self.cluster_num, self.vv_num)
        self.model = self.model.cuda()

        self.optimizer_D = optim.AdamW([{'params':self.model.parameters()},], lr=self.lr)  
        self.scheduler_D = self.set_lr_scheduler()


    def set_lr_scheduler(self):
        fac = lambda epoch: 0.96 ** (epoch / 50)
        scheduler_D = optim.lr_scheduler.LambdaLR(self.optimizer_D, lr_lambda=fac)
        return scheduler_D  

    def test(self, users_to_test, users_to_cold_test, users_to_warm_test, user_one_hop, user_two_hop, photo_one_hop, photo_two_hop, user_cluster_ids, photo_cluster_id, photo_vv, item_output, is_val):
        self.model.eval()
        with torch.no_grad():
            ua_embeddings, ia_embeddings, _, _, _, _, _ = self.model(user_one_hop, user_two_hop, photo_one_hop, photo_two_hop, user_cluster_ids, photo_cluster_id, photo_vv, item_output)
        result = test_torch(ua_embeddings, ia_embeddings, users_to_test, is_val)
        result_cold = cold_test_torch(ua_embeddings, ia_embeddings, users_to_cold_test, is_val)
        result_warm = warm_test_torch(ua_embeddings, ia_embeddings, users_to_warm_test, is_val)
        return result, result_cold, result_warm
    

    def train(self):

        self.criterion = nn.MSELoss(reduction='none')

        rou = 0.07
        vae_input_dim = args.codedim
        vae_embed_dim = args.codedim
        vae_hidden_dim = args.codedim
        if args.dataset == 'tiktok':
            visual_input_dim = 128
            visual_hidden_dim = 64
            visual_output_dim = int(args.codedim / 2)
            text_input_dim = 768
            text_hidden_dim = 192
            text_output_dim = int(args.codedim / 2)
        elif args.dataset == 'sports' or args.dataset == 'baby':
            visual_input_dim = 4096
            visual_hidden_dim = 256
            visual_output_dim = int(args.codedim / 2)
            text_input_dim = 1024
            text_hidden_dim = 128
            text_output_dim = int(args.codedim / 2)
        elif args.dataset == 'netflix':
            visual_input_dim = 512
            visual_hidden_dim = 128
            visual_output_dim = int(args.codedim / 2)
            text_input_dim = 768
            text_hidden_dim = 256
            text_output_dim = int(args.codedim / 2)

        
        vae_n_layers = args.codelen
        vae_codebook_size = [args.codesize] * args.codelen
        

        item_load_path = '/home/shiqifan/rq_vae/STB/datasets/' + args.dataset + '/item_codelen_{}_codesize_{}_codedim_{}.pth'.format(args.codelen, args.codesize, args.codedim)
        

        item_codebook = RqVae(
            rou=rou,
            input_dim=vae_input_dim,
            embed_dim=vae_embed_dim,
            hidden_dim=vae_hidden_dim,
            visual_input_dim=visual_input_dim,
            visual_hidden_dim=visual_hidden_dim,
            visual_output_dim=visual_output_dim,
            text_input_dim=text_input_dim,
            text_hidden_dim=text_hidden_dim,
            text_output_dim=text_output_dim,
            codebook_size=vae_codebook_size,
            n_layers=vae_n_layers
        )

        item_codebook.load_state_dict(torch.load(item_load_path))
        item_codebook = item_codebook.cuda()

        now_time = datetime.now()
        run_time = datetime.strftime(now_time,'%Y_%m_%d__%H_%M_%S')

        training_time_list = []
        loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []

        stopping_step = 0
        should_stop = False
        cur_best_pre_0 = 0. 

        n_batch = data_generator.n_train // args.batch_size + 1
        best_recall = 0

        final_recall, final_precison, final_ndcg = [], [], []
        final_c_recall, final_c_precison, final_c_ndcg = [], [], []
        final_w_recall, final_w_precison, final_w_ndcg = [], [], []
        for epoch in range(args.epoch):
            t1 = time()
            loss, mf_loss, emb_loss, reg_loss, cl_loss = 0., 0., 0., 0., 0.

            n_batch = data_generator.n_train // args.batch_size + 1
            sample_time = 0.


            for idx in tqdm(range(n_batch)):
                self.model.train()
                sample_t1 = time()
                users, pos_items, neg_items, items_vv_rate = data_generator.sample()
                items_vv_rate = torch.tensor(items_vv_rate).unsqueeze(1).float().cuda()
                sample_time += time() - sample_t1

                # user_one_hop 每个用户交互的u2i，做好填充 [[10个], [10个], ...]
                # user_two_hop 每个用户交互的u2i2u，做好填充 [[10个], [10个], ...]
                # photo_one_hop 每个视频被交互的i2u，做好填充 [[10个], [10个], ...]
                # photo_two_hop 每个视频被交互的i2u2i，做好填充 [[10个], [10个], ...]
                # user_cluster_ids 每个用户交互的u2i的聚类id，做好填充 [[10个], [10个], ...]
                # photo_cluster_id 每个视频的聚类id [id, id, ...] done
                # photo_vv 每个视频的vv分层id [id, id, ...] done

                user_one_hop = []
                user_two_hop = []
                user_cluster_ids = []
                for i in range(self.n_users):
                    if i in self.u2i.keys():
                        user_history = self.u2i[i]
                        if len(user_history) >= self.seq_len:
                            tmp_one_hop = random.sample(user_history, self.seq_len)
                            user_one_hop.append(tmp_one_hop)

                            tmp_cluster_id = []
                            tmp_two_hop = []
                            for item in tmp_one_hop:
                                if item < len(self.cluster_ids):
                                    tmp_cluster_id.append(self.cluster_ids[item])
                                else:
                                    tmp_cluster_id.append(self.cluster_num)

                                if item in self.i2u.keys() and len(self.i2u[item]) > 0:
                                    tmp_two_hop.append(random.choice(self.i2u[item]))
                                else:
                                    tmp_two_hop.append(self.n_users)

                            user_cluster_ids.append(tmp_cluster_id)
                            user_two_hop.append(tmp_two_hop)
                        else:
                            tmp_one_hop = user_history + [self.n_items] * (self.seq_len - len(user_history))
                            user_one_hop.append(tmp_one_hop)

                            tmp_cluster_id = []
                            tmp_two_hop = []
                            for item in tmp_one_hop:
                                if item < len(self.cluster_ids):
                                    tmp_cluster_id.append(self.cluster_ids[item])
                                else:
                                    tmp_cluster_id.append(self.cluster_num)

                                if item in self.i2u.keys() and len(self.i2u[item]) > 0:
                                    tmp_two_hop.append(random.choice(self.i2u[item]))
                                else:
                                    tmp_two_hop.append(self.n_users)
                            user_cluster_ids.append(tmp_cluster_id)
                            user_two_hop.append(tmp_two_hop)
                            
                    else:
                        user_one_hop.append([self.n_items] * self.seq_len)
                        user_two_hop.append([self.n_users] * self.seq_len)
                        user_cluster_ids.append([self.cluster_num] * self.seq_len)

                user_one_hop.append([self.n_items] * self.seq_len)
                user_two_hop.append([self.n_users] * self.seq_len)
                user_cluster_ids.append([self.cluster_num] * self.seq_len)


                photo_one_hop = []
                photo_two_hop = []
                for i in range(self.n_items):
                    if i in self.i2u.keys():
                        item_history = self.i2u[i]
                        if len(item_history) >= self.seq_len:
                            tmp_one_hop = random.sample(item_history, self.seq_len)
                            photo_one_hop.append(tmp_one_hop)

                            tmp_two_hop = []
                            for user in tmp_one_hop:
                                if user in self.u2i.keys() and len(self.u2i[user]) > 0:
                                    tmp_two_hop.append(random.choice(self.u2i[user]))
                                else:
                                    tmp_two_hop.append(self.n_items)
                            photo_two_hop.append(tmp_two_hop)
                        else:
                            tmp_one_hop = item_history + [self.n_items] * (self.seq_len - len(item_history))
                            photo_one_hop.append(tmp_one_hop)
       
                            tmp_two_hop = []
                            for user in tmp_one_hop:
                                if user in self.u2i.keys() and len(self.u2i[user]) > 0:
                                    tmp_two_hop.append(random.choice(self.u2i[user]))
                                else:
                                    tmp_two_hop.append(self.n_items)
                            photo_two_hop.append(tmp_two_hop)
                            
                    else:
                        photo_one_hop.append([self.n_users] * self.seq_len)
                        photo_two_hop.append([self.n_items] * self.seq_len)
                
                photo_one_hop.append([self.n_users] * self.seq_len)
                photo_two_hop.append([self.n_items] * self.seq_len)


                user_one_hop = torch.tensor(np.array(user_one_hop, dtype=np.int32)).cuda()
                user_two_hop = torch.tensor(np.array(user_two_hop, dtype=np.int32)).cuda()
                photo_one_hop = torch.tensor(np.array(photo_one_hop, dtype=np.int32)).cuda()
                photo_two_hop = torch.tensor(np.array(photo_two_hop, dtype=np.int32)).cuda()
                user_cluster_ids = torch.tensor(np.array(user_cluster_ids, dtype=np.int32)).cuda()
                photo_cluster_id = torch.tensor(self.cluster_ids).cuda()
                photo_vv = torch.tensor(self.vv_ids).cuda()

                image_feats_tensor = torch.tensor(self.image_feats).float().cuda()
                text_feats_tensor = torch.tensor(self.text_feats).float().cuda()
                item_output = item_codebook.get_semantic_ids(image_feats_tensor, text_feats_tensor)

                G_ua_embeddings, G_ia_embeddings, G_pid_embeddings, G_sid_embeddings, u_v_emb, u_t_emb, u_a_emb = self.model(user_one_hop, user_two_hop, photo_one_hop, photo_two_hop, user_cluster_ids, photo_cluster_id, photo_vv, item_output)

                # G_ua_embeddings, G_ia_embeddings, u_v_emb, u_t_emb, u_a_emb = self.model(user_one_hop, user_two_hop, photo_one_hop, photo_two_hop, user_cluster_ids, photo_cluster_id, photo_vv)

                G_u_g_embeddings = G_ua_embeddings[users]
                G_pos_i_g_embeddings = G_ia_embeddings[pos_items]
                G_neg_i_g_embeddings = G_ia_embeddings[neg_items]

                G_pos_pid_embeddings = G_pid_embeddings[pos_items]
                G_pos_sid_embeddings = G_sid_embeddings[pos_items]

                transfer_loss = self.transfer(G_pos_pid_embeddings, G_pos_sid_embeddings, items_vv_rate)
                G_batch_mf_loss, G_batch_emb_loss, G_batch_reg_loss, G_batch_cl_loss = self.bpr_loss(G_u_g_embeddings, G_pos_i_g_embeddings, G_neg_i_g_embeddings, u_v_emb, u_t_emb, u_a_emb)

                batch_loss = G_batch_mf_loss + G_batch_emb_loss + G_batch_reg_loss + G_batch_cl_loss + transfer_loss
                # batch_loss = G_batch_mf_loss + G_batch_emb_loss + G_batch_reg_loss + G_batch_cl_loss
                                                                                                   
                self.optimizer_D.zero_grad()  
                batch_loss.backward(retain_graph=False)
                self.optimizer_D.step()

                loss += float(batch_loss)
                mf_loss += float(G_batch_mf_loss)
                emb_loss += float(G_batch_emb_loss)
                reg_loss += float(G_batch_reg_loss)
                cl_loss += float(G_batch_cl_loss)
    
    
            del G_ua_embeddings, G_ia_embeddings, G_u_g_embeddings, G_neg_i_g_embeddings, G_pos_i_g_embeddings


            if math.isnan(loss) == True:
                self.logger.logging('ERROR: loss is nan.')
                sys.exit()

            if (epoch + 1) % args.verbose != 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f + %.5f]' % (
                    epoch, time() - t1, loss, mf_loss, emb_loss, reg_loss, cl_loss)
                training_time_list.append(time() - t1)
                self.logger.logging(perf_str)

            t2 = time()
            users_to_test = list(data_generator.test_set.keys())
            users_to_val = list(data_generator.val_set.keys())
            users_to_cold_test = list(data_generator.cold_test_set.keys())
            users_to_warm_test = list(data_generator.warm_test_set.keys())
            # ret = self.test(users_to_val, user_one_hop, user_two_hop, photo_one_hop, photo_two_hop, user_cluster_ids, photo_cluster_id, photo_vv, is_val=True)  
            test_ret, test_cold_ret, test_warm_ret = self.test(users_to_test, users_to_cold_test, users_to_warm_test, user_one_hop, user_two_hop, photo_one_hop, photo_two_hop, user_cluster_ids, photo_cluster_id, photo_vv, item_output, is_val=False)
            training_time_list.append(t2 - t1)

            t3 = time()

            loss_loger.append(loss)
            rec_loger.append(test_ret['recall'].data)
            pre_loger.append(test_ret['precision'].data)
            ndcg_loger.append(test_ret['ndcg'].data)
            hit_loger.append(test_ret['hit_ratio'].data)

            tags = ["recall", "precision", "ndcg"]

            
            self.logger.logging("Test_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]" % (eval(args.Ks)[-1], test_ret['recall'][-1], test_ret['precision'][-1], test_ret['ndcg'][-1]))
            self.logger.logging("Test_Cold_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]" % (eval(args.Ks)[-1], test_cold_ret['recall'][-1], test_cold_ret['precision'][-1], test_cold_ret['ndcg'][-1]))
            self.logger.logging("Test_Warm_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]" % (eval(args.Ks)[-1], test_warm_ret['recall'][-1], test_warm_ret['precision'][-1], test_warm_ret['ndcg'][-1]))
            final_recall.append(test_ret['recall'][-1])
            final_precison.append(test_ret['precision'][-1])
            final_ndcg.append(test_ret['ndcg'][-1])

            final_c_recall.append(test_cold_ret['recall'][-1])
            final_c_precison.append(test_cold_ret['precision'][-1])
            final_c_ndcg.append(test_cold_ret['ndcg'][-1])

            final_w_recall.append(test_warm_ret['recall'][-1])
            final_w_precison.append(test_warm_ret['precision'][-1])
            final_w_ndcg.append(test_warm_ret['ndcg'][-1])

            # if ret['recall'][-1] > best_recall:
            #     best_recall = ret['recall'][-1]
            #     # test_ret = self.test(users_to_test, user_one_hop, user_two_hop, photo_one_hop, photo_two_hop, user_cluster_ids, photo_cluster_id, photo_vv, is_val=False)
            #     test_ret, test_cold_ret, test_warm_ret = self.test(users_to_test, users_to_cold_test, users_to_warm_test, user_one_hop, user_two_hop, photo_one_hop, photo_two_hop, user_cluster_ids, photo_cluster_id, photo_vv, item_output, is_val=False)
            #     self.logger.logging("Test_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]" % (eval(args.Ks)[-1], test_ret['recall'][-1], test_ret['precision'][-1], test_ret['ndcg'][-1]))
            #     self.logger.logging("Test_Cold_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]" % (eval(args.Ks)[-1], test_cold_ret['recall'][-1], test_cold_ret['precision'][-1], test_cold_ret['ndcg'][-1]))
            #     self.logger.logging("Test_Warm_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]" % (eval(args.Ks)[-1], test_warm_ret['recall'][-1], test_warm_ret['precision'][-1], test_warm_ret['ndcg'][-1]))
            #     stopping_step = 0
            # elif stopping_step < args.early_stopping_patience:
            #     stopping_step += 1
            #     self.logger.logging('#####Early stopping steps: %d #####' % stopping_step)
            # else:
            #     self.logger.logging('#####Early stop! #####')
            #     break
            

        index = torch.argmax(torch.tensor(final_recall))
        self.logger.logging('#####final result! #####')
        self.logger.logging("Test_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]" % (eval(args.Ks)[-1], final_recall[index],  final_precison[index], final_ndcg[index]))
        self.logger.logging("Test_Cold_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]" % (eval(args.Ks)[-1], final_c_recall[index], final_c_precison[index], final_c_ndcg[index]))
        self.logger.logging("Test_Warm_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]" % (eval(args.Ks)[-1], final_w_recall[index], final_w_precison[index], final_w_ndcg[index]))

        return best_recall, run_time 

    def transfer(self, pid_emb, sid_emb, vv_rate):
        loss_hot = self.criterion(sid_emb, pid_emb.detach()).mean(dim=1, keepdim=True)
        loss_cold = self.criterion(pid_emb, sid_emb.detach()).mean(dim=1, keepdim=True)
        loss = vv_rate * loss_hot + 10*(1 - vv_rate) * loss_cold
        return loss.mean()

    def bpr_loss(self, users, pos_items, neg_items, u_v_emb, u_t_emb, u_a_emb):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1./2*(users**2).sum() + 1./2*(pos_items**2).sum() + 1./2*(neg_items**2).sum()        
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.decay * regularizer
        reg_loss = 0.0
        # cl_loss = 0.0
        
        if args.dataset == 'tiktok':
            u_t_emb_transposed = torch.transpose(u_t_emb, 0, 1)
            u_a_emb_transposed = torch.transpose(u_a_emb, 0, 1)
            
            orthogonal_inner_product_vt = torch.matmul(u_v_emb, u_t_emb_transposed)
            orthogonal_inner_product_va = torch.matmul(u_v_emb, u_a_emb_transposed)
            orthogonal_inner_product_ta = torch.matmul(u_t_emb, u_a_emb_transposed)
            
            cl_loss = torch.sum(torch.square(orthogonal_inner_product_vt)) + torch.sum(torch.square(orthogonal_inner_product_va)) + torch.sum(torch.square(orthogonal_inner_product_ta))
        else:
            u_t_emb_transposed = torch.transpose(u_t_emb, 0, 1)
            orthogonal_inner_product = torch.matmul(u_v_emb, u_t_emb_transposed) 
            cl_loss =  torch.sum(torch.square(orthogonal_inner_product))
        return mf_loss, emb_loss, reg_loss, args.alpha * cl_loss

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)  

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    set_seed(args.seed)
    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    trainer = Trainer(data_config=config)
    trainer.train()