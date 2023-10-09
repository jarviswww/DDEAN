import torch
import math
import dgl
from entmax import entmax_bisect
import numpy as np
from util import *
from torch import nn
from torch.nn import Module


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())

    def forward(self, emb_i, emb_j, target):
        SIZE = emb_i.shape[0]
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = torch.mm(representations, representations.t().contiguous())
        sim_ij = torch.diag(similarity_matrix, SIZE)
        sim_ji = torch.diag(similarity_matrix, -SIZE)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        nominator = torch.exp(positives / self.temperature)
        negatives_mask = trans_to_cuda(~torch.eye(SIZE * 2, SIZE * 2, dtype=bool)).float()
        negative_sample_mask = self.sample_mask(target)
        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)
        denominator = negative_sample_mask * denominator  # 按位相乘
        loss_partial = -torch.log(nominator / (torch.sum(denominator, dim=1) + 1e-7))
        loss = torch.sum(loss_partial) / (2 * SIZE)
        return loss

    def sample_mask(self, targets):
        targets = targets.cpu().numpy()
        targets = np.concatenate([targets, targets])

        cl_dict = {}
        for i, target in enumerate(targets):
            cl_dict.setdefault(target, []).append(i)
        mask = np.ones((len(targets), len(targets)))
        for i, target in enumerate(targets):
            for j in cl_dict[target]:
                if abs(j - i) != len(targets) / 2:  # 防止mask将正样本的位置置为零
                    mask[i][j] = 0
        return trans_to_cuda(torch.Tensor(mask)).float()

class LocalAggregator(nn.Module):
    def __init__(self, dim, dropout=0.1, name=None):
        super(LocalAggregator, self).__init__()
        self.dim = dim
        self.hidden = int(dim / 2)
        self.dropout = dropout

        self.a_0 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_1 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_3 = nn.Parameter(torch.Tensor(self.dim, 1))

        self.s_0 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.s_1 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.s_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.s_3 = nn.Parameter(torch.Tensor(self.dim, 1))

        self.dp = nn.Dropout(dropout)
        self.bias = nn.Parameter(torch.Tensor(self.dim))
        self.linear = nn.Linear(2 * dim, dim)

        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, semantic_embed, adj, structure_embed, semantic_alpha, structure_alpha, mask_item=None):

        weight1 = self.semantic_weight(semantic_embed, semantic_alpha, adj)
        weight2 = self.structure_weight(structure_embed, structure_alpha, adj)

        h = self.dp(semantic_embed)
        weight = weight1 * weight2
        sum = torch.sum(weight,-1).unsqueeze(-1).repeat(1,1,weight.shape[-1])
        weight = weight/(sum+1e-7)
        output = torch.matmul(weight, h)

        return output

    def semantic_weight(self, h, semantic_alpha, adj):
        batch_size = h.shape[0]
        N = h.shape[1]

        a_input = (h.repeat(1, 1, N).view(batch_size, N * N, self.dim)
                   * h.repeat(1, N, 1)).view(batch_size, N, N, self.dim)

        e_0 = torch.matmul(a_input, self.a_0)
        e_1 = torch.matmul(a_input, self.a_1)
        e_2 = torch.matmul(a_input, self.a_2)
        e_3 = torch.matmul(a_input, self.a_3)

        e_0 = self.leakyrelu(e_0).squeeze(-1).view(batch_size, N, N)
        e_1 = self.leakyrelu(e_1).squeeze(-1).view(batch_size, N, N)
        e_2 = self.leakyrelu(e_2).squeeze(-1).view(batch_size, N, N)
        e_3 = self.leakyrelu(e_3).squeeze(-1).view(batch_size, N, N)

        mask = -9e6 * torch.ones_like(e_0)
        alpha = torch.where(adj.eq(1), e_0, mask)
        alpha = torch.where(adj.eq(2), e_1, alpha)
        alpha = torch.where(adj.eq(3), e_2, alpha)
        alpha = torch.where(adj.eq(4), e_3, alpha)
        alpha = entmax_bisect(alpha, semantic_alpha, dim=-1)
        #alpha = torch.softmax(alpha, dim=-1)

        return alpha

    def structure_weight(self, h, structure_alpha, adj):
        batch_size = h.shape[0]
        N = h.shape[1]

        a_input = (h.repeat(1, 1, N).view(batch_size, N * N, self.dim)
                   * h.repeat(1, N, 1)).view(batch_size, N, N, self.dim)

        e_0 = torch.matmul(a_input, self.s_0)
        e_1 = torch.matmul(a_input, self.s_1)
        e_2 = torch.matmul(a_input, self.s_2)
        e_3 = torch.matmul(a_input, self.s_3)

        e_0 = self.leakyrelu(e_0).squeeze(-1).view(batch_size, N, N)
        e_1 = self.leakyrelu(e_1).squeeze(-1).view(batch_size, N, N)
        e_2 = self.leakyrelu(e_2).squeeze(-1).view(batch_size, N, N)
        e_3 = self.leakyrelu(e_3).squeeze(-1).view(batch_size, N, N)

        mask = -9e6 * torch.ones_like(e_0)
        alpha = torch.where(adj.eq(1), e_0, mask)
        alpha = torch.where(adj.eq(2), e_1, alpha)
        alpha = torch.where(adj.eq(3), e_2, alpha)
        alpha = torch.where(adj.eq(4), e_3, alpha)

        w2 = alpha.detach().cpu().numpy()

        alpha = entmax_bisect(alpha, structure_alpha, dim=-1)
        w3 = alpha.detach().cpu().numpy()
        #alpha = torch.softmax(alpha, dim=-1)

        return alpha


class SpatialEncoder(nn.Module):
    def __init__(self, dim, dp):
        super(SpatialEncoder, self).__init__()
        self.dim = dim
        self.dp = dp
        self.sema1 = nn.Linear(self.dim, 1)
        self.sema2 = nn.Linear(self.dim, 1)
        self.stru = nn.Linear(self.dim, 1)
        self.linear = nn.Linear(self.dim, self.dim)
        self.agg1 = LocalAggregator(self.dim, self.dp)
        self.agg2 = LocalAggregator(self.dim, self.dp)

    def forward(self, input1, input2, structure_embed, adjS, adjND, mask, inputs_index, test):
        self.test = test
        structure_embed = self.linear(structure_embed)
        seq_input1 = torch.ones_like(input1)
        seq_input2 = torch.ones_like(input2)
        seq_structure = torch.ones_like(structure_embed)
        for i in range(len(input1)):
            seq_input1[i] = input1[i][inputs_index[i]]
            seq_input2[i] = input2[i][inputs_index[i]]
            seq_structure[i] = structure_embed[i][inputs_index[i]]

        semantic_alpha1 = self.semantic_rep1(input1, seq_input1[:,-1,:], mask)
        semantic_alpha2 = self.semantic_rep2(input2, seq_input2[:,-1,:], mask)
        structure_alpha = self.structure_rep(structure_embed, seq_structure[:,-1,:], mask)

        output1 = self.agg1(input1, adjS, structure_embed, semantic_alpha1, structure_alpha)
        output2 = self.agg2(input2, adjND, structure_embed, semantic_alpha2, structure_alpha)
        return output1, output2

    def semantic_rep1(self, x, last, mask):
        target = self.avgPool(x, mask) + last
        semantic_rep = torch.sigmoid(self.sema1(target))+1
        alpha = self.get_alpha(semantic_rep)
        return alpha

    def semantic_rep2(self, x, last, mask):
        target = self.avgPool(x, mask) + last
        y = self.sema2(target)
        semantic_rep = torch.sigmoid(y)+1
        alpha = self.get_alpha(semantic_rep)
        return alpha

    def structure_rep(self, x, last, mask):
        target = self.avgPool(x, mask) + last
        y = self.stru(target)
        structure_rep = torch.sigmoid(y)+1
        alpha = self.get_alpha(structure_rep)
        return alpha

    def get_alpha(self, x=None, number=None):
        alpha_ent = self.add_value(x).unsqueeze(1)
        alpha_ent = alpha_ent.expand(-1, 69, -1)
        return alpha_ent

    def add_value(self, value):
        mask_value = (value == 1).float()
        value = value.masked_fill(mask_value == 1, 1.0001)
        return value


    def avgPool(self, input, mask):
        input = input * mask
        input_sum = torch.sum(input, dim=1)
        dev = torch.sum(mask.squeeze(-1).int(), dim=-1).unsqueeze(-1).repeat(1, input.shape[-1])
        return input_sum / dev

class TempSpatialEncoder(nn.Module):
    def __init__(self, opt, item_dim, pos_embedding, k, layers, dropout_in, dropout_hid):
        super(TempSpatialEncoder, self).__init__()
        self.opt = opt
        self.dim = item_dim
        self.layers = layers
        self.structural_dim = k
        self.dpin = dropout_in
        self.dphid = dropout_hid
        self.pos_embedding = pos_embedding

        self.structureE = nn.Linear(self.structural_dim, self.dim)
        self.structureE_out = nn.Linear(self.dim, self.structural_dim)

        self.SpatEnds = []
        for i in range(self.layers):
            Sp = SpatialEncoder(self.dim, self.dphid)
            self.add_module('spatial_encoder_{}'.format(i), Sp)
            self.SpatEnds.append(Sp)

        self.w_1 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.final_cat = nn.Linear(self.dim * 2, self.dim)
        self.cat = nn.Linear(self.dim * 2, self.dim, bias=False)
        self.fn = nn.Linear(self.dim * 2, self.dim)
        self.gnn_cat = nn.Linear(self.dim * 2, self.dim)

        self.glu1 = nn.Linear(self.dim, self.dim)
        self.glu2 = nn.Linear(self.dim, self.dim, bias=False)
        self.glu3 = nn.Linear(self.dim, self.dim, bias=False)
        self.dpin1 = nn.Dropout(self.dpin)
        self.dpin2 = nn.Dropout(self.dpin)

    def forward(self, gnn_input, structure_embed, pos, adj, input_index, mask):
        self.test = False
        adj = adj[0]
        adjND = adj[1]
        gnn_input_mask = mask[0]
        gnn_mask = mask[1]
        len = gnn_input.shape[1]
        batch_size = gnn_input.shape[0]
        sementic_embed = gnn_input

        pos_emb = pos[:len]  # Seq_len x Embed_dim
        pos = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)  # Batch x Seq_len x Embed_dim

        sementic_embed = sementic_embed * gnn_input_mask
        structure_embed = self.structureE(structure_embed) * gnn_input_mask

        gnn_out = self.gnn_encoder(sementic_embed, structure_embed, adj, adjND, input_index, gnn_input_mask)
        gnn_out_q = self.SoftAtten(gnn_out, pos, gnn_mask)

        return gnn_out_q, self.structureE_out(structure_embed)


    def predict(self, sementic_embed, structure_embed, pos, adj, adjND, input_index, mask):
        self.test = True
        gnn_input_mask = mask[0]
        gnn_mask = mask[1]
        len = sementic_embed.shape[1]
        batch_size = sementic_embed.shape[0]

        sementic_embed = sementic_embed * gnn_input_mask
        structure_embed = self.structureE(structure_embed) * gnn_input_mask

        pos_emb = pos[:len]  # Seq_len x Embed_dim
        pos = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)  # Batch x Seq_len x Embed_dim


        gnn_out = self.gnn_encoder(sementic_embed, structure_embed, adj, adjND, input_index, gnn_input_mask)
        gnn_out_q = self.SoftAtten(gnn_out, pos, gnn_mask)

        return gnn_out_q

    def avgPool(self, input, mask):
        input = input * mask
        input_sum = torch.sum(input, dim=1)
        dev = torch.sum(mask.squeeze(-1).int(), dim=-1).unsqueeze(-1).repeat(1, input.shape[-1])
        return input_sum / dev

    def gnn_encoder(self, sementic_embed, structure_embed, adj, adjND, inputs_index, mask):
        sementic_embed = self.dpin2(sementic_embed)
        structure_embed = self.dpin2(structure_embed)

        last_0 = sementic_embed
        last_1 = sementic_embed
        for i in range(self.layers):
            output_0, output_1 = self.SpatEnds[i](last_0, last_1, structure_embed, adj, adjND, mask, inputs_index, self.test)
            last_0 = output_0
            last_1 = output_1

        for i in range(len(output_0)):
            output_0[i] = output_0[i][inputs_index[i]]
            output_1[i] = output_1[i][inputs_index[i]]

        hidden = torch.cat((output_0, output_1), dim=-1)
        alpha = torch.sigmoid(self.gnn_cat(hidden))
        output = alpha * output_0 + (1 - alpha) * output_1

        return output

    def SoftAtten(self, hidden, pos, mask):
        mask = mask.float().unsqueeze(-1)  # Batch x Seq_len x 1

        batch_size = hidden.shape[0]  # Batch
        lens = hidden.shape[1]  # Seq_len
        pos_emb = pos  # Batch x Seq_len x Embed_dim

        hs = torch.sum(hidden * mask, -2) / (torch.sum(mask, 1) + 1e-7)
        hs = hs.unsqueeze(-2).repeat(1, lens, 1)
        nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        nh = torch.tanh(nh)
        nl = hidden[:, -1, :].unsqueeze(-2).repeat(1, lens, 1)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs) + self.glu3(nl))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * hidden, 1)

        return select


class CMCL(Module):
    def __init__(self, opt, num_node):
        super(CMCL, self).__init__()
        # hyper-parameter definition
        self.opt = opt
        self.item_dim = opt.embedding
        self.pos_dim = opt.posembedding
        self.batch_size = opt.batchSize

        self.num_node = num_node
        self.lam_alpha = opt.lambda_alpha
        self.lam_beta = opt.lambda_beta

        self.dropout_in = opt.dropout_in
        self.dropout_hid = opt.dropout_hid

        self.layers = opt.layer
        self.item_centroids = None
        self.item_2cluster = None
        self.k = opt.k
        self.p = opt.p
        self.threshold = opt.threshold

        # embedding definition
        self.embedding = nn.Embedding(self.num_node, self.item_dim, max_norm=1.5)
        self.pos_embedding = nn.Embedding(self.num_node, self.item_dim, max_norm=1.5)

        # component definition
        self.model = TempSpatialEncoder(self.opt, self.item_dim, self.pos_embedding, self.k, self.layers, self.dropout_in, self.dropout_hid)

        # training definition
        self.loss_function = nn.CrossEntropyLoss()
        self.laplacian_loss = LapLoss(self.k, self.lam_alpha)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=0)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step,
                                                         gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.item_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, target, inputs, adjND, adjS, structure_embed, session_items, inputs_index):

        # ------------------------------ interest-orient decoupling module -------------------------------------#
        # subsession, adjS, adjND = self.cate_based_denoise(adjS, adjND, inputs, session_items)

        # ---------------------------------- input and mask generator --------------------------------------#
        # sequential mask
        gnn_seq = torch.ones_like(session_items)
        for i in range(len(session_items)):
            gnn_seq[i] = session_items[i][inputs_index[i]]
        gnn_seq_mask = (gnn_seq != 0).float()

        # item mask
        timeline_mask = trans_to_cuda(torch.BoolTensor(session_items.detach().cpu().numpy() == 0))
        mask_crop = ~timeline_mask.unsqueeze(-1)

        session_items_embed = self.embedding.weight[session_items]
        target_embed = self.embedding.weight[target]
        mask = [mask_crop, gnn_seq_mask]
        adj = [adjS, adjND]

        # ---------------------------------- Spatial and Temporal encoders --------------------------------------#
        output, structure_embed_2 = self.model(session_items_embed, structure_embed, self.pos_embedding.weight, adj, inputs_index, mask)

        # ------------------------------------------Compute Score---------------------------------------------- #
        Result1 = self.decoder(output)
        LpLoss = self.LpLoss(structure_embed_2, adj[0])

        return Result1, LpLoss * self.lam_beta

    def LpLoss(self, P, A):
        L = []
        for adj in A:
            n = len(adj)
            in_degrees = adj.sum(dim=1)
            in_degrees = torch.clamp(in_degrees, min=1)
            N = torch.diag(torch.pow(in_degrees.float(), -0.5))
            L1 = torch.eye(n) - torch.mm(torch.mm(N, adj), N).detach().cpu()
            L.append(L1.numpy())
        L = torch.tensor(L).cuda()

        #A = torch.where(A == 0, 0, 1)
        #D = torch.diag_embed(torch.sum(A, dim=2))

        # 计算拉普拉斯矩阵 L（L = D - A）
        #L = D - A

        Loss = self.laplacian_loss(P, L.float())

        return Loss


    def cate_based_denoise(self, adj, adjND, inputs, items):

        adj = adj.detach().cpu()
        adjND = adjND.detach().cpu()
        items = items.detach().cpu()
        p = 1 - self.p
        maskpad1 = (inputs != 0).cpu()
        maskpad2 = (items != 0).cpu()
        item_cat_seq = self.item_2cluster[inputs].detach().cpu() * maskpad1
        item_cat_items = self.item_2cluster[items].detach().cpu() * maskpad2
        item_cat_embed = self.item_centroids[item_cat_items].detach().cpu()

        most_click = []
        pad_cat = torch.tensor(0)
        for i in item_cat_seq:
            weights = torch.where(i != pad_cat, 1, 0)
            most_click.append(torch.argmax(torch.bincount(i, weights)))
        most_click = torch.tensor(most_click)
        most_clicks = most_click.unsqueeze(-1).repeat(1, adj.shape[-1])
        most_clicks_embed = self.item_centroids[most_clicks].detach().cpu()
        sim_matrix_0 = torch.cosine_similarity(item_cat_embed, most_clicks_embed, dim=2)
        mask_0 = 1 - torch.where(sim_matrix_0 > 0, 1, 0)

        last_click = item_cat_seq[:, -1]
        last_click = last_click
        last_clicks = last_click.unsqueeze(-1).repeat(1, adj.shape[-1])
        last_clicks_embed = self.item_centroids[last_clicks].detach().cpu()
        sim_matrix_1 = torch.cosine_similarity(item_cat_embed, last_clicks_embed, dim=2)
        mask_1 = 1 - torch.where(sim_matrix_1 > 0, 1, 0)
        mask_ = torch.bitwise_and(mask_0, mask_1)

        mask_matrix_reverse = rand_matrix * mask_  # 只有那些属性同lastclick不一样的物品会被以概率丢弃，这里就是按它们的位置生成概率，同时mask掉属性同lastclick相同的物品
        mask = 1 - torch.where(mask_matrix_reverse < p, 0, 1)
        mask_col = mask.unsqueeze(1).repeat(1, mask.shape[-1], 1)
        mask_row = mask.unsqueeze(-1).repeat(1, 1, mask.shape[-1])
        adj = adj * mask_col * mask_row
        adjND = adjND * mask_col * mask_row
        items = items * mask

        return trans_to_cuda(items), trans_to_cuda(adj), trans_to_cuda(adjND)  # , trans_to_cuda(items)


    def e_step(self):
        items_embedding = self.embedding.weight.detach().cpu().numpy()
        self.item_centroids, self.item_2cluster = self.run_kmeans(items_embedding[:])

    def run_kmeans(self, x):
        kmeans = faiss.Kmeans(d=x.shape[-1], niter=50, k=self.k, gpu=True)
        kmeans.train(x)
        cluster_cents = kmeans.centroids
        self.cluster_cents = cluster_cents
        _, I = kmeans.index.search(x, 1)
        self.items_cents = I

        # convert to cuda Tensors for broadcast
        centroids = trans_to_cuda(torch.Tensor(cluster_cents))
        centroids = F.normalize(centroids, p=2, dim=1)

        node2cluster = trans_to_cuda(torch.LongTensor(I).squeeze())
        return centroids, node2cluster

    def decoder(self, select):
        l_c = (select / torch.norm(select, dim=-1).unsqueeze(1))
        l_emb = self.embedding.weight[1:] / torch.norm(self.embedding.weight[1:], dim=-1).unsqueeze(1)
        z = 13 * torch.matmul(l_c, l_emb.t())

        return z

    def predict(self, data, k):
        # note that for prediction, CMCL don't have the decoupling module
        target, x_test, adjS, adjND, session_items, inputs_index, structure_embed = data
        print(session_items[68])
        adjS = trans_to_cuda(adjS).float()
        adjND = trans_to_cuda(adjND).float()
        session_items = trans_to_cuda(session_items).long()
        structure_embed = trans_to_cuda(structure_embed).float()

        gnn_seq = torch.ones_like(session_items)
        for i in range(len(session_items)):
            gnn_seq[i] = session_items[i][inputs_index[i]]
        gnn_seq_mask = (gnn_seq != 0).float()

        timeline_mask = trans_to_cuda(torch.BoolTensor(session_items.detach().cpu().numpy() == 0))
        mask_crop = ~timeline_mask.unsqueeze(-1)  # broadcast in last dim

        session_items_embed = self.embedding.weight[session_items]

        mask = [mask_crop, gnn_seq_mask]

        output = self.model.predict(session_items_embed, structure_embed, self.pos_embedding.weight, adjS, adjND,inputs_index,mask)
        result = self.decoder(output)
        rank = torch.argsort(result, dim=1, descending=True)
        return rank[:, 0:k]


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
        # return variable
    else:
        return variable



def forward(model, data):
    target, u_input, adjS, adjND, session_items, u_input_index, structure_embed = data

    session_items = trans_to_cuda(session_items).long()
    u_input = trans_to_cuda(u_input).long()
    adjND = trans_to_cuda(adjND).float()
    adjS = trans_to_cuda(adjS).float()
    target = trans_to_cuda(target).long()
    structure_embed = trans_to_cuda(structure_embed).float()
    Result1, LpLoss = model(target, u_input, adjND, adjS, structure_embed,  session_items, u_input_index)
    return Result1, LpLoss
