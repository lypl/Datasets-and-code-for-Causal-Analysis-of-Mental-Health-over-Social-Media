import math, numpy as np, os, torch
import torch.nn.functional as F

def ContrastiveLoss(v_ins, logits, all_labels, num_classes, tau=1.0):
    # tau: temperature
    def sim(x, y):
        # 这里的sim和lwan.py里的sim不太一样
        norm_x = F.normalize(x, dim=-1)
        norm_y = F.normalize(y, dim=-1)
        return torch.matmul(norm_x, norm_y.transpose(1,0))
    # instance-prototype loss
    sim_mat = torch.exp(logits / tau) # 每两个向量之间的相似度值：batch_size, num_classes
    # sim_mat = torch.exp(logits) # 每两个向量之间的相似度值：batch_size, num_classes
    loss = 0.
    for i in range(v_ins.shape[0]):
        pos_score = sim_mat[i][int(torch.argmax(all_labels[i]))]
        neg_score = (sim_mat[i, :].sum() - pos_score)
        loss += - torch.log(pos_score / (pos_score + neg_score)).sum()
    loss = loss / (v_ins.shape[0] * num_classes)
    
    # instance-instance loss
    loss_ins = 0.
    for i in range(v_ins.shape[0]):
        sim_instance = torch.exp(sim(v_ins, v_ins[i].reshape(1, -1)) / tau)
        # sim_instance = torch.exp(sim(v_ins, v_ins[i].reshape(1, -1)))
        pos_ins = 0.0
        neg_ins = 0.0
        for j in range(v_ins.shape[0]):
            if i == j:
                continue
            if int(torch.argmax(all_labels[j])) == int(torch.argmax(all_labels[i])):
                pos_ins += sim_instance[j][0]
            else:
                neg_ins += sim_instance[j][0]
        if pos_ins == 0.0:
            continue
        loss_ins += - torch.log(pos_ins / (pos_ins + neg_ins))
    loss_ins = loss_ins / (v_ins.shape[0] * num_classes * v_ins.shape[0] * num_classes)
    loss = loss + loss_ins

    return loss


# def NT_XentLoss(v_ins, logits, all_labels, num_classes, tau):
#     x = torch.cat((xi, xj), dim=0)

#     is_cuda = x.is_cuda
#     sim_mat = torch.mm(x, x.T)
#     if self.normalize:
#         sim_mat_denom = torch.mm(torch.norm(x, dim=1).unsqueeze(1), torch.norm(x, dim=1).unsqueeze(1).T)
#         sim_mat = sim_mat / sim_mat_denom.clamp(min=1e-16)

#     sim_mat = torch.exp(sim_mat / tau)

#     # no diag because it's not diffrentiable -> sum - exp(1 / tau)
#     # diag_ind = torch.eye(xi.size(0) * 2).bool()
#     # diag_ind = diag_ind.cuda() if use_cuda else diag_ind

#     # sim_mat = sim_mat.masked_fill_(diag_ind, 0)

#     # top
#     if self.normalize:
#         sim_mat_denom = torch.norm(xi, dim=1) * torch.norm(xj, dim=1)
#         sim_match = torch.exp(torch.sum(xi * xj, dim=-1) / sim_mat_denom / tau)
#     else:
#         sim_match = torch.exp(torch.sum(xi * xj, dim=-1) / tau)

#     sim_match = torch.cat((sim_match, sim_match), dim=0)

#     norm_sum = torch.exp(torch.ones(x.size(0)) / tau)
#     norm_sum = norm_sum.cuda() if is_cuda else norm_sum
#     loss = torch.mean(-torch.log(sim_match / (torch.sum(sim_mat, dim=-1) - norm_sum)))

#     return loss


# 原protoverbalizer的loss
# def ContrastiveLoss(v_ins, logits, labels, num_classes):
#     def sim(x, y):
#         norm_x = F.normalize(x, dim=-1)
#         norm_y = F.normalize(y, dim=-1)
#         return torch.matmul(norm_x, norm_y)
#     # instance-prototype loss
#     sim_mat = torch.exp(logits)
#     num = sim_mat.shape[1]
#     loss = 0.
#     for i in range(num):
#         pos_score = torch.diag(sim_mat[:,i,:])
#         neg_score = (sim_mat[:,i,:].sum(1) - pos_score)
#         loss += - torch.log(pos_score / (pos_score + neg_score)).sum()
#     loss = loss / (num * num_classes * num_classes)

#     # instance-instance loss
#     loss_ins = 0.
#     for i in range(v_ins.shape[0]):
#         sim_instance = torch.exp(sim(v_ins, v_ins[i]))
#         pos_ins = sim_instance[i]
#         neg_ins = (sim_instance.sum(0) - pos_ins).sum(0)
#         loss_ins += - torch.log(pos_ins / (pos_ins + neg_ins)).sum()
#     loss_ins = loss_ins / (num * num_classes * num * num_classes)
#     loss = loss + loss_ins


def pcl_loss(self, v_ins):
        
        sim_mat = torch.exp(self.sim(v_ins, self.proto))
        num = sim_mat.shape[1]
        loss = 0.
        for i in range(num):
            pos_score = torch.diag(sim_mat[:,i,:])
            neg_score = (sim_mat[:,i,:].sum(1) - pos_score)
            loss += - torch.log(pos_score / (pos_score + neg_score)).sum()
        loss = loss / (num * self.num_classes * self.num_classes)

        # instance-instance loss

        loss_ins = 0.
        for i in range(v_ins.shape[0]):
            sim_instance = torch.exp(self.sim(v_ins, v_ins[i]))
            pos_ins = sim_instance[i]
            neg_ins = (sim_instance.sum(0) - pos_ins).sum(0)
            loss_ins += - torch.log(pos_ins / (pos_ins + neg_ins)).sum()
        loss_ins = loss_ins / (num * self.num_classes * num * self.num_classes)
        loss = loss + loss_ins

        return loss

'''[Mar-30-2022] https://github.com/pytorch/fairseq/blob/main/fairseq/modules/sinusoidal_positional_embedding.py#L36'''
def get_sinusoidal_embeddings(num_embeddings, embedding_dim, padding_idx=None):
    assert embedding_dim % 2 == 0
    emb = math.log(10000) / (embedding_dim // 2 - 1)
    emb = torch.exp(torch.arange(embedding_dim // 2, dtype=torch.float) * -emb)
    emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
    if padding_idx is not None:
        emb[padding_idx, :] = 0
    return emb


'''[20211102]'''
def select_hidden_states(hidden_states, indicies):
    # given a tensor of shape (batch size, sequence length, embedding_size)
    # choose hidden states corresponding to several positions
    bs, _, dim = hidden_states.size()
    assert bs == indicies.size(0)
    sq = indicies.size(1)
    select_indices = indicies.unsqueeze(-1).expand(bs, sq, dim)
    hidden_states = hidden_states.gather(1, select_indices)
    return hidden_states # (bs, sq, dim)
    # alternative solution : https://github.com/allenai/allennlp/blob/v2.8.0/allennlp/nn/util.py#L1301
    # batch_idx = torch.arange(0, bs)
    # batch_idx = torch.repeat_interleave(batch_idx, sq)
    # selected_hidden_sates = hidden_states[batch_idx, indices.reshape(-1)]
    # selected_hidden_sates = selected_hidden_sates.reshape((bs, sq, =1))


'''[2021-Dec-29]'''
def copy_weights_from_numpy(weights, module, name, w_shape=None, w_t=False, b_shape=None, b_t=False, layer_norm=False):
    w = torch.from_numpy(weights[os.path.join(name, "scale" if layer_norm else "kernel")])
    if w_shape is not None: w = w.view(w_shape)
    if w_t: w = w.t()
    module.weight.copy_(w)
    b = torch.from_numpy(weights[os.path.join(name, "bias")])
    if b_shape is not None: b = b.view(b_shape)
    if b_t: b = b.t()
    module.bias.copy_(b)


'''[2021-Jul-26] https://github.com/huggingface/transformers/blob/v4.9.1/src/transformers/models/bert/modeling_bert.py#L415'''
class LinearThenGelu(torch.nn.Module):
    def __init__(self, input_dim, output_dim, dropout=None):
        super(LinearThenGelu, self).__init__()
        self.dense = torch.nn.Linear(input_dim, output_dim)
        self.dropout = None if dropout is None else torch.nn.Dropout(dropout)

    def forward(self, hidden_states):
        hidden_states = torch.nn.functional.gelu(self.dense(hidden_states))
        if self.dropout is not None: hidden_states = self.dropout(hidden_states)
        return hidden_states


'''[2021-Jul-26] https://github.com/huggingface/transformers/blob/v4.9.1/src/transformers/models/bert/modeling_bert.py#L430'''
class LinearThenLayerNorm(torch.nn.Module):
    def __init__(self, input_dim, output_dim, layer_norm_eps=None, dropout=0.1):
        super(LinearThenLayerNorm, self).__init__()
        self.dense = torch.nn.Linear(input_dim, output_dim)
        self.LayerNorm = None if layer_norm_eps is None else torch.nn.LayerNorm(output_dim, eps=layer_norm_eps)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, hidden_states, input_tensor=None):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if self.LayerNorm is not None:
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


'''[2021-Dec-29] https://github.com/allenai/allennlp/blob/v2.8.0/allennlp/modules/feedforward.py#L13'''
class FeedForward(torch.nn.Module):
    def __init__(self, input_dim, num_layers, hidden_dims, activations, dropout=0.0):
        super(FeedForward, self).__init__()
        if not isinstance(hidden_dims, list): hidden_dims = [hidden_dims] * num_layers
        if not isinstance(activations, list): activations = [activations] * num_layers
        if not isinstance(dropout, list): dropout = [dropout] * num_layers
        assert len(hidden_dims) == num_layers and len(activations) == num_layers and len(dropout) == num_layers

        self._activations = torch.nn.ModuleList(activations)
        input_dims = [input_dim] + hidden_dims[:-1]
        self._linear_layers = torch.nn.ModuleList([torch.nn.Linear(i, o) for i, o in zip(input_dims, hidden_dims)])
        self._dropout = torch.nn.ModuleList([torch.nn.Dropout(p=v) for v in dropout])

    def forward(self, inputs):
        outputs = inputs
        for layer, activation, dropout in zip(self._linear_layers, self._activations, self._dropout):
            outputs = dropout(activation(layer(outputs)))
        return outputs


'''[2022-Jan-15] https://github.com/LorrinWWW/Pyramid/blob/master/functions/initializations.py#L12'''
def init_linear(linear_module):
    sampling_range = np.sqrt(6.0 / (linear_module.weight.size(0) + linear_module.weight.size(1)))
    torch.nn.init.uniform_(linear_module.weight, -sampling_range, sampling_range)
    if linear_module.bias is not None:
        linear_module.bias.data.zero_()


'''[2022-Jan-15] https://github.com/LorrinWWW/Pyramid/blob/master/functions/initializations.py#L12'''
def init_embedding(embedding_module):
    sampling_range = np.sqrt(3.0 / embedding_module.weight.size(1))
    torch.nn.init.uniform_(embedding_module.weight, -sampling_range, sampling_range)


'''[2022-Jan-14] https://github.com/LorrinWWW/Pyramid/blob/master/functions/initializations.py#L28'''
def init_lstm(lstm_module, num_layers, hidden_size):
    """
    weight_ih_l[k]: the learnable input-hidden weights of the k-th layer,
        of shape `(hidden_size * input_size)` for `k = 0`. Otherwise, the shape is `(hidden_size * hidden_size)`

    weight_hh_l[k]: the learnable hidden-hidden weights of the k-th layer, of shape `(hidden_size * hidden_size)`
    """
    for i in range(num_layers):
        weight = eval(f"lstm_module.weight_ih_l{i}")
        sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        torch.nn.init.uniform_(weight, -sampling_range, sampling_range)

        bias = eval(f"lstm_module.bias_ih_l{i}")
        bias.data.zero_()
        bias.data[hidden_size: 2 * hidden_size] = 1

        weight = eval(f"lstm_module.weight_hh_l{i}")
        sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        torch.nn.init.uniform_(weight, -sampling_range, sampling_range)

        bias = eval(f"lstm_module.bias_hh_l{i}")
        bias.data.zero_()
        bias.data[hidden_size: 2 * hidden_size] = 1

        weight = eval(f"lstm_module.weight_ih_l{i}_reverse")
        sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        torch.nn.init.uniform_(weight, -sampling_range, sampling_range)

        bias = eval(f"lstm_module.bias_ih_l{i}_reverse")
        bias.data.zero_()
        bias.data[hidden_size: 2 * hidden_size] = 1

        weight = eval(f"lstm_module.weight_hh_l{i}_reverse")
        sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        torch.nn.init.uniform_(weight, -sampling_range, sampling_range)

        bias = eval(f"lstm_module.bias_hh_l{i}_reverse")
        bias.data.zero_()
        bias.data[hidden_size: 2 * hidden_size] = 1


if __name__ == "__main__":
    hidden_states = torch.rand(3, 4, 5)
    indices = [[0, 2], [1, 2], [2, 3]]
    print(hidden_states)
    hidden_states = select_hidden_states(hidden_states, torch.tensor(indices))
    print(hidden_states)