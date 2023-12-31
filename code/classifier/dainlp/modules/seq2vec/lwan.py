import torch
import torch.nn.functional as F


'''[2021-10-16] https://github.com/jamesmullenbach/caml-mimic/blob/master/learn/models.py#L128'''
class MullenbachModel(torch.nn.Module):
    def __init__(self, input_dim, num_labels):
        super(MullenbachModel, self).__init__()
        self.U = torch.nn.Linear(input_dim, num_labels)
        torch.nn.init.xavier_uniform_(self.U.weight)
        self.final = torch.nn.Linear(input_dim, num_labels)
        torch.nn.init.xavier_uniform_(self.final.weight)

    def forward(self, inputs):
        # inputs.transpose(1, 2).shape: batch_size, hidden_dim, seq_len
        alpha = torch.softmax(self.U.weight.matmul(inputs.transpose(1, 2)), dim=2) # batch_size, num_labels, seq_len
        hidden_states = alpha.matmul(inputs) # batch_size, num_labels, dim
        logits = self.final.weight.mul(hidden_states).sum(dim=2).add(self.final.bias) # batch_size, num_labels
        return logits

class Prototype(torch.nn.Module):
    def __init__(self, input_dim, num_labels):
        super(Prototype, self).__init__()
        proto_dim = input_dim
        # ⭐option: input_dim -> mid_dim(1024->128\64等原型空间的dim)[forward()里也要改]
        # self.head = torch.nn.Linear(input_dim, num_labels, bias=False)
        # proto_dim = 64/128/256...
        # initialize weight
        # option 1: 正态分布随机初始化
        w = torch.empty((proto_dim, num_labels))
        torch.nn.init.xavier_uniform_(w)
        # option 2: 用带有类别含义的vector初始化
        # to do
        self.proto = torch.nn.Parameter(w, requires_grad=True)


    def forward(self, inputs):
        # inputs: batch_size, hidden_dim
        def sim(x, y):
            norm_x = F.normalize(x, dim=-1)
            norm_y = F.normalize(y, dim=0)
            return torch.matmul(norm_x, norm_y)

        logits = sim(inputs, self.proto)# batch_size, num_labels
        return logits


# [2021-11-06]
class VuIjcai(torch.nn.Module):
    def __init__(self, input_dim, num_labels):
        super(VuIjcai, self).__init__()
        self.W = torch.nn.Linear(input_dim, input_dim)
        self.U = torch.nn.Linear(input_dim, num_labels)
        torch.nn.init.xavier_uniform_(self.W.weight)
        torch.nn.init.xavier_uniform_(self.U.weight)

        self.final = torch.nn.Linear(input_dim, num_labels)
        torch.nn.init.xavier_uniform_(self.final.weight)

    def forward(self, inputs):
        Z = torch.tanh(self.W(inputs)) # batch_size, seq_len, input_dim
        alpha = torch.softmax(self.U.weight.matmul(Z.transpose(1, 2)), dim=2) # batch_size, num_labels, seq_len
        hidden_states = alpha.matmul(inputs) # batch_size, num_labels, dim
        logits = self.final.weight.mul(hidden_states).sum(dim=2).add(self.final.bias) # batch_size, num_labels
        return logits


if __name__ == "__main__":
    inputs = torch.rand((3, 4, 5))

    model = VuIjcai(input_dim=5, num_labels=6)
    outputs = model(inputs)