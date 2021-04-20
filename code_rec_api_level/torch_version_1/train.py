import torch
import time
from sklearn import metrics

def train(epoch, dataloader, model, criterion, optimizer, opt):
    model.train()
    loss_sum = 0
    correct = 0
    sample_count = 0

    for i, (annotation, target, adj_matrix, vars) in enumerate(dataloader, 0):
        # annotation, target, A, variable
        # annotation:[b, n]
        # target: [b]
        # adj_matrix: [b, n, e * n * 2]
        # vars: [b, v]

        # clear gradient
        model.zero_grad()

        # init_state_padding = torch.zeros(len(annotation), opt.n_nodes, opt.hidden_dim - opt.annotation_dim).double()
        # init_state = torch.cat((annotation.double(), init_state_padding), 2)

        if opt.cuda:
            # init_state = init_state.cuda()
            adj_matrix = adj_matrix.cuda()
            annotation = annotation.cuda()
            target = target.cuda()
            vars = vars.cuda()

        start = time.time()

        sample_count += len(annotation)
        output = model(annotation, adj_matrix, vars)

        # loss
        loss = criterion(output, target)
        loss_sum += loss.data * len(annotation)
        # prediction
        pred = output.data.max(1, keepdim=True)[1]
        cur_correct_count = pred.eq(target.data.view_as(pred)).cpu().sum()
        correct += cur_correct_count

        labels = target.data.cpu().numpy()
        predict = torch.max(output.data, 1)[1].cpu().numpy()
        cur_acc = metrics.accuracy_score(labels, predict)
        # cur_acc = cur_correct_count / len(annotation)

        # calculate gradients of parameters
        loss.backward()
        # update parameters
        optimizer.step()

        end = time.time()
        cost = end - start

        print('epoch %d, batch %d, loss %.4f, acc %.4f correct count %d time cost %.4f' % (epoch + 1, i + 1, loss.data, cur_acc, cur_correct_count, cost))

        if i % int(len(dataloader) / 10 + 1) == 0 and opt.verbal:
            # print('[%d/%d][%d/%d] Loss: %.4f' % (epoch, opt.niter, i, len(dataloader), loss.data[0]))
            print('[%d/%d][%d/%d] Loss: %.4f' % (epoch + 1, opt.niter, i + 1, len(dataloader), loss.data))

    print('Average loss for epoch: %.4f' % (loss_sum / sample_count),
          ', accuracy: %.4f' % (float(correct) / float(sample_count)),
          ' [%d/%d]' % (correct, sample_count))
