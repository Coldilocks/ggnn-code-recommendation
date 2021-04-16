def test(dataloader, model, criterion, opt):
    test_loss = 0
    sample_count = 0
    correct = 0

    model.eval()
    for i, (annotation, target, adj_matrix, vars) in enumerate(dataloader, 0):
        # annotation, target, A, variable
        # annotation:[b, n]
        # target: [b]
        # adj_matrix: [b, n, e * n * 2]
        # vars: [b, v]

        if opt.cuda:
            # init_state = init_state.cuda()
            adj_matrix = adj_matrix.cuda()
            annotation = annotation.cuda()
            target = target.cuda()
            vars = vars.cuda()

        sample_count += len(annotation)
        output = model(annotation, adj_matrix, vars)

        test_loss += criterion(output, target).data
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(dataloader.dataset)
    accuracy = 100. * correct / len(dataloader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(dataloader.dataset),
        accuracy))

    return correct

