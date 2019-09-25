from tqdm import tqdm
from sacred import Experiment

from configs.bc import train_ingredient
from bc.model import utils, log
from bc.utils import misc

ex = Experiment('train', ingredients=[train_ingredient])


@ex.capture
def train(train_loader, eval_loader, net_path, net, optimizer, starting_epoch, train):
    print('Starting training...')
    losses_train = {}
    for epoch in range(starting_epoch, train['epochs']):
        print('Epoch {}'.format(epoch))
        for i, batch in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            losses_batch = net.compute_loss(*batch, eval=False)
            optimizer.step()
            utils.append_losses(losses_train, losses_batch)
            log.last(losses_train, epoch * len(train_loader) + i)

        if epoch % train['eval_interval'] == 0 and eval_loader is not None:
            print('Evaluating after epoch {}'.format(epoch))
            eval_loss_dict = utils.run_evaluation(net, eval_loader)
            log.mean(eval_loss_dict, (epoch + 1) * len(train_loader))

        if epoch % 2 == 0:
            utils.save_model(net_path, epoch, net, optimizer)


@ex.automain
def main(model, dataset):
    model, dataset = misc.update_arguments(model=model, dataset=dataset)
    train_loader, eval_loader, env_name, statistics = utils.make_loader(
        model=model, dataset=dataset)
    net, optimizer, starting_epoch, net_path = utils.make_net(
        model=model,
        dataset=dataset,
        env_name=env_name,
        statistics=statistics)
    utils.write_info(model, dataset)
    log.init_writers(net_path)
    train(train_loader, eval_loader, net_path, net, optimizer, starting_epoch)
