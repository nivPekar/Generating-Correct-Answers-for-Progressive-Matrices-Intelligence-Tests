

import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import models
from data.data_utils import get_data



parser = argparse.ArgumentParser(description='baseline')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--data_path', type=str, default="")
parser.add_argument('--save', type=str, default='./results/checkpoint/')
parser.add_argument('--log', type=str, default='./results/log/')
parser.add_argument('--img_size', type=int, default=80)










args = parser.parse_args()
args.cuda = torch.cuda.is_available()


if args.cuda:
    torch.cuda.set_device(args.device)
    torch.cuda.manual_seed(args.seed)





trainloader = get_data(data_root=args.data_path, data_cache=None , dataname="PGM", image_size=args.img_size,
             dataset_type='train', regime="neutral", subset=None,
             use_cache=False, save_cache=False, pin_memory=False, in_memory=False,
             batch_size=args.batch_size, drop_last=True, num_workers=8, ratio=None, shuffle=True, flip=False, permute=False)








loaded_CEM = models.CEM_3res(with_meta=True)


latent_dim = 64
params_for_geneator = {
        'nc': 1,  # Number of channles in the training images. For coloured images this is 3.
        'nz': latent_dim,  # Size of the Z latent vector (the input to the generator).
        'ngf': 64,  # Size of feature maps in the generator. The depth will be multiples of this.
        'ndf': 64,  # Size of features maps in the discriminator. The depth will be multiples of this.
        'lr': 0.0002,  # Learning rate for optimizers
        'beta1': 0.5  # Beta1 hyperparam for Adam optimizer
    }

generator_for_vae = models.Generator_for_vae(params_for_geneator)
encoder = models.Encoder_module(latent_dim)
loaded_vae = models.VAE(encoder, generator_for_vae)



with_dynamic_embeddings_loss_coef = True
generation_cycle = models.generation_cycle_res3(loaded_CEM, loaded_vae, with_discriminator=True, with_dynamic_embeddings_loss_coef=with_dynamic_embeddings_loss_coef)

pre_trained_generation_cycle = False

if pre_trained_generation_cycle:
    loaded_generation_cycle_name = "pre_generation_cycle"
    generation_cycle.load_state_dict(torch.load(args.save + '{}.pth'.format(loaded_generation_cycle_name)))




def plot_two_array_graf(array1, array2, label1, label2, plot_name):
    plt.plot(array1, label=label1)
    plt.legend()
    plt.plot(array2, label=label2)
    plt.legend()
    plt.savefig(args.save + plot_name + '.jpg')
    plt.clf()

def plot_one_array_graf(array, name):
    plt.plot(array, label=name)
    plt.legend()
    plt.savefig(args.save + name + '.jpg')
    plt.clf()


if with_dynamic_embeddings_loss_coef:
    avg_positive_dist_array = []
    avg_negative_dist_array = []


latent_loss_array = []
pic_loss_array = []
kld_of_cem_generated_array = []
embeddings_loss_array = []
embeddings_loss_on_negative_choice_array = []
accuracy_discriminator_target_array = []
accuracy_discriminator_generated_array = []
loss_g_generated_array = []
eval_every = 1000


for i in range(0, args.epochs):

    train_iter = iter(trainloader)
    counter = 0
    latent_loss_all = 0.0
    pic_loss_all = 0.0
    kld_of_cem_generated_all = 0.0
    embeddings_loss_all = 0.0
    embeddings_loss_on_negative_choice_all = 0.0
    accuracy_discriminator_target_all = 0.0
    accuracy_discriminator_generated_all = 0.0
    loss_g_generated_all = 0.0


    for _ in tqdm(range(len(train_iter))):
        counter += 1
        # image, target, meta_target, relation_structure = next(train_iter)  #, relation_structure
        image, target, meta_target, relation_structure, structure_encoded, *_ = next(train_iter)
        image = ((image / 255) - 0.5) * 2

        if args.cuda:
            image = image.cuda()
            target = target.cuda()
            meta_target = meta_target.cuda()

        losses_and_acc, generated_imgs = generation_cycle.train_(image, target, meta_target)
        embeddings_loss_all += losses_and_acc["embeddings_loss"]
        kld_of_cem_generated_all += losses_and_acc["total_kld_of_cem_generated"]
        embeddings_loss_on_negative_choice_all += losses_and_acc["embeddings_loss_on_negative_choice"]
        latent_loss_all += losses_and_acc["total_kld"]
        pic_loss_all += losses_and_acc["loss_pic"]

        if "accuracy_discriminator_target" in losses_and_acc.keys():
            accuracy_discriminator_target_all += losses_and_acc["accuracy_discriminator_target"]
            accuracy_discriminator_generated_all += losses_and_acc["accuracy_discriminator_generated"]
            loss_g_generated_all += losses_and_acc["loss_g_generated"]



        if counter % eval_every == 0:
            if with_dynamic_embeddings_loss_coef:
                generation_cycle.update_embeddings_loss_coef()
                avg_positive_dist = losses_and_acc["avg_positive_dist"]
                avg_negative_dist = losses_and_acc["avg_negative_dist"]
                avg_positive_dist_array.append(avg_positive_dist)
                avg_negative_dist_array.append(avg_negative_dist)
                plot_two_array_graf(avg_positive_dist_array, avg_negative_dist_array,
                                    "avg_positive_choice_dist", "avg_negative_choice_dist", "avg_gen_to_choice_relation_dist")



            embeddings_loss_array.append(embeddings_loss_all/ float(counter))
            kld_of_cem_generated_array.append(kld_of_cem_generated_all/ float(counter))
            embeddings_loss_on_negative_choice_array.append(embeddings_loss_on_negative_choice_all / float(counter))
            latent_loss_array.append(latent_loss_all / float(counter))
            pic_loss_array.append(pic_loss_all / float(counter))
            if "accuracy_discriminator_target" in losses_and_acc.keys():
                accuracy_discriminator_target_array.append(accuracy_discriminator_target_all/ float(counter))
                accuracy_discriminator_generated_array.append(accuracy_discriminator_generated_all/ float(counter))
                loss_g_generated_array.append(loss_g_generated_all/ float(counter))

            counter = 0
            latent_loss_all = 0.0
            pic_loss_all = 0.0
            embeddings_loss_on_negative_choice_all = 0.0
            embeddings_loss_all = 0.0
            accuracy_discriminator_target_all = 0.0
            accuracy_discriminator_generated_all = 0.0
            loss_g_generated_all = 0.0
            plot_one_array_graf(pic_loss_array, "pic_loss_vae")
            plot_one_array_graf(latent_loss_array, "latent_loss_vae")
            plot_one_array_graf(kld_of_cem_generated_array, "kld_of_cem_generated")
            plot_one_array_graf(embeddings_loss_array, "embeddings_loss")
            plot_one_array_graf(embeddings_loss_on_negative_choice_array, "embeddings_loss_on_negative_choice")
            if "accuracy_discriminator_target" in losses_and_acc.keys():
                plot_one_array_graf(loss_g_generated_array, "loss_g_generated")
                plot_two_array_graf(accuracy_discriminator_target_array, accuracy_discriminator_generated_array, "acc_D_target", "acc_D_generated", "D_accuracy")


    torch.save(generation_cycle.state_dict(), args.save + '{}_epoch_{}.pth'.format("generation_cycle", i))




















