from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
from dataset import *
from model import Unet_C, UNet_G
import cv2


def dice_loss(pred, label):
    p = torch.sum(pred, (-1, -2))
    tx = torch.sum(label, (-1, -2))
    pt = torch.sum(pred * label, (-1, -2))

    num = 2 * pt
    den = p + tx
    dsc = torch.mean(num / den, 0)
    return dsc


def test(args):
    source_test_data = DataLoader(dataset('images/stare/', train= False, img_count= 15, height=605, width=605, type= 'stare', transforms=transforms.Compose([
        ToTensor(),
        normalize()])),
        batch_size= 1, shuffle=False)

    target_test_data = DataLoader(dataset('images/chasedb1/',train= False, img_count= 15, height= 605, width= 605, type= 'chasedb1', transforms=transforms.Compose([
        ToTensor(),
        normalize()])),
        batch_size= 1, shuffle=False)

    model_g = UNet_G()
    model_mc = Unet_C()


    device = torch.device("cuda:" + args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint)
    model_g.load_state_dict(checkpoint['g_state_dict'])
    model_mc.load_state_dict(checkpoint['mc_state_dict'])


    model_g.to(device)
    model_mc.to(device)

    model_g.eval()
    model_mc.eval()

    source_loss_list = []
    target_loss_list = []

    with torch.no_grad():
        ## Source
        dsc_src_sum = 0
        for n, sample in enumerate(source_test_data):
            data, mask_cpu, name = sample["img"], sample["label"], sample["name"]
            data, mask = data.to(device), mask_cpu.to(device)
            output = model_g(data)
            output = model_mc(output)
            dsc = dice_loss(output, mask).item()
            dsc_src_sum += dsc
            source_loss_list.append(dsc)
            # print("Source {} - DSC: {}".format(filename, dsc))

            pred = output.squeeze().cpu().numpy()
            pred[pred >= 0.5] = 1.0
            pred[pred < 0.5] = 0.0
            final_pred = np.uint8(pred * 255)
            # cv2.imwrite("./pred/source/{}_source_pred.png".format(filename), final_pred)
            mask_cpu = mask_cpu.squeeze().numpy()
            print(mask_cpu.shape)
            compare_output = np.stack([final_pred, mask_cpu,final_pred], axis=2)
            cv2.imwrite("./pred/source/{}_source_pred.png".format(name), final_pred)
            cv2.imwrite("./pred/source_compare/{}_source_pred.png".format(name), compare_output)

        print("Source Average DSC - {}".format(dsc_src_sum / (n+1)))
        print()
        ## Target
        dsc_tar_sum = 0
        for m, sample in enumerate(target_test_data):
            data, mask_cpu, name = sample["img"], sample["label"], sample["name"]
            # filename = name[0]
            data, mask = data.to(device), mask_cpu.to(device)
            output = model_g(data)
            output = model_mc(output)
            dsc = dice_loss(output, mask).item()
            dsc_src_sum += dsc
            source_loss_list.append(dsc)
            # print("Target {} - DSC: {}".format(filename, dsc))

            pred = output.squeeze().cpu().numpy()
            pred[pred >= 0.5] = 1.0
            pred[pred < 0.5] = 0.0
            final_pred = np.uint8(pred * 255)
            mask_cpu = mask_cpu.squeeze().numpy()
            print(mask_cpu.shape)
            compare_output = np.stack([final_pred, mask_cpu, final_pred], axis=2)
            cv2.imwrite("./pred/target/{}_target_pred.png".format(name), final_pred)
            cv2.imwrite("./pred/target_compare/{}_target_pred.png".format(name), compare_output)
            # cv2.imwrite("./pred/target/{}_target_pred.png".format(filename), final_pred)
        print("Target Average DSC - {}".format(dsc_tar_sum / (m+1)))


def main(args):
    test(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="0")
    parser.add_argument('--checkpoint', type=str, default="D:/domain/Untitle/check/535.tar")
    args = parser.parse_args()

    print("Testing images")
    main(args)