import cv2
from PIL import Image
import torch
import torchvision
from torchvision import transforms
import numpy as np
from models.arcface import resnet_face18
from models.generator import Generator
from face_detection_dsfd.face_ssd_infer import SSD
from face_detection_dsfd.data import widerface_640, TestBaseTransform
from face_landmarks.landmarks_model import hrnet_wlfw
from face_landmarks.utils import LandmarksHeatMapEncoder

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def init_models():
    arcface_path = './weights/resnet18_100_arc.pth'
    generator_path = './checkpoints/G_9.pth'
    detector_path = './weights/WIDERFace_DSFD_RES152.pth'
    lm_detector_path = './weights/hr18_wflw_landmarks.pth'

    arcface = resnet_face18(use_se=False)
    arcface = torch.nn.DataParallel(arcface)
    arcface.load_state_dict(torch.load(arcface_path, map_location=device))
    arcface.to(device)
    arcface.eval()

    generator = Generator(n_input=101, device=device)
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    generator.to(device)
    generator.eval()

    detector = SSD("test").to(device)
    detector.load_state_dict(torch.load(detector_path, map_location=device))
    detector.eval()

    lm_detector = hrnet_wlfw().to(device)
    lm_detector.load_state_dict(torch.load(lm_detector_path, map_location=device)['state_dict'])
    lm_detector.eval()

    return arcface, generator, detector, lm_detector

alpha = 0.5

img_path = 'source.jpeg'
img_cv = cv2.imread(img_path)
H, W, _ = img_cv.shape

arcface, generator, detector, lm_detector = init_models()

################ detect faces ################
detector_trans = TestBaseTransform((104, 117, 123))
thresh = widerface_640['conf_thresh']
img_tensor = torch.from_numpy(detector_trans(img_cv)[0]).permute(2, 0, 1).unsqueeze(0).to(device)
dets = detector(img_tensor)

scale = torch.Tensor([W, H, W, H])

bboxs = []
for det in dets:
    det = det.unsqueeze(0)
    for i in range(det.size(1)):
        j = 0
        while det[0, i, j, 0] >= thresh:
            curr_det = det[0, i, j, [1, 2, 3, 4, 0]].cpu().numpy()
            curr_det[:4] *= scale.cpu().numpy()
            bboxs.append(curr_det)
            j += 1
if len(bboxs) == 0:
    print("No Face detected")
    exit()

bboxs = np.row_stack(bboxs)
bboxs = bboxs[bboxs[:, 4] > 0.8, :4]

############# detect landmarks ###############
img = np.array(Image.open(img_path))
sizes = bboxs[:, 2:] - bboxs[:, :2]


lm_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])
gen_trans = transforms.Compose([transforms.ToTensor(),
                               transforms.Resize(128),
                               transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

crop_imgs = []
imgs = []
for i in range(len(bboxs)):
    bbox = bboxs[i].astype(np.int)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    size = int((w + h) * 0.8)
    bbox[0] = 0 if bbox[0] < 0 else bbox[0]
    bbox[1] = 0 if bbox[1] < 0 else bbox[1]
    x = bbox[0] + int(w / 2)
    y = bbox[1] + int(h / 2)
    size = W if size > W else size
    size = H if size > H else size

    if x - int(size / 2) < 0:
        left = 0
        right = left + size
    else:
        left = x - int(size / 2)
        right = left + size
    if right > W:
        right = W
        left = right - size

    if y - int(size / 2) < 0:
        top = 0
        bottom = top + size
    else:
        top = y - int(size / 2)
        bottom = top + size
    if bottom > H:
        bottom = H
        top = bottom - size
    bbox = [left, top, right, bottom]
    crop_img = img[top:bottom, left:right, :]
    crop_imgs.append(lm_trans(crop_img).unsqueeze(0))
    imgs.append(gen_trans(crop_img).unsqueeze(0))

crop_imgs = torch.cat(crop_imgs, dim=0)
imgs = torch.cat(imgs, dim=0)
hm_encoder = LandmarksHeatMapEncoder().to(device)

lndms_hm = lm_detector(crop_imgs.to(device))
lndms = hm_encoder(lndms_hm)

im_masks = []
mask_trans = transforms.ToTensor()
for i in range(len(lndms)):
    outline = lndms[i][:33] * 128
    outline = outline.detach().cpu().numpy().astype(np.int)
    im_mask = cv2.fillConvexPoly(np.ones((128, 128, 3), dtype=np.float32), outline, (0, 0, 0))
    im_masks.append(mask_trans(im_mask).unsqueeze(0))
im_masks = torch.cat(im_masks, dim=0)

############ Face Annoymization #############
imgs_bg = imgs * im_masks
imgs_embs = arcface(imgs)

mean, std = np.load('embeddings_std_var.npy')
embds = torch.as_tensor(np.random.normal(mean, std, (len(imgs_bg), 512)).astype(np.float32))
embds = alpha * embds + (1-alpha) * imgs_embs.cpu()


imgs_gen = generator(imgs_bg.to(device), lndms.to(device), embds.to(device))
imgs_mix = torch.cat([imgs_gen.cpu(), imgs], dim=0)
torchvision.utils.save_image(imgs_mix / 2+0.5, 'target.jpg')

