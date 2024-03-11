from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision
import cv2
import numpy as np
import time

from model import get_pose_net
from config import _C as cfg
from config import update_config
from utils import get_affine_transform, get_final_preds, place_camera

from metadata import coco_metadata, COCO_KEYPOINT_INDEXES, COCO_INSTANCE_CATEGORY_NAMES, SKELETON, CocoColors

NUM_KPTS = 17

PARENT_BOX_B_1 = [685, 418, 960, 540]
PARENT_BOX_B_2 = [314, 0, 500, 127]

PARENT_BOX_M_1 = [175, 49, 251, 127]
PARENT_BOX_M_2 = [619, 115, 683, 184]


CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def draw_pose(keypoints,img):
    """draw the keypoints and the skeletons.
    :params keypoints: the shape should be equal to [17,2]
    :params img:
    """
    assert keypoints.shape == (NUM_KPTS,2)
    for i in range(len(SKELETON)):
        #if i == 14 : 
        kpt_a, kpt_b = SKELETON[i][0], SKELETON[i][1]
        x_a, y_a = keypoints[kpt_a][0],keypoints[kpt_a][1]
        x_b, y_b = keypoints[kpt_b][0],keypoints[kpt_b][1] 
        cv2.circle(img, (int(x_a), int(y_a)), 6, CocoColors[i], -1)
        cv2.circle(img, (int(x_b), int(y_b)), 6, CocoColors[i], -1)
        cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), CocoColors[i], 2)

def draw_bbox(box,img):
    """draw the detected bounding box on the image.
    :param img:
    """
    cv2.rectangle(img, box[0], box[1], color=(0, 255, 0),thickness=3)


def get_person_detection_boxes(model, img, threshold=0.5):
    pred = model(img)
    all_person_boxes = []
    for p in range(len(pred)):
        pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i]
                        for i in list(pred[p]['labels'].cpu().numpy())]  # Get the Prediction Score
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
                    for i in list(pred[p]['boxes'].detach().cpu().numpy())]  # Bounding boxes
        pred_score = list(pred[p]['scores'].detach().cpu().numpy())
        if not pred_score or max(pred_score)<threshold:
            all_person_boxes.append([])
            continue
        # Get list of index with score greater than threshold
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
        pred_boxes = pred_boxes[:pred_t+1]
        pred_classes = pred_classes[:pred_t+1]

        person_boxes = []
        for idx, box in enumerate(pred_boxes):
            if pred_classes[idx] == 'person':
                person_boxes.append(box)
        all_person_boxes.append(person_boxes)
    return all_person_boxes

def get_pose_estimation_prediction(pose_model, image, center, scale):
    rotation = 0

    # pose estimation transformation
    trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
    model_input = cv2.warpAffine(
        image,
        trans,
        (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
        flags=cv2.INTER_LINEAR)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # pose estimation inference
    model_input = transform(model_input).unsqueeze(0)
    # switch to evaluate mode
    pose_model.eval()
    with torch.no_grad():
        # compute output heatmap
        output = pose_model(model_input)
        preds, _ = get_final_preds(
            cfg,
            output.clone().cpu().numpy(),
            np.asarray([center]),
            np.asarray([scale]))

        return preds

def box_to_center_scale(box, model_image_width, model_image_height):
    """convert a box to center,scale information required for pose transformation
    Parameters
    ----------
    box : list of tuple
        list of length 2 with two tuples of floats representing
        bottom left and top right corner of a box
    model_image_width : int
    model_image_height : int

    Returns
    -------
    (numpy array, numpy array)
        Two numpy arrays, coordinates for the center of the box and the scale of the box
    """
    center = np.zeros((2), dtype=np.float32)

    bottom_left_corner = box[0]
    top_right_corner = box[1]
    box_width = top_right_corner[0]-bottom_left_corner[0]
    box_height = top_right_corner[1]-bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str, default='config.yaml')
    parser.add_argument('--video', type=str)
    parser.add_argument('--image',type=str)
    parser.add_argument('--write',action='store_true')
    parser.add_argument('--showFps',action='store_true')
    parser.add_argument('--onset', type=int, default=0)
    parser.add_argument('--offset', type=int, default=-1)

    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # args expected by supporting codebase  
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    return args

def main():
    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    args = parse_args()
    update_config(cfg, args)

    box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    box_model.to(CTX)
    box_model.eval()

    pose_model = get_pose_net(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        pose_model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        print('expected model defined in config at TEST.MODEL_FILE')

    pose_model = torch.nn.DataParallel(pose_model, device_ids=cfg.GPUS)
    pose_model.to(CTX)
    pose_model.eval()

    # Loading an video or an image or webcam 
    if args.video:
        vidcap = cv2.VideoCapture(args.video)
    else:
        print('please use --video or --webcam or --image to define the input.')
        return 

    name = args.video.split("/")[-1]
    base_name = name.split(".")[0]
    

    if args.write:
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        save_path = 'output/output_' + name
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #out = cv2.VideoWriter(save_path,fourcc, 12.0, (int(vidcap.get(3)//2),int(vidcap.get(4)//2)))
        out = cv2.VideoWriter(save_path,fourcc, fps, (int(vidcap.get(3)),int(vidcap.get(4))))
    with open('output/output_' + base_name + ".txt", "w") as f: 
        f.write("")
    count = 0
    dataset_kpt = {}

    ## S29
    #starts = np.array([6*60+26, 17*60+50])*fps
    #ends = np.array([16*60+42, 27*60+57])*fps

    # S31_03
    #starts = np.array([20*60+18])*fps
    #ends = np.array([30*60+37])*fps

    # S18_03
    #starts = np.array([9*60+2])*fps
    #ends = np.array([19*60+14])*fps

    # S15_05
    starts = np.array([2*60+29])*fps
    ends = np.array([12*60+46])*fps

    # S 21_04
    #starts = np.array([7*60])*fps
    #ends = np.array([18*60])*fps

    # S 19_04 or 31_03
    #starts = np.array([20*60])*fps
    #ends = np.array([31*60])*fps

    # S 07_04 
    #starts = (np.array([7*60 + 1])*fps).astype(int)
    #ends = (np.array([17*60+49])*fps).astype(int)

    # S 07_03
    #starts = (np.array([7*60+56])*fps).astype(int)
    #ends = (np.array([18*60+38])*fps).astype(int)

    starts = starts.astype(int)
    ends = ends.astype(int)
    count = 0 #starts[0]
    #vidcap.set(1, count)
    while True:

        #if count > ends[-1] + 20: break
        #if count > ends[0]: break
        #if count > starts[0] + 5000: break
        ret, image_bgr = vidcap.read()
        print(count, starts[0], ends[0])
        
        if ret:
            h, w, _ = image_bgr.shape

            #image_bgr = image_bgr[:h//2, w//2:]
            last_time = time.time()
            image = image_bgr[:, :, [2, 1, 0]]
            
                
            img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img/255.).permute(2,0,1).float().to(CTX)
            img_tensor1, img_tensor2 = img_tensor[:, :h//2, :w//2], img_tensor[:, :h//2, w//2:]
            img_tensor3, img_tensor4 = img_tensor[:,h//2:,:w//2], img_tensor[:,h//2: ,w//2:]
            input = [img_tensor1, img_tensor2, img_tensor3, img_tensor4]
            # object detection box
            with torch.no_grad():
                pred_boxes = get_person_detection_boxes(box_model, input, threshold=0.84)

            # pose estimation
            #dataset_kpt[count] = {"views":[], "bbox":[], "kpt":[]}
            if len(pred_boxes) >= 1:
                bbox_tensor = []
                kps = []
                views = []
                

                for im_id in range(len(pred_boxes)):
                    for b_id, box in enumerate(pred_boxes[im_id]):
                        center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
                        image_pose = image.copy() if cfg.DATASET.COLOR_RGB else image_bgr.copy()
                        with torch.no_grad():
                            if im_id == 0:
                                pose_preds = get_pose_estimation_prediction(pose_model, image_pose[:h//2, :w//2], center, scale)
                            elif im_id == 1:
                                box = [ (int(box[0][0] +w//2) , int(box[0][1])), (int(box[1][0] +w//2) , int(box[1][1])) ]
                                pose_preds = get_pose_estimation_prediction(pose_model, image_pose[:h//2, w//2:], center, scale)
                                pose_preds[:,:,0] = pose_preds[:,:,0]+w//2
                            elif im_id == 2:
                                box = [ (int(box[0][0]) , int(box[0][1]+h//2)), (int(box[1][0]), int(box[1][1]+h//2)) ]
                                pose_preds = get_pose_estimation_prediction(pose_model, image_pose[h//2:, :w//2], center, scale)
                                pose_preds[:,:,1] = pose_preds[:,:,1]+h//2   
                            else:
                                box = [ (int(box[0][0]+w//2) , int(box[0][1]+h//2)), (int(box[1][0]+w//2) , int(box[1][1]+h//2)) ]
                                pose_preds = get_pose_estimation_prediction(pose_model, image_pose[h//2:, w//2:], center, scale)
                                pose_preds[:,:,0] = pose_preds[:,:,0]+w//2
                                pose_preds[:,:,1] = pose_preds[:,:,1]+h//2
                            bbox_tensor.append([box[0][0], box[0][1], box[1][0], box[1][1], 1.0])
                            if args.write:
                                draw_bbox(box,image_bgr)


                        if len(pose_preds)>=1:
                            kps_logit = np.zeros((pose_preds[0].shape[0], 2))
                            kps_logit[:,1] = 1.0
                            kps_all = np.concatenate((pose_preds[0], kps_logit), axis=1)
                            kps.append(kps_all)
                            if args.write:
                                for kpt in pose_preds:
                                    draw_pose(kpt,image_bgr) # draw the poses
                                    #print(box, kpt.shape)
                                    with open('output_' + base_name + ".txt", "a") as f: 
                                        f.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(count,0, b_id, 
                                            box[0][0], box[0][1], box[1][0], box[1][1], kpt[15,0], kpt[15,1], kpt[0,0], kpt[0,1]))

                        
                dataset_kpt[count] = {"views":views, "bbox":bbox_tensor, "kpt":kps}

            if args.write:
                out.write(image_bgr)
            count += 1

            if count % 50 == 0:
                print(count)

            if args.showFps:
                fps = 1/(time.time()-last_time)
                img = cv2.putText(image_bgr, 'fps: '+ "%.2f"%(fps), (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

            if count  > 100: break
            """cv2.imshow('demo',image_bgr)
            if cv2.waitKey(1) & 0XFF==ord('q'):
                break"""
        else:
            print('cannot load the video.')
            break

    #cv2.destroyAllWindows()
    vidcap.release()
    if args.write:
        print('video has been saved as {}'.format(save_path))
        out.release()
    # Video resolution
    metadata_ = {
        'w': w,
        'h': h,
    }
    #np.savez_compressed('npz_' + base_name , boxes=boxes, segments=segments, keypoints=keypoints, metadata=metadata)
    #data, video_metadata = decode(data)

    metadata = coco_metadata
    metadata['video_metadata'] = metadata_
    metadata["keypoints_name"] = COCO_KEYPOINT_INDEXES
    metadata["video"] = base_name
    np.savez_compressed('data_2d_' + base_name, data=dataset_kpt, metadata=metadata)



if __name__ == '__main__':
    main()
