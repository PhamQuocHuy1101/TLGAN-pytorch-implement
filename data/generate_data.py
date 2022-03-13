import os
import os.path as path
import sys
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from tqdm.auto import tqdm

def extract_coordinate(file):
    with open(file, 'r') as f:
        data = f.read().splitlines()
        lines = [line.split(',')[:8] for line in data]
        return [[int(value) for value in line] for line in lines]


def create_map(img_size, boxes):
    '''
        img_size: (h, w)
        boxes[i]: x0, y0, x1, y1, x2, y2, x3, y3
    '''
    map_img = np.zeros(img_size)
    for i, box in enumerate(boxes):
        xmin, xmax = min(box[::2]), max(box[::2])
        ymin, ymax = min(box[1::2]), max(box[1::2])

        width_y = ymax - ymin
        y_center = (ymax + ymin) / 2
        M = lambda y: 1.0 / (2 * np.pi * width_y) * np.exp(-y**2 / (2 * width_y**2))
        for y in range(ymin, ymax, 1):
            map_img[y, xmin:xmax] = M(y - y_center)
    return map_img

def crop_data(images_path, txt_path, crop_path, crop_size, n_loop):
    crop_image_path = path.join(crop_path, 'images')
    crop_map_image_path = path.join(crop_path, 'map_images')
    os.makedirs(crop_image_path, exist_ok = True)
    os.makedirs(crop_map_image_path, exist_ok = True)
    
    crop = transforms.RandomCrop((crop_size, crop_size))
    files = sorted(os.listdir(images_path))
    
    for img_file in tqdm(files):
        name = img_file[:-4]
        image = Image.open(path.join(images_path, img_file)).convert('RGB')
        image = torch.tensor(np.array(image))

        boxes = extract_coordinate(path.join(txt_path, f'{name}.txt'))
        map_image = create_map(image.shape[:2], boxes)
        map_image = torch.tensor(map_image)

        tensor_image = torch.cat([image, map_image.unsqueeze(2)], dim = 2)
        tensor_image = tensor_image.permute(2, 0, 1)

        for i in range(n_loop):
            cropedImage = crop(tensor_image)
            im = cropedImage[:3].permute(1, 2, 0).numpy()
            im = im.astype(np.uint8)
            
            Image.fromarray(im).save(path.join(crop_image_path, f'{name}_{i}.jpg'))
            torch.save(cropedImage[3].clone(), path.join(crop_map_image_path, f'{name}_{i}.pt'))

if __name__ == '__main__':
    image_dir = sys.argv[1]
    txt_dir = sys.argv[2]
    crop_dir = sys.argv[3]
    crop_size = int(sys.argv[4])
    n_loop = int(sys.argv[5])
    
    crop_data(image_dir, txt_dir, crop_dir, crop_size, n_loop)
