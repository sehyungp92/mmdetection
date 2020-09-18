# Instance Segmentation of NovelObjects in a Conveyor Setting

## Installation

Clone the repository to your local machine:
```
git clone https://github.c!git clone https://github.com/sehyungp92/mmdetection
pip install mmcv==0.6.2 terminaltables
cd mmdetection
python setup.py develop
pip install "git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools"
pip install git+https://github.com/lucasb-eyer/pydensecrf.git
```
If using Colab, also:
```
pip install --upgrade albumentations
```
Unfortunately due to version changes in the middle, the ```imflip``` and ```imflip_``` functions in ```mmcv/image/geometric.py``` need to be changed to the following:

```
def imflip(img, direction='horizontal'):
    """Flip an image horizontally or vertically.
    Args:
        img (ndarray): Image to be flipped.
        direction (str): The flip direction, either "horizontal" or
            "vertical" or "diagonal".
    Returns:
        ndarray: The flipped image.
    """
    assert direction in ['horizontal', 'vertical', 'diagonal']
    if direction == 'horizontal':
        return np.flip(img, axis=1)
    elif direction == 'vertical':
        return np.flip(img, axis=0)
    else:
        return np.flip(img, axis=(0, 1))


def imflip_(img, direction='horizontal'):
    """Inplace flip an image horizontally or vertically.
    Args:
        img (ndarray): Image to be flipped.
        direction (str): The flip direction, either "horizontal" or
            "vertical" or "diagonal".
    Returns:
        ndarray: The flipped image (inplace).
    """
    assert direction in ['horizontal', 'vertical', 'diagonal']
    if direction == 'horizontal':
        return cv2.flip(img, 1, img)
    elif direction == 'vertical':
        return cv2.flip(img, 0, img)
    else:
        return cv2.flip(img, -1, img)
```

## Instructions

We have created ipynbs with more detailed instructions.

Please use:

```supervised_baseline.ipynb``` on how to train the baseline model

```pseudo_labels.ipynb``` on how to generated pseudo labels with the trained teacher models

```annotation_creator.ipynb``` on how to create custom annotations in COCO format

```semi_supervised.ipynb``` on how to train and test the student model
