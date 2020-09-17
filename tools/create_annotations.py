import json
import copy

def filter_classes(file_path, supercat_list, output_path, rem_iscrowd=True):

    with open (file_path,'r') as f:
        instances = json.load(f)

    # Filter categories by supercategory
    cat_id = list()
    categories = list()
    for category in instances["categories"]:
        if category["supercategory"] not in supercat_list:
            categories.append(category)
        else:
            cat_id.append(category["id"])

    # Count annotations in an image
    ann_dict = dict()
    for annotation in instances["annotations"]:
        if annotation["image_id"] not in ann_dict:
            ann_dict[annotation["image_id"]] = []
        ann_dict[annotation["image_id"]].append(annotation)

    # Filter images by category
    img_ids = list()
    annotations = list()
    for annotation in instances["annotations"]:
        # Remove annotations with iscrowd, and remove the image if it's the only annotation
        if rem_iscrowd:
            if annotation["category_id"] not in cat_id:
                if len(ann_dict[annotation["image_id"]])==1 and annotation["iscrowd"]==1:
                    img_ids.append(annotation["image_id"])
                elif annotation["iscrowd"]!=1:
                    annotations.append(annotation)
            else:
                img_ids.append(annotation["image_id"])

        else:
            if annotation["category_id"] not in cat_id:
                annotations.append(annotation)
            else:
                img_ids.append(annotation["image_id"])
    print(len(annotations))
    print(len(set(img_ids)))

    images = list()
    for image in instances["images"]:
        if image["id"] not in img_ids:
            images.append(image)

    new_instances = copy.deepcopy(instances)
    new_instances.update({"images": images, "annotations": annotations, "categories": categories})

    with open(output_path + '.json', 'w', encoding='utf8') as json_file:
        json.dump(new_instances, json_file)

def merge_classes(file_path, output_path):

    with open (file_path,'r') as f:
        instances = json.load(f)

    annotations = instances["annotations"]
    for annotation in annotations:
        annotation.update((k, 1) for k, v in annotation.items() if k == "category_id")

    categories = [{"supercategory": "agnostic",
                   "id": 1,
                   "name": "object"}]

    new_instances = copy.deepcopy(instances)
    new_instances.update({"annotations": annotations, "categories": categories})

    with open(output_path + '.json', 'w', encoding='utf8') as json_file:
        json.dump(new_instances, json_file)

if __name__ == "__main__":
    supercat_list = ['person', 'animal']
    ANN_DIR = "/vol/bitbucket/shp2918/coco/annotations/"
    IMG_DIR = "/vol/bitbucket/shp2918/coco/images/"

    filter_classes(ANN_DIR+"instances_val2017.json", supercat_list, ANN_DIR+"instances_filtered_val2017", rem_iscrowd=True)
    merge_classes(ANN_DIR+"instances_val2017.json", ANN_DIR+"instances_merged_val2017")
    merge_classes(ANN_DIR+"instances_filtered_val2017.json", ANN_DIR+"instances_filtered_merged_val2017")

    filter_classes(ANN_DIR+"instances_train2017.json", supercat_list, ANN_DIR+"instances_filtered_train2017", rem_iscrowd=True)
    merge_classes(ANN_DIR+"instances_train2017.json", ANN_DIR+"instances_merged_train2017")
    merge_classes(ANN_DIR+"instances_filtered_train2017.json", ANN_DIR+"instances_filtered_merged_train2017")

