import cv2
import numpy as np
import os
import logging
import json

def calculate_single_image_masks_label(mask_file, pred_mask_file, category_list, sam_mask_label_file_name, sam_mask_label_file_dir):
    """
 mask_index, category_id, category_name, category_count, mask_count
    """
    sam_mask_data = np.load(mask_file)
    pred_mask_img = cv2.imread(pred_mask_file)[:,:,-1] # red channel
    shape_size = pred_mask_img.shape[0] * pred_mask_img.shape[1]
    logger = logging.getLogger()
    folder_path = os.path.dirname(pred_mask_file)
    sam_mask_category_folder = os.path.join(folder_path, sam_mask_label_file_dir)
    os.makedirs(sam_mask_category_folder, exist_ok=True)
    mask_category_path = os.path.join(sam_mask_category_folder, sam_mask_label_file_name)
    with open(mask_category_path, 'w') as f:
        f.write("id,category_id,category_name,category_count_ratio,mask_count_ratio\n")
        for i in range(sam_mask_data.shape[0]):
            single_mask = sam_mask_data[i]
            single_mask_labels = pred_mask_img[single_mask]
            unique_values, counts = np.unique(single_mask_labels, return_counts=True, axis=0)
            max_idx = np.argmax(counts)
            single_mask_category_label = unique_values[max_idx]
            count_ratio = counts[max_idx]/counts.sum()

            logger.info(f"{folder_path}/sam_mask/{i} assign label: [ {single_mask_category_label}, {category_list[single_mask_category_label]}, {count_ratio:.2f}, {counts.sum()/shape_size:.4f} ]")
            f.write(f"{i},{single_mask_category_label},{category_list[single_mask_category_label]},{count_ratio:.2f},{counts.sum()/shape_size:.4f}\n")

    f.close()


def predict_sam_label(data_folder, category_txt, img_path,
                      masks_path_name="sam_mask/masks.npy",
                      sam_mask_label_file_name="sam_mask_label.txt",
                      pred_mask_file_name="pred_mask.png",
                      sam_mask_label_file_dir="sam_mask_label"):

    category_lists = []
    with open(category_txt, 'r') as f:
        category_lines = f.readlines()
        category_list = [' '.join(line_data.split('\t')[1:]).strip() for line_data in category_lines]
        f.close()
        category_lists.append(category_list)
    target_id = os.path.basename(img_path).split('.')[0] if img_path else None
    for test_path, category_list in zip(data_folder, category_lists):
        img_ids = os.listdir(test_path)
        for img_id in img_ids:
            if target_id and img_id != target_id:
                continue
            mask_file_path = os.path.join(test_path, img_id, masks_path_name)
            pred_mask_file_path = os.path.join(test_path, img_id, pred_mask_file_name)
            if os.path.exists(mask_file_path) and os.path.exists(pred_mask_file_path):
                calculate_single_image_masks_label(mask_file_path, pred_mask_file_path, category_list, sam_mask_label_file_name, sam_mask_label_file_dir)




def visualization_save(mask, save_dir, img_path, color_list):
    values = set(mask.flatten().tolist())
    final_masks = []
    label = []
    for v in values:
        final_masks.append((mask[:,:] == v, v))
    np.random.seed(42)
    if len(final_masks) == 0:
        return
    h, w = final_masks[0][0].shape[:2]
    result = np.zeros((h, w, 3), dtype=np.uint8) 
    for m, label in final_masks:
        result[m, :] = color_list[label] 
    image = cv2.imread(img_path)
    vis = cv2.addWeighted(image, 0.5, result, 0.5, 0)

    names = ["background", "candy", "egg tart", "french fries", "chocolate", "biscuit", "popcorn", "pudding", "ice cream", "cheese butter", "cake", "wine", "milkshake", "coffee", "juice", "milk", "tea", "almond", "red beans", "cashew", "dried cranberries", "soy", "walnut", "peanut", "egg", "apple", "date", "apricot", "avocado", "banana", "strawberry", "cherry", "blueberry", "raspberry", "mango", "olives", "peach", "lemon", "pear", "fig", "pineapple", "grape", "kiwi", "melon", "orange", "watermelon", "steak", "pork", "chicken duck", "sausage", "fried meat", "lamb", "sauce", "crab", "fish", "shellfish", "shrimp", "soup", "bread", "corn", "hamburg", "pizza", "hanamaki baozi", "wonton dumplings", "pasta", "noodles", "rice", "pie", "tofu", "eggplant", "potato", "garlic", "cauliflower", "tomato", "kelp", "seaweed", "spring onion", "rape", "ginger", "okra", "lettuce", "pumpkin", "cucumber", "white radish", "carrot", "asparagus", "bamboo shoots", "broccoli", "celery stick", "cilantro mint", "snow peas", "cabbage", "bean sprouts", "onion", "pepper", "green beans", "French beans", "king oyster mushroom", "shiitake", "enoki mushroom", "oyster mushroom", "white button mushroom", "salad", "other ingredients"]
    areas = []
    font_size = 0.5 * np.sqrt(h * w / 512 / 512)
    font_thick = 1+int(h*w > 512*512)+int(h*w > 1024*1024)


    real_egg_area = 4*4*3.14
    ratio = 1
    for m, label in final_masks:
        m = m.astype(np.uint8)
        logger = logging.getLogger()
        logger.info(f'label: {label}')
        if label == 0:  # skip background
            continue
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            food_area_ratio = cv2.contourArea(contour)/m.size
            if food_area_ratio < 0.005:
                continue
            # Calculate the centroid of the contour to place the text
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                found = False
                for v, area in areas:
                    if v == label:
                        area += food_area_ratio
                        found = True
                if not found:
                    areas.append((label, food_area_ratio))
                    if names[label] == 'egg':
                        ratio = real_egg_area/food_area_ratio
                cv2.putText(vis, names[label], (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, font_size, [255,255,255], font_thick)
                cv2.drawContours(vis, [contour], -1, [255,255,255], 2)  # -1 indicates drawing all contours
    vis_save_path = os.path.join(save_dir, 'enhance_vis.png')
    cv2.imwrite(vis_save_path, vis)

    area_save_path = os.path.join(save_dir, 'food_area.txt')
    meat = 0
    vege = 0
    food_area_pair = {}
    with open(area_save_path, 'w') as f:
        with open('nutrition.csv', 'r') as csv:
            rows = csv.readlines()
            f.write(rows[0]+'\n')
            for v, area in areas:
                f.write(str(names[v]) + ', ' + str(area*ratio) + ', ')
                food_area_pair[v] = area
                # not sure how to get the number we want
                cols = rows[v].split(',')
                meat += int(cols[15])*area*ratio
                vege += int(cols[16])*area*ratio
                for data in cols[1:14]:
                    f.write(str(float(data)*ratio)+', ')
                f.write("\n\n")
            f.write("meat area : vege area = "+str(meat/(meat+vege)*100)+" : "+str(vege/(meat+vege)*100))
    json_area_save_path = os.path.join(save_dir, 'food_area.json')
    json_string = json.dumps(food_area_pair, indent=4)
    with open(json_area_save_path, 'w') as json_file:
        json_file.write(json_string)





def enhance_masks(data_folder, category_txt, color_list_path, img_path=None, num_class=104, area_thr=0, ratio_thr=0.5, top_k=80,
                  masks_path_name="sam_mask/masks.npy",
                  new_mask_label_file_name="semantic_masks_category.txt",
                  pred_mask_file_name="pred_mask.png",
                  enhance_mask_name='enhance_mask.png',
                  enhance_mask_vis_name='enhance_vis.png',
                  sam_mask_label_file_dir='sam_mask_label'):
        
    predict_sam_label([data_folder], category_txt, img_path, masks_path_name, new_mask_label_file_name, pred_mask_file_name, sam_mask_label_file_dir)
    color_list = np.load(color_list_path)
    color_list[0] = [238, 239, 20]
    target_folder = os.path.basename(img_path).split('.')[0] if img_path else None  # 2023/10/24, I added this line to do single img
    for img_folder in os.listdir(data_folder):
        if img_folder == 'sam_process.log':
            continue
        if target_folder and img_folder != target_folder: # 2023/10/24, I added this line to do single img
            continue
        category_info_path = os.path.join(data_folder, img_folder, sam_mask_label_file_dir, new_mask_label_file_name)
        sam_mask_folder = os.path.join(data_folder, img_folder)
        pred_mask_path = os.path.join(data_folder, img_folder, pred_mask_file_name)
        img_path = os.path.join(data_folder, img_folder, 'input.jpg')
        save_dir = os.path.join(data_folder, img_folder)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, enhance_mask_name)
        vis_save_path = os.path.join(save_dir, enhance_mask_vis_name)

        pred_mask = cv2.imread(pred_mask_path)[:,:,2]
        f = open(category_info_path, 'r')
        category_info = f.readlines()[1:]
        category_area = np.zeros((num_class,))
        f.close()
        for info in category_info:
            label, area = int(info.split(',')[1]), float(info.split(',')[4])
            category_area[label] += area

        category_info = sorted(category_info, key=lambda x:float(x.split(',')[4]), reverse=True)
        category_info = category_info[:top_k]
        
        enhanced_mask = pred_mask
        
        sam_masks = np.load(os.path.join(sam_mask_folder, masks_path_name))
        for info in category_info:
            idx, label, count_ratio, area = info.split(',')[0], int(info.split(',')[1]), float(info.split(',')[3]), float(info.split(',')[4])
            if area < area_thr:
                continue
            if count_ratio < ratio_thr:
                continue
            sam_mask = sam_masks[int(idx)].astype(bool)
            assert (sam_mask.sum()/ (sam_mask.shape[0] * sam_mask.shape[1]) - area) < 1e-4
            enhanced_mask[sam_mask] = label
        cv2.imwrite(save_path, enhanced_mask)
        visualization_save(enhanced_mask, save_dir, img_path, color_list)
