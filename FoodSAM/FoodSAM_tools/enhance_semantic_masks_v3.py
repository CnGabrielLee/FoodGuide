import cv2
import numpy as np
import os
import logging
import json

# 引入自定義模組：LLM 建議與資料庫查詢功能
from FoodSAM_tools.llm_advice import generate_advice, open_ollama
from FoodSAM_tools.db_operation import predict_mask_with_data

def calculate_single_image_masks_label(
    mask_file, pred_mask_file, category_list,
    sam_mask_label_file_name, sam_mask_label_file_dir
):
    """
    計算單張影像中每一個 SAM 遮罩對應的語義分割標籤資訊，
    並將「遮罩索引、類別 ID、類別名稱、類別佔遮罩比例、遮罩佔影像比例」
    寫入指定的文字檔。
    """
    # 讀取 SAM 輸出遮罩矩陣 (N, H, W)
    sam_mask_data = np.load(mask_file)
    # 讀取語義分割預測影像的紅色通道 (存放標籤值)
    pred_mask_img = cv2.imread(pred_mask_file)[:, :, -1]
    img_area = pred_mask_img.shape[0] * pred_mask_img.shape[1]

    logger = logging.getLogger()
    folder_path = os.path.dirname(pred_mask_file)
    # 建立用於存放 SAM 遮罩標籤結果的資料夾
    sam_mask_category_folder = os.path.join(folder_path, sam_mask_label_file_dir)
    os.makedirs(sam_mask_category_folder, exist_ok=True)
    mask_category_path = os.path.join(
        sam_mask_category_folder, sam_mask_label_file_name
    )

    with open(mask_category_path, 'w') as f:
        # 寫入檔頭
        f.write("id,category_id,category_name,category_count_ratio,mask_count_ratio\n")

        # 逐一處理每個 SAM 遮罩
        for i in range(sam_mask_data.shape[0]):
            single_mask = sam_mask_data[i]                  # 取得第 i 張遮罩
            single_mask_labels = pred_mask_img[single_mask] # 遮罩範圍內的預測標籤
            # 統計各標籤出現次數
            unique_values, counts = np.unique(single_mask_labels, return_counts=True)
            max_idx = np.argmax(counts)                     # 取出最多次數的標籤
            label_id = unique_values[max_idx]
            # 計算該標籤佔遮罩內的比例、遮罩佔整張影像的比例
            count_ratio = counts[max_idx] / counts.sum()
            mask_ratio = counts.sum() / img_area

            # 寫入一行結果
            f.write(f"{i},{label_id},{category_list[label_id]},"
                    f"{count_ratio:.2f},{mask_ratio:.4f}\n")

def predict_sam_label(
    data_folder, category_txt, img_path,
    masks_path_name="sam_mask/masks.npy",
    sam_mask_label_file_name="sam_mask_label.txt",
    pred_mask_file_name="pred_mask.png",
    sam_mask_label_file_dir="sam_mask_label"
):
    """
    批次呼叫 calculate_single_image_masks_label，
    依據資料夾中文件名稱或單張影像路徑產生對應的 SAM 遮罩標籤文字檔。
    """
    # 讀取類別對照表
    category_lists = []
    with open(category_txt, 'r') as f:
        lines = f.readlines()
        category_list = [
            ' '.join(line.split('\t')[1:]).strip()
            for line in lines
        ]
        category_lists.append(category_list)

    # 若指定單張影像，取得其 ID
    target_id = os.path.basename(img_path).split('.')[0] if img_path else None

    # 逐一處理資料夾及對應類別列表
    for test_path, category_list in zip(data_folder, category_lists):
        for img_id in os.listdir(test_path):
            # 若僅處理指定影像，跳過其他資料夾
            if target_id and img_id != target_id:
                continue

            mask_file_path = os.path.join(test_path, img_id, masks_path_name)
            pred_mask_file_path = os.path.join(test_path, img_id, pred_mask_file_name)
            if os.path.exists(mask_file_path) and os.path.exists(pred_mask_file_path):
                calculate_single_image_masks_label(
                    mask_file_path, pred_mask_file_path,
                    category_list, sam_mask_label_file_name, sam_mask_label_file_dir
                )

def visualization_save(mask, save_dir, img_path, color_list, email):
    """
    根據最終增強遮罩，產生多種可視化結果並存檔：
    - 原圖 + 顏色遮罩半透明
    - 灰階化標註「不建議食用」區域
    - 類別輪廓與文字標註
    同時呼叫 predict_mask_with_data 查詢每個類別是否健康，可進一步交由 LLM 建議。
    """
    # 取得影像所有標籤值集合
    values = set(mask.flatten().tolist())
    # 預先定義「不建議食用」之標籤列表
    not_to_eat = [2,4,9,12,15,17,18,19,21,22,23,24,47,49,50,53,54,55,56,59,61,68,70,97,98,99,100,101]

    # 生成原始影像與空白結果圖
    image = cv2.imread(img_path)
    h, w = image.shape[:2]
    vis = np.zeros_like(image)
    result = np.zeros_like(image)

    # 依標籤分割出各遮罩
    final_masks = [(mask == v, v) for v in values]
    unique_labels = [v for _, v in final_masks]

    # 向資料庫查詢各標籤顏色映射及健康狀態
    mask_dict, healthy, _ = predict_mask_with_data(unique_labels, email)

    # 一一處理每個遮罩區域
    for m, label in final_masks:
        # 灰階化原圖與半透明疊加
        if not healthy and label in not_to_eat:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image[m] = cv2.cvtColor(gray[m], cv2.COLOR_GRAY2BGR)
            vis[m] = cv2.addWeighted(image[m], 0.5, mask_dict[label], 0.5, 0)
        else:
            result[m] = mask_dict[label]
            vis[m] = image[m]

    # 產生彩色邊界與標註
    color_vis = cv2.addWeighted(image, 0.5, result, 0.5, 0)
    # 儲存可視化影像
    cv2.imwrite(os.path.join(save_dir, 'enhance_vis.png'), vis)
    cv2.imwrite(os.path.join(save_dir, 'color_enhance_vis.png'), color_vis)

    # 呼叫 LLM 產生飲食建議
    open_ollama(unique_labels, save_dir, email, {})  # 以預設參數呼叫

def enhance_masks(
    data_folder, category_txt, color_list_path, img_path=None,
    num_class=104, area_thr=0, ratio_thr=0.5, top_k=80, email='xienli1025@gmail.com',
    masks_path_name="sam_mask/masks.npy",
    new_mask_label_file_name="semantic_masks_category.txt",
    pred_mask_file_name="pred_mask.png",
    enhance_mask_name='enhance_mask.png',
    enhance_mask_vis_name='enhance_vis.png',
    sam_mask_label_file_dir='sam_mask_label',
):
    """
    主流程：結合 SAM 遮罩與語義分割結果，輸出最終增強遮罩圖並進行可視化。
    步驟：
    1. 呼叫 predict_sam_label 產生 SAM 遮罩標籤
    2. 讀取顏色表、語義分割預測，排序並篩選前 top_k 個遮罩
    3. 依面積與比例門檻疊加 SAM 遮罩標籤到原預測結果
    4. 儲存最終增強遮罩與可視化圖
    """
    # 1. 產生 SAM 遮罩對應標籤
    predict_sam_label(
        [data_folder], category_txt, img_path,
        masks_path_name, new_mask_label_file_name,
        pred_mask_file_name, sam_mask_label_file_dir
    )

    # 2. 載入顏色列表並讀取資料夾
    color_list = np.load(color_list_path)
    color_list[0] = [238, 239, 20]  # 制定背景顏色
    target_folder = os.path.basename(img_path).split('.')[0] if img_path else None

    for img_folder in os.listdir(data_folder):
        if img_folder == 'sam_process.log' or (target_folder and img_folder != target_folder):
            continue

        base_dir = os.path.join(data_folder, img_folder)
        # 讀取 SAM 遮罩標籤檔與預測結果
        category_info_path = os.path.join(
            base_dir, sam_mask_label_file_dir, new_mask_label_file_name
        )
        pred_mask_path = os.path.join(base_dir, pred_mask_file_name)
        masks_path = os.path.join(base_dir, masks_path_name)
        save_mask_path = os.path.join(base_dir, enhance_mask_name)

        # 讀取並累積每個類別在影像中的面積比
        category_area = np.zeros((num_class,))
        with open(category_info_path, 'r') as f:
            for line in f.readlines()[1:]:
                parts = line.split(',')
                label = int(parts[1]); area = float(parts[4])
                category_area[label] += area

        # 篩選面積最大的 top_k 個遮罩資訊
        sorted_info = sorted(
            open(category_info_path).readlines()[1:],
            key=lambda x: float(x.split(',')[4]),
            reverse=True
        )[:top_k]

        # 3. 疊加符合門檻的 SAM 遮罩到原預測結果
        enhanced_mask = cv2.imread(pred_mask_path)[:, :, 2]
        sam_masks = np.load(masks_path)
        for info in sorted_info:
            idx, label, _, (area_ratio) = (
                info.split(',')[0],
                int(info.split(',')[1]),
                float(info.split(',')[3]),
                float(info.split(',')[4])
            )
            if area_ratio < area_thr or float(info.split(',')[3]) < ratio_thr:
                continue
            mask_bool = sam_masks[int(idx)].astype(bool)
            enhanced_mask[mask_bool] = label

        # 4. 儲存最終遮罩並呼叫可視化
        cv2.imwrite(save_mask_path, enhanced_mask)
        visualization_save(enhanced_mask, base_dir, os.path.join(base_dir, 'input.jpg'),
                           color_list, email)
