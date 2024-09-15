import os
import cv2
import torch
from decalib.datasets import datasets
from decalib.utils import util

def get_image_list(folder_path):
    """ 指定されたフォルダ内の画像ファイルをリストに取得 """
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

def crop_and_save_images(input_folder, output_folder, detector='fan'):
    """ 指定されたフォルダ内の画像を顔検出でクロップし、出力フォルダに保存する """
    
    # 入力フォルダの画像一覧を取得
    image_list = get_image_list(input_folder)
    
    # 出力フォルダを作成
    os.makedirs(output_folder, exist_ok=True)

    # DECAのTestDataを使用して顔を検出・クロップするための準備
    testdata = datasets.TestData(input_folder, iscrop=True, face_detector=detector)

    for i, img_path in enumerate(image_list):
        # 画像のファイル名を取得
        img_name = os.path.basename(img_path)
        
        # DECAの顔検出機能を使用して顔をクロップ
        try:
            # 画像データをテンソルからNumpy配列に変換し、正しい形式に変換
            cropped_img_tensor = testdata[i]['image']  # テンソル形式
            cropped_img_np = util.tensor2image(cropped_img_tensor)  # テンソルを画像に変換 (H, W, C)形式
        except Exception as e:
            print(f"顔の検出に失敗しました: {img_path} - エラー: {e}")
            continue

        # クロップした画像を保存
        save_path = os.path.join(output_folder, img_name)
        cv2.imwrite(save_path, cropped_img_np)
        print(f"保存しました: {save_path}")

if __name__ == "__main__":
    input_folder = "../ghost-train/examples/quantitative-evaluation/source_images/"
    output_folder = "../ghost-train/examples/quantitative-evaluation/source_images/crop/"
    
    crop_and_save_images(input_folder, output_folder, detector='fan')
