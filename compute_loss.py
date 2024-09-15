import torch
import os
from shape_deca import initialize_deca, load_and_preprocess_image, get_mesh_data, compute_procrustes_loss

# 2つの画像間のProcrustesロスを計算する関数
def calculate_loss_between_images(source_path, generate_path, target_path, device, result_file):
    # DECAモデルを初期化
    deca = initialize_deca(device)

    # 画像1をロードして前処理（顔検出とクロップを含む）
    image_tensor1 = load_and_preprocess_image(source_path, device)

    # 画像2をロードして前処理（顔検出とクロップを含む）
    image_tensor2 = load_and_preprocess_image(generate_path, device)

    # 画像1と画像2のメッシュデータを取得
    source_mesh_data = get_mesh_data(deca, image_tensor1)
    generated_mesh_data = get_mesh_data(deca, image_tensor2)

    # Procrustesロスを計算
    disparity = compute_procrustes_loss(source_mesh_data, generated_mesh_data)
    
    # 結果を表示とファイル保存
    result_message = f'Procrustes Loss between {source_path} and {generate_path} from {target_path}: {disparity}\n'
    print(result_message)
    result_file.write(result_message)
    
    # ロスを返す
    return disparity

def get_image_list(folder_path):
    # 指定フォルダ内のすべての画像ファイルのパスを取得
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

if __name__ == "__main__":
    # 使用するデバイスを指定（GPUがあればGPU、なければCPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # フォルダのパスを指定
    generate_folder = '../ghost-train/examples/results/id200-enhanced8/crop/'
    source_folder = '../ghost-train/examples/quantitative-evaluation/source_images/crop/'
    target_folder = '../ghost-train/examples/quantitative-evaluation/target_images/'

    # フォルダ内のすべての画像をリストに取得
    generate_list = sorted(get_image_list(generate_folder))
    source_list = get_image_list(source_folder)
    target_list = get_image_list(target_folder)

    total_loss = 0  # 全体のロスを計算するための変数
    count = 0  # 画像ペアの総数をカウント

    # 結果を保存するファイルを開く
    with open('procrustes_loss_results_id200.txt', 'w') as result_file:
        result_file.write("Procrustes Loss Results:\n")

        # すべてのリストの長さを最小に合わせる（リスト間の不整合を回避）
        min_length = min(len(source_list), len(generate_list), len(target_list))

        # source 画像を固定し、すべての target 画像との組み合わせを実行
        for j, source_path in enumerate(source_list):
            for i, target_path in enumerate(target_list):
            # 2つの画像間のProcrustesロスを計算

                # 2つの画像間のProcrustesロスを計算し、結果ファイルに保存
                loss = calculate_loss_between_images(source_path, generate_list[count], target_path, device, result_file)
                total_loss += loss  # ロスを合計
                count += 1  # カウントを増やす

        # 全体の平均ロスを計算
        if count > 0:
            average_loss = total_loss / count
            print(f'Average Procrustes Loss for all image pairs: {average_loss}')
            result_file.write(f'\nAverage Procrustes Loss: {average_loss}\n')
        else:
            print("No image pairs were processed.")
            result_file.write("No image pairs were processed.\n")
