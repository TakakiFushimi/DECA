# demo_train.py

import torch
from demos.shape_deca import demo_procrustes_loss  # shape_deca から関数をインポート

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 画像のパスを指定
    image_path = 'mywork/001.jpg'  # ここにテストする画像のパスを指定

    # shape_deca.py の関数を使ってデモを実行
    demo_procrustes_loss(device, image_path)

if __name__ == "__main__":
    main()
