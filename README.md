# FaceCrop+ 使用ガイド

このプログラムでは、以下のことが可能です:

1. **顔検出**: YOLO Faceモデルを使用して、画像内の顔を検出します。
2. **顔領域のクロップ**: 検出した顔領域を選択してクロップできます。
3. **画像の強調処理**: クロップした画像のサイズが300×300ピクセル未満の場合、Real-ESRGANモデルを使用して高解像度にアップスケールします。
4. **出力画像の保存**: 処理された画像を指定のフォルダに保存します。
5. **複数画像の処理**: 同一フォルダ内の画像を連続して処理できます。

このガイドでは、`FaceCropPlus.py`ツールのセットアップと使用方法について説明します。

## 必要条件

以下のPythonパッケージをインストールしてください:

```plaintext
Pillow>=9.0
ultralytics>=8.0.31
realesrgan
opencv-python
numpy==1.26.4
```

上記の要件に加えて、以下のコマンドを実行してCUDAサポート付きのPyTorchをインストールしてください:

```bash
pip3 install torch==1.13.0 torchvision --index-url https://download.pytorch.org/whl/cu117
```

## 推奨されるセットアップ

依存関係を分離するために、仮想環境を使用することを強くお勧めします。以下の手順に従ってください:

1. 仮想環境を作成:

   ```bash
   python3 -m venv myenv
   ```

2. 仮想環境を有効化:

   - Linux/macOS:
     ```bash
     source myenv/bin/activate
     ```
   - Windows:
     ```cmd
     myenv\Scripts\activate
     ```

3. 必要なパッケージをインストール:

   ```bash
   pip install -r requirements.txt
   pip3 install torch==1.13.0 torchvision --index-url https://download.pytorch.org/whl/cu117
   ```

## 必須のセットアップ

1. 必要なモデルをダウンロード:

   - **Real-ESRGANモデル**:
     [Real-ESRGANのGitHubリポジトリ](https://github.com/xinntao/Real-ESRGAN)から`RealESRGAN_x4plus.pth`をダウンロードし、`FaceCropPlus.py`と同じディレクトリに配置してください。

   - **YOLO Faceモデル**:
     [YOLO FaceのGitHubリポジトリ](https://github.com/akanametov/yolo-face)から`yolov11n-face.pt`をダウンロードし、`FaceCropPlus.py`と同じディレクトリに配置してください。

2. スクリプトを実行する前に、両方のファイルが正しい場所にあることを確認してください。

## スクリプトの実行

すべての依存関係とモデルが準備できたら、以下のコマンドでスクリプトを実行してください:

```bash
python FaceCropPlus.py
```

アプリケーションがGUIを起動し、顔検出付きで画像をクロップできます。

## 出力

クロップおよび強調処理された画像は、スクリプトと同じフォルダ内に作成される`output`ディレクトリに保存されます。

---

# FaceCrop+ Usage Guide

This program provides the following functionalities:

1. **Face Detection**: Uses the YOLO Face model to detect faces in images.
2. **Face Cropping**: Allows you to select and crop detected face regions.
3. **Image Enhancement**: If the cropped image size is less than 300×300 pixels, it utilizes the Real-ESRGAN model to upscale it to a higher resolution.
4. **Saving Processed Images**: Saves the processed images to a specified folder.
5. **Batch Processing**: Processes multiple images from the same folder consecutively.

This guide provides instructions for setting up and using the `FaceCropPlus.py` tool.

## Requirements

Ensure the following Python packages are installed:

```plaintext
Pillow>=9.0
ultralytics>=8.0.31
realesrgan
opencv-python
numpy==1.26.4
```

In addition to the above requirements, execute the following command to install PyTorch with CUDA support:

```bash
pip3 install torch==1.13.0 torchvision --index-url https://download.pytorch.org/whl/cu117
```

## Recommended Setup

It is strongly recommended to use a virtual environment to isolate dependencies. Follow these steps:

1. Create a virtual environment:

   ```bash
   python3 -m venv myenv
   ```

2. Activate the virtual environment:

   - On Linux/macOS:
     ```bash
     source myenv/bin/activate
     ```
   - On Windows:
     ```cmd
     myenv\Scripts\activate
     ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   pip3 install torch==1.13.0 torchvision --index-url https://download.pytorch.org/whl/cu117
   ```

## Required Setup

1. Download the required models:

   - **Real-ESRGAN model**:
     Download `RealESRGAN_x4plus.pth` from the [Real-ESRGAN GitHub repository](https://github.com/xinntao/Real-ESRGAN) and place it in the same directory as `FaceCropPlus.py`.

   - **YOLO Face model**:
     Download `yolov11n-face.pt` from the [YOLO Face GitHub repository](https://github.com/akanametov/yolo-face) and place it in the same directory as `FaceCropPlus.py`.

2. Verify that both files are in the correct location before running the script.

## Running the Script

Once all dependencies and models are in place, execute the script with the following command:

```bash
python FaceCropPlus.py
```

The application will launch a GUI for cropping images with face detection.

## Output

Cropped and enhanced images will be saved in an `output` directory created in the same folder as the script.

