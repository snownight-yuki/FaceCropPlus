import os
from tkinter import Tk, Canvas, Button, Label, filedialog, Frame
from PIL import Image, ImageTk, ImageFilter
from ultralytics import YOLO
import torch
import numpy as np
import cv2
from basicsr.archs.rrdbnet_arch import RRDBNet

# Real-ESRGANer のインポート
try:
    from realesrgan import RealESRGANer
except ImportError:
    print("RealESRGANer モジュールが見つかりません。必ず公式リポジトリからセットアップしてください。")
    RealESRGANer = None  # アップスケール処理をスキップするための対策

class ImageCropperWithFaceDetection:
    def __init__(self, root):
        self.root = root
        self.root.title("Accurate Crop Tool with Face Detection")

        # Canvasとコントロール用フレームの作成
        self.canvas = Canvas(root, bg="gray")
        self.canvas.pack(fill="both", expand=True)

        self.control_frame = Frame(root, bg="white", height=50)
        self.control_frame.pack(fill="x", side="bottom")

        # ボタンとステータスラベル
        self.open_folder_btn = Button(self.control_frame, text="Open Folder", command=self.open_folder)
        self.save_btn = Button(self.control_frame, text="Crop & Save", command=self.crop_selected_face, state="disabled")
        self.status_label = Label(self.control_frame, text="No folder selected", bg="white")

        self.open_folder_btn.pack(side="left", padx=10)
        self.save_btn.pack(side="left", padx=10)
        self.status_label.pack(side="left", padx=10)

        # 属性
        self.image_list = []
        self.current_image_index = 0
        self.original_image = None
        self.tk_image = None
        self.image_offset = [0, 0]
        self.scale = 1.0
        self.output_size = 1024  # 最終出力サイズ
        self.face_boxes = []  # 顔検出の座標リスト（画像内座標）
        self.selected_face_index = None

        # ドラッグ操作用
        self.drag_start_x = 0
        self.drag_start_y = 0

        # クロップ枠移動用の属性
        self.is_moving_crop = False
        self.crop_drag_start_x = 0
        self.crop_drag_start_y = 0
        self.original_box_coords = None  # 移動開始時の枠の位置（画像内座標）

        # リサイズ状態の属性
        self.is_resizing = False
        self.resize_handle_size = 10
        self.active_handle_index = None   # 0:左上, 1:右上, 2:左下, 3:右下
        self.fixed_point = None            # リサイズ中に固定する対角の点（画像内座標）

        # YOLOモデルのロード
        self.model_path = os.path.join(os.getcwd(), "yolov11n-face.pt")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}. Please place the YOLOv11 model in the current directory.")
        self.model = YOLO(self.model_path)

        # マウスイベントのバインド
        self.canvas.bind("<MouseWheel>", self.on_zoom)
        self.canvas.bind("<ButtonPress-1>", self.on_drag_start)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)

    def open_folder(self):
        """フォルダを選択し、最初の画像をロード"""
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.image_list = [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith(('png', 'jpg', 'jpeg'))
            ]
            self.current_image_index = 0
            if self.image_list:
                os.makedirs("output", exist_ok=True)
                self.load_image()
                self.update_ui(folder_loaded=True)
            else:
                self.update_ui(folder_loaded=False)

    def update_ui(self, folder_loaded=False):
        if folder_loaded:
            self.status_label.config(text=f"{len(self.image_list)} images loaded.")
            self.save_btn.config(state="normal" if self.face_boxes else "disabled")
        else:
            self.status_label.config(text="No folder selected")
            self.save_btn.config(state="disabled")
        self.root.update_idletasks()

    def load_image(self):
        if self.current_image_index < len(self.image_list):
            image_path = self.image_list[self.current_image_index]
            self.original_image = Image.open(image_path).convert("RGBA")
            self.image_offset = [0, 0]
            self.scale = 1.0
            self.detect_faces(image_path)
        else:
            self.update_ui(folder_loaded=False)

    def display_image(self):
        if self.original_image is None:
            return
        self.canvas.delete("all")
        scaled_width = int(self.original_image.width * self.scale)
        scaled_height = int(self.original_image.height * self.scale)
        resized_image = self.original_image.resize((scaled_width, scaled_height), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized_image)
        self.canvas.create_image(
            self.image_offset[0], self.image_offset[1],
            image=self.tk_image, anchor="nw", tags="image"
        )
        # 顔検出枠の描画（青枠）
        for idx, (x1, y1, x2, y2) in enumerate(self.face_boxes):
            self.draw_face_box(idx, x1, y1, x2, y2)
        # 選択領域（赤枠とリサイズハンドル）の描画
        if self.selected_face_index is not None:
            x1, y1, x2, y2 = self.face_boxes[self.selected_face_index]
            self.canvas.create_rectangle(
                round(x1 * self.scale + self.image_offset[0]),
                round(y1 * self.scale + self.image_offset[1]),
                round(x2 * self.scale + self.image_offset[0]),
                round(y2 * self.scale + self.image_offset[1]),
                outline="red", width=3, tags="highlight"
            )
            self.add_resize_handles(
                x1 * self.scale + self.image_offset[0],
                y1 * self.scale + self.image_offset[1],
                x2 * self.scale + self.image_offset[0],
                y2 * self.scale + self.image_offset[1]
            )

    def detect_faces(self, image_path):
        """顔を検出してCanvasに描画"""
        self.face_boxes = []
        self.selected_face_index = None
        self.display_image()  # 画像描画

        # 顔検出
        results = self.model.predict(image_path, conf=0.25)
        detections = results[0].boxes.xyxy.tolist() if results else []

        if detections:
            for idx, (x1, y1, x2, y2) in enumerate(detections):
                face_height = y2 - y1
                padding_top = face_height * 0.3
                padding_bottom = face_height * 0.3
                padding_left = face_height * 0.3  # 左側は広めに
                padding_right = face_height * 0.3    # 右側は狭めに

                x1 -= padding_left * 1.8
                y1 -= padding_top * 1.8
                x2 += padding_right
                y2 += padding_bottom

                # 正方形に補正（長い方に合わせる）
                side = max(x2 - x1, y2 - y1)
                self.face_boxes.append((x1, y1, x1 + side, y1 + side))
                self.draw_face_box(idx, x1, y1, x1 + side, y1 + side)

            # 検出結果が1件のみなら自動選択
            if len(detections) == 1:
                self.selected_face_index = 0
                self.display_image()
        else:
            self.add_default_box()

    def draw_face_box(self, idx, x1, y1, x2, y2):
        x1_canvas = round(x1 * self.scale + self.image_offset[0])
        y1_canvas = round(y1 * self.scale + self.image_offset[1])
        x2_canvas = round(x2 * self.scale + self.image_offset[0])
        y2_canvas = round(y2 * self.scale + self.image_offset[1])
        self.canvas.create_rectangle(
            x1_canvas, y1_canvas, x2_canvas, y2_canvas,
            outline="blue", width=2, tags=f"visual_{idx}"
        )
        self.canvas.create_rectangle(
            x1_canvas, y1_canvas, x2_canvas, y2_canvas,
            outline="", fill="", tags=f"clickable_{idx}"
        )
        self.canvas.tag_bind(f"clickable_{idx}", "<Button-1>", lambda event, i=idx: self.select_face(i))

    def add_default_box(self):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        # デフォルトは正方形512x512
        x1_canvas = (canvas_width - 512) // 2
        y1_canvas = (canvas_height - 512) // 2
        x2_canvas = x1_canvas + 512
        y2_canvas = y1_canvas + 512
        # ※ここではキャンバス座標をそのまま画像内座標とする（scale=1, offset=0想定）
        self.face_boxes = [(x1_canvas, y1_canvas, x2_canvas, y2_canvas)]
        self.selected_face_index = 0
        self.canvas.create_rectangle(
            x1_canvas, y1_canvas, x2_canvas, y2_canvas,
            outline="red", width=3, tags="default"
        )
        self.canvas.tag_bind("default", "<Button-1>", lambda event: self.select_face(0))

    def select_face(self, index):
        self.selected_face_index = index
        self.display_image()

    def add_resize_handles(self, x1, y1, x2, y2):
        handle_size = self.resize_handle_size
        handles = [
            (x1, y1),  # 左上
            (x2, y1),  # 右上
            (x1, y2),  # 左下
            (x2, y2)   # 右下
        ]
        for idx, (hx, hy) in enumerate(handles):
            tag = f"resize_handle_{idx}"
            self.canvas.create_rectangle(
                hx - handle_size, hy - handle_size,
                hx + handle_size, hy + handle_size,
                fill="red", outline="black", tags=tag
            )
            self.canvas.tag_bind(tag, "<ButtonPress-1>", lambda event, i=idx: self.start_resize(i))
            self.canvas.tag_bind(tag, "<ButtonRelease-1>", self.end_resize)

    def start_resize(self, handle_index):
        """リサイズ開始時に、操作しているハンドルの対角を固定する"""
        if self.selected_face_index is None:
            return
        self.is_resizing = True
        self.active_handle_index = handle_index
        x1, y1, x2, y2 = self.face_boxes[self.selected_face_index]
        if handle_index == 0:        # 左上を操作 → 右下固定
            self.fixed_point = (x2, y2)
        elif handle_index == 1:      # 右上 → 左下固定
            self.fixed_point = (x1, y2)
        elif handle_index == 2:      # 左下 → 右上固定
            self.fixed_point = (x2, y1)
        elif handle_index == 3:      # 右下 → 左上固定
            self.fixed_point = (x1, y1)

    def end_resize(self, event):
        self.is_resizing = False
        self.active_handle_index = None
        self.fixed_point = None

    def on_drag_start(self, event):
        # クリック位置が赤い選択枠内かチェックし、枠移動モードに切り替え
        if not self.is_resizing and self.selected_face_index is not None:
            x1, y1, x2, y2 = self.face_boxes[self.selected_face_index]
            box_left = x1 * self.scale + self.image_offset[0]
            box_top = y1 * self.scale + self.image_offset[1]
            box_right = x2 * self.scale + self.image_offset[0]
            box_bottom = y2 * self.scale + self.image_offset[1]
            if box_left <= event.x <= box_right and box_top <= event.y <= box_bottom:
                self.is_moving_crop = True
                self.crop_drag_start_x = event.x
                self.crop_drag_start_y = event.y
                self.original_box_coords = (x1, y1, x2, y2)
                return
        # 画像移動処理
        self.drag_start_x = event.x
        self.drag_start_y = event.y

    def on_drag(self, event):
        if self.is_resizing and self.active_handle_index is not None and self.selected_face_index is not None:
            current_img_x = (event.x - self.image_offset[0]) / self.scale
            current_img_y = (event.y - self.image_offset[1]) / self.scale
            fixed_x, fixed_y = self.fixed_point
            dx = current_img_x - fixed_x
            dy = current_img_y - fixed_y
            side = max(abs(dx), abs(dy))
            new_x = fixed_x + (side if dx >= 0 else -side)
            new_y = fixed_y + (side if dy >= 0 else -side)
            x1_new = min(fixed_x, new_x)
            y1_new = min(fixed_y, new_y)
            x2_new = max(fixed_x, new_x)
            y2_new = max(fixed_y, new_y)
            min_size = 20
            if (x2_new - x1_new) < min_size or (y2_new - y1_new) < min_size:
                return
            self.face_boxes[self.selected_face_index] = (x1_new, y1_new, x2_new, y2_new)
            self.display_image()
        elif self.is_moving_crop and self.selected_face_index is not None:
            dx = (event.x - self.crop_drag_start_x) / self.scale
            dy = (event.y - self.crop_drag_start_y) / self.scale
            orig_x1, orig_y1, orig_x2, orig_y2 = self.original_box_coords
            self.face_boxes[self.selected_face_index] = (orig_x1 + dx, orig_y1 + dy, orig_x2 + dx, orig_y2 + dy)
            self.display_image()
        else:
            dx = event.x - self.drag_start_x
            dy = event.y - self.drag_start_y
            self.image_offset[0] += dx
            self.image_offset[1] += dy
            self.drag_start_x = event.x
            self.drag_start_y = event.y
            self.display_image()

    def on_mouse_release(self, event):
        if self.is_resizing:
            self.end_resize(event)
        if self.is_moving_crop:
            self.is_moving_crop = False
            self.original_box_coords = None

    def on_zoom(self, event):
        zoom_factor = 1.1 if event.delta > 0 else 0.9
        self.scale *= zoom_factor
        self.display_image()

    def crop_selected_face(self):
        """選択した顔領域をクロップし、RealESRGANer でアップスケール後、最終出力サイズにダウンスケールして保存"""
        if self.selected_face_index is None or self.original_image is None:
            return

        # 1. クロップ（画像内座標で切り抜く）
        x1, y1, x2, y2 = self.face_boxes[self.selected_face_index]
        cropped_image = self.original_image.crop((int(x1), int(y1), int(x2), int(y2)))
        w, h = cropped_image.size
        # 2. 前処理
        new_image = np.array(cropped_image, dtype=np.uint8)
        preprocessed_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)

        # 3. アップスケール処理
        if RealESRGANer is not None and w < 300:
            try:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                # ここではモデルスケール4の例。モデルの重みファイルパスを適宜指定してください。
                model_path = os.path.join("", "RealESRGAN_x4plus.pth")
                # モデルの定義
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                               num_block=23, num_grow_ch=32, scale=4)
                netscale = 4
                # RealESRGANer を利用してアップスケール
                upscaler = RealESRGANer(
                    scale=netscale,
                    model_path=model_path,
                    model=model,
                    device=device,
                    half=False  # GPU があれば半精度を利用する設定
                )
                upscaled_image = upscaler.enhance(preprocessed_image, outscale=netscale)[0]
                cropped_image = Image.fromarray(cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2RGB))
            except Exception as e:
                print("アップスケール処理に失敗しました。", e)
                upscaled_image = cropped_image
        else:
            upscaled_image = cropped_image

        # 4. 最終出力サイズに合わせてリサイズ（ここではBICUBIC補間でダウンスケール）
        final_image = cropped_image.resize((self.output_size, self.output_size), Image.BICUBIC)

        # 5. 保存処理
        output_path = os.path.join("output", f"cropped_{self.current_image_index + 1}.png")
        final_image.save(output_path)
        print(f"Saved: {output_path}")

        # 次の画像へ
        self.current_image_index += 1
        if self.current_image_index < len(self.image_list):
            self.load_image()
        else:
            self.update_ui(folder_loaded=False)

if __name__ == "__main__":
    root = Tk()
    root.geometry("1200x800")
    app = ImageCropperWithFaceDetection(root)
    root.mainloop()
