import cv2
from insightface.app import FaceAnalysis
import torch
import random
import requests
import os
import re
import json
import base64
from io import BytesIO
from datetime import datetime
from diffusers import StableDiffusionXLPipeline, DPMSolverSDEScheduler
from PIL import Image, ImageTk
from assets.ip_adapter.ip_adapter_faceid import IPAdapterFaceIDXL
import customtkinter
from huggingface_hub import hf_hub_download
from tkinter import filedialog


class GoobWorkshop(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.title("Goob Workshop")
        self.geometry("1300x1040")
        self.grid_columnconfigure((0, 1, 2, 3), weight=1)
        self.output_images = None
        self.negative_prompt = ""
        self.last_artist = None

        self.download_sdxl_models()
        self.sdxl_scheduler = DPMSolverSDEScheduler()
        self.sdxl_pipe = StableDiffusionXLPipeline.from_single_file("assets/zavychromaxl_v30.safetensors", torch_dtype=torch.float16,
                                                  scheduler=self.sdxl_scheduler, add_watermarker=False, use_safetensors=True,
                                                  low_cpu_mem_usage=True)
        self.ip_pipe = IPAdapterFaceIDXL(self.sdxl_pipe, "assets/ip-adapter-faceid_sdxl.bin", "cuda")
        self.face_embeds = self.make_face_embeds()

        placeholder_image = Image.new("RGB", (1024, 1024), color="#181818")
        self.display_photo = customtkinter.CTkImage(light_image=placeholder_image, dark_image=placeholder_image, size=(1024, 1024))
        self.image_label = customtkinter.CTkLabel(self, text="")  # Label to display the image
        self.image_label.grid(row=0, column=0, rowspan=8)
        self.image_label.configure(image=self.display_photo)

        self.prompt_label = customtkinter.CTkLabel(self, text="Prompt:")
        self.prompt_label.grid(row=0, column=1, padx=5, pady=5, sticky="nw", columnspan=2)
        self.prompt_entry = customtkinter.CTkTextbox(self, height=180)
        self.prompt_entry.insert("1.0", "A goblin man")
        self.prompt_entry.grid(row=0, column=1, padx=5, pady=5, sticky="sew", columnspan=2)

        self.negative_prompt_label = customtkinter.CTkLabel(self, text="Negative Prompt:")
        self.negative_prompt_label.grid(row=1, column=1, padx=5, pady=5, sticky="nw", columnspan=2)
        self.negative_prompt_entry = customtkinter.CTkTextbox(self, height=180)
        self.negative_prompt_entry.insert("1.0", self.negative_prompt)
        self.negative_prompt_entry.grid(row=1, column=1, padx=5, pady=5, sticky="sew", columnspan=2)

        self.artist_button = customtkinter.CTkButton(self, text="Random Artist", command=self.random_artist)
        self.artist_button.grid(row=2, column=1, padx=5, pady=5, sticky="ew", columnspan=2)

        self.generate_button = customtkinter.CTkButton(self, text="Generate", command=self.generate)
        self.generate_button.grid(row=3, column=1, padx=5, pady=5, sticky="ew", columnspan=2)

        self.bind("<Return>", self.generate)
        self.image_label.bind("<Button-3>", self.copy_image_to_clipboard)

        self.load_image_button = customtkinter.CTkButton(self, text="Select new face", command=self.load_new_face)
        self.load_image_button.grid(row=4, column=1, padx=5, pady=5, sticky="ew", columnspan=2)

    def load_new_face(self):
        imagepath = filedialog.askopenfilename()
        if imagepath:
            lighty_image = cv2.imread(imagepath)
            face_pipe = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider'])
            face_pipe.prepare(ctx_id=0, det_size=(640, 640))
            faces = face_pipe.get(lighty_image)
            self.face_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)


    def copy_image_to_clipboard(self, event):
        if self.output_images and len(self.output_images) > 0:

            # Convert PIL image to base64
            pil_image = self.output_images[0]
            image_buffer = BytesIO()
            pil_image.save(image_buffer, format="PNG")
            base64_image = base64.b64encode(image_buffer.getvalue()).decode("utf-8")

            # Format the base64 data with PNG markup
            base64_data = f"data:image/png;base64,{base64_image}"

            # Copy the base64 data to the clipboard
            self.clipboard_clear()
            self.clipboard_append(base64_data)
            self.update()

    def random_artist(self):
        if self.last_artist is not None:
                prompt = self.prompt_entry.get("1.0", "end-1c").replace(f'. {self.last_artist}', '')
                self.prompt_entry.delete("0.0", "end")
                self.prompt_entry.insert("1.0", prompt)
        self.last_artist = self.get_random_artist_prompt()
        prompt = f'{self.prompt_entry.get("1.0", "end-1c")}. {self.last_artist}'
        self.prompt_entry.delete("0.0", "end")
        self.prompt_entry.insert("1.0", prompt)



    def generate(self, event=None):

        prompt = self.prompt_entry.get("1.0", "end-1c")
        negative_prompt = self.negative_prompt_entry.get("1.0", "end-1c")
        self.output_images = self.ip_pipe.generate(prompt=prompt, negative_prompt=negative_prompt, faceid_embeds=self.face_embeds, num_samples=1,
                           width=1024, height=1024, num_inference_steps=50, guidance_scale=7,
                           seed=random.randint(-2147483648, 2147483647))
        self.display_photo = customtkinter.CTkImage(light_image=self.output_images[0], dark_image=self.output_images[0], size=(1024, 1024))
        self.image_label.configure(image=self.display_photo)
        self.save_image()

    def make_face_embeds(self):
        lighty_image = cv2.imread("assets/image.png")
        face_pipe = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider'])
        face_pipe.prepare(ctx_id=0, det_size=(640, 640))
        faces = face_pipe.get(lighty_image)
        return torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)

    def download_sdxl_models(self):
        if not os.path.exists("assets/zavychromaxl_v30.safetensors"):
            hf_hub_download(repo_id="misri/zavychromaxl_v30", filename="zavychromaxl_v30.safetensors", local_dir="assets", local_dir_use_symlinks=False)
        if not os.path.exists("assets/ip-adapter-faceid_sdxl.bin"):
            hf_hub_download(repo_id="h94/IP-Adapter-FaceID", filename="ip-adapter-faceid_sdxl.bin", local_dir="assets", local_dir_use_symlinks=False)

    def save_image(self):
        prompt = self.prompt_entry.get("1.0", "end-1c")
        sanitized_prompt = re.sub(r'[^\w\s\-.]', '', prompt.replace('\n', ''))[:100]
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"outputs/{current_datetime}-{sanitized_prompt}.png"
        self.output_images[0].save(filename)

    def get_random_artist_prompt(self):
        with open('assets/artist.json', 'r') as file:
            data = json.load(file)
            selected_artist = random.choice(data)
            return selected_artist.get('prompt')

goobgui = GoobWorkshop()
goobgui.mainloop()
