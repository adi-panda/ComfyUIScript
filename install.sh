#!/bin/bash

# Create the directory if it doesn't exist
mkdir -p models/checkpoints

# URL to download from
anima_url="https://civitai.com/api/download/models/505691"

# Use wget with content disposition to download the file into the specified directory
wget --content-disposition -P models/checkpoints "$anima_url"


# Create the directory if it doesn't exist
mkdir -p models/ipadapter
ipadapter_url="https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors"

wget --content-disposition -P models/ipadapter "$ipadapter_url"

# Create the directory if it doesn't exist
mkdir -p models/clip_vision
clip_vision_url="https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors"

wget --content-disposition -P models/clip_vision "$clip_vision_url"


# Create the directory if it doesn't exist
mkdir -p models/inpaint
inpaint_head="https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/fooocus_inpaint_head.pth"
inpaint_patch="https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v26.fooocus.patch"
big_lama="https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"
mat="https://github.com/Sanster/models/releases/download/add_mat/Places_512_FullData_G.pth"

wget --content-disposition -P models/inpaint "$inpaint_head"
wget --content-disposition -P models/inpaint "$inpaint_patch"
wget --content-disposition -P models/inpaint "$big_lama"
wget --content-disposition -P models/inpaint "$mat"

pip install -r requirements.txt

git clone https://github.com/ltdrdata/ComfyUI-Manager.git custom_nodes/ComfyUI_Manager
git clone https://github.com/WASasquatch/was-node-suite-comfyui custom_nodes/was-node-suite-comfyui
git clone https://github.com/BadCafeCode/masquerade-nodes-comfyui custom_nodes/masquerade-nodes-comfyui
git clone https://github.com/JPS-GER/ComfyUI_JPS-Nodes custom_nodes/ComfyUI_JPS-Nodes
git clone https://github.com/Acly/comfyui-inpaint-nodes custom_nodes/comfyui-inpaint-nodes

