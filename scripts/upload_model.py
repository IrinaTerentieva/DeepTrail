from huggingface_hub import upload_file

upload_file(
    path_or_fileobj="/home/irina/trails-tracks-mapper/HF_models/Trail_UNet_DTM10cm_512px.h5",
    path_in_repo="Trail_UNet_DTM10cm_512px/model.h5",  # name inside the repo
    repo_id="IrroIrro/Trail_UNet",
    repo_type="model",
    commit_message="Upload Keras .h5 model"
)
