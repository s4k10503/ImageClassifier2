import os
import json
import requests
import urllib.request
import shutil


def download_json():
    """
    Download and save the ImageNet class index JSON from a specified URL.

    This function retrieves a JSON file containing the ImageNet class index from
    a predefined URL and saves it to a local directory.

    The JSON file is saved in the './json' directory. If the directory does not
    exist, it is created.
    """

    json_url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    json_folder = "./json"

    if not os.path.exists(json_folder):
        os.makedirs(json_folder)

    response = requests.get(json_url)
    data = response.json()

    json_path = os.path.join(json_folder, "imagenet_class_index.json")

    with open(json_path, "w") as json_file:
        json.dump(data, json_file)

    print(f"JSON file saved to {json_path}")


def download_model(model_url, model_name):
    """
    Download and save a model file from a specified URL.

    This function downloads a model file from the given URL and saves it to a
    local directory.

    Args:
        model_url (str): The URL from where the model file is to be downloaded.
        model_name (str): The name of the file under which the model is saved.

    The model file is saved in the './model' directory. If the directory does not
    exist, it is created.
    """

    model_folder = "./model"

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    model_path = os.path.join(model_folder, model_name)

    with urllib.request.urlopen(model_url) as response, open(model_path, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)

    print(f"ONNX model {model_name} saved to {model_path}")


def get_github_files(repo, path):
    """
    Retrieve a list of ONNX file names from a specified GitHub repository and path.

    This function fetches file names from a given GitHub repository and path, filtering
    for files with an '.onnx' extension.

    Args:
        repo (str): The GitHub repository from which to fetch file names.
        path (str): The path within the repository to search for files.

    Returns:
        list: A list of file names with an '.onnx' extension found at the specified path.
    """
    
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    response = requests.get(url)
    files = response.json()
    return [file['name'] for file in files if file['name'].endswith('.onnx')]


if __name__ == "__main__":
    download_json()

    repo = "onnx/models"
    path = "vision/classification/resnet/model"
    base_url = f"https://github.com/{repo}/raw/main/{path}/"

    onnx_files = get_github_files(repo, path)

    for onnx_file in onnx_files:
        download_model(f"{base_url}{onnx_file}", onnx_file)