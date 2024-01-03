import shutil
import os
import uuid
import uvicorn
from fastapi import FastAPI, Depends, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from inference import ImageClassifierModel

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_model():
    """
    Load and return an instance of the Image Classifier Model.

    This function loads the ImageClassifierModel using the specified model path
    and labels path. If the model fails to load, it raises an HTTPException.

    Returns:
        ImageClassifierModel: The loaded image classification model.

    Raises:
        HTTPException: If the model fails to load.
    """

    model = ImageClassifierModel()
    model_path = "./models/resnet18-v1-7.onnx"
    labels_path = "./json/imagenet_class_index.json"
    try:
        model.load_model(model_path, labels_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return model


@app.post("/predict")
async def predict(file: UploadFile = File(...),
                  model: ImageClassifierModel = Depends(get_model)):   
    """
    Perform image classification on the uploaded image file and return the results.

    This endpoint performs image classification on the provided image file and
    returns the classification result along with its confidence score.

    Args:
        file (UploadFile): The image file to be classified.
        model (ImageClassifierModel, optional): The model to perform image classification.
            Defaults to Depends(get_model).

    Returns:
        JSONResponse: A JSON response containing the classification result and confidence score.

    Raises:
        HTTPException: If an invalid file type is uploaded or an unexpected error occurs.
    """

    # Check that the uploaded file is an image
    if file.content_type is None or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file type")

    # Generate unique filenames
    filename = f"temp_image_{uuid.uuid4()}.jpg"

    try:
        with open(filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result, confidence = model.perform_inference(filename)
        return JSONResponse(content={"result": result,
                                     "confidence": confidence})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Delete file
        if os.path.exists(filename):
            os.remove(filename)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
