from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
import cv2
from deepface import DeepFace
import numpy as np
import os
import tempfile
from collections import Counter

app = FastAPI()

models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]
metrics = ["cosine", "euclidean", "euclidean_l2"]

def recognize_from_image(image_path: str, model_name: str, metric: str):
    try:
        results = DeepFace.find(
            img_path=image_path,
            db_path="Data/",
            model_name=model_name,
            distance_metric=metric,
            enforce_detection=False
        )

        recognized_people = [
            os.path.dirname(person['identity'][0])  # Extract folder name
            for person in results
        ] if results else []
        
        return recognized_people

    except Exception as e:
        return {"error": f"Error in face recognition: {str(e)}"}

@app.post("/recognize/")
async def recognize_face(
    files: list[UploadFile] = File(...),
    model_name: str = Query("Facenet512", enum=models),
    metric: str = Query("euclidean", enum=metrics)
):
    try:
        folder_results = []
        temp_paths = []

        # Save each uploaded image temporarily and perform recognition
        for file in files:
            image_data = await file.read()
            image_array = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(image_data)
                temp_path = temp_file.name
                temp_paths.append(temp_path)

            recognized_folders = recognize_from_image(temp_path, model_name, metric)
            if recognized_folders:
                folder_results.extend(recognized_folders)
            else:
                folder_results.append(f"No faces detected in {file.filename}")

        for temp_path in temp_paths:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        if folder_results:
            most_common_folder = Counter(folder_results).most_common(1)
            final_folder = most_common_folder[0][0] if most_common_folder else None
            name = final_folder.split("/")[-1] if final_folder else None
            return JSONResponse(content={"name": name})
        else:
            return JSONResponse(content={"message": "Unkwnown"})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
