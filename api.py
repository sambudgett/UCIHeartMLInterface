from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
from data_utils import read_data_features
from ml_class import MachineLearningEnsemble
from data_visualisation import visualise_data
import numpy as np


class PredictRequest(BaseModel):
    features: Dict[str, float] = {'time': 23, 'ejection_fraction': 38, 'serum_creatinine': 130}


class TrainRequest(BaseModel):
    list_of_features: List[str] = ['time', 'ejection_fraction', 'serum_creatinine']
    data_loc: str = 'data/heart_failure_clinical_records_dataset.csv'
    training_split: float = 0.6
    repeats: int = 10


class TrainResponse(BaseModel):
    accuracy: Optional[float]
    f1_score: Optional[float]
    area_under_curve: Optional[float]
    list_of_accuracies: Optional[Dict[str, float]]
    error: Optional[str]


class ModelResponse(BaseModel):
    features: Dict[str, float]
    predictions: Optional[List[Dict[str, float]]]
    error: Optional[str]


all_predictions = []
app = FastAPI()
ml_model = MachineLearningEnsemble()
data = None


@app.get("/", response_model=ModelResponse)
async def root() -> ModelResponse:
    return ModelResponse(predictions=all_predictions)


@app.get("/visualise/pair")
async def visualise_pair_plot_api() -> FileResponse:
    if ml_model.model_trained_flag:
        visualise_data(ml_model.data_frame, list_of_plots=['pair'],
                       list_of_features=ml_model.list_of_features,
                       label_name="DEATH_EVENT", save_location='test_data_visualisations/')
    return FileResponse('test_data_visualisations/pair_plot.png')


@app.get("/visualise/hist")
async def visualise_pair_plot_api() -> FileResponse:
    if ml_model.model_trained_flag:
        visualise_data(ml_model.data_frame, list_of_plots=['hist'],
                       list_of_features=ml_model.list_of_features,
                       label_name="DEATH_EVENT", save_location='test_data_visualisations/')
    return FileResponse('test_data_visualisations/histograms.png')


@app.post("/predict")
async def get_model_predictions(request: PredictRequest) -> ModelResponse:
    """
    gets model predictions from trained model
    :param request:
    :return:
    """
    if ml_model.model_trained_flag:
        query_data = []
        for feature_name in ml_model.list_of_features:
            # check feature has been supplied, if not throw error
            if feature_name in request.features:
                query_data.append(request.features[feature_name])
            else:
                return ModelResponse(
                    error=f'{feature_name} has not been found in input. '
                    f'Please provide values for the full feature list used in training: {ml_model.list_of_features}'
                )
        query_data = np.expand_dims(np.asarray(query_data), axis=0)
        print(query_data.shape)
        predictions, combined_score = ml_model.predict(query_data)
        predictions_out = {'Combined_predictions': combined_score, 'non-linear-svm': predictions[0],
                           'decision-tree': predictions[1], 'random-forest': predictions[2],
                           'ada-boost': predictions[3], 'mlp': predictions[4], 'log_reg': predictions[5],
                           'k-nn': predictions[6]}
        all_predictions.append({"features": request.features,
                                "predictions": predictions_out})
        return ModelResponse(predictions=all_predictions)

    else:
        return ModelResponse(
            error="Model has not been trained, please train the ML models before prediction can happen"
        )


@app.post("/train")
async def train_model(request: TrainRequest) -> TrainResponse:
    """
    Trains models with data from path
    :param request: features to use and data location
    :return:
    """
    area_under_curve, accuracy, f1_score, mcc_score, fpr, tpr, component_accuracies, component_f1_scores, \
        component_mcc_scores = ml_model.initialise_model_from_data_file(request.list_of_features, request.data_loc,
                                                                        training_split=request.training_split)

    return TrainResponse(accuracy=accuracy, f1_score=f1_score, area_under_curve=area_under_curve)


@app.post("/evaluate")
async def evaluate_model(request: TrainRequest) -> TrainResponse:
    """
    Trains models with data from path
    :param request: features to use and data location
    :return:
    """
    test_input_data = read_data_features(request.list_of_features, data_path=request.data_loc)[0].T
    test_target = np.squeeze(read_data_features(["DEATH_EVENT"])[0])
    list_area_under_curve, list_accuracy, list_f1_score, list_mcc_score, list_fpr, list_tpr\
        = ml_model.average_results(test_input_data, test_target, repeats=request.repeats,
                                   plot_save_loc='./results/overall_roc_curve.png')
    return TrainResponse(accuracy=np.mean(list_accuracy), f1_score=np.mean(list_f1_score),
                         area_under_curve=np.mean(list_area_under_curve))

