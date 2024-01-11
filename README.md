<img align="right" src="images/viu_cabecera.webp" width="200px">

# Máster en Inteligencia Artificial <br><br>

## TFM: Comparación de Algoritmos de DL frente a Algoritmos de ML Clásicos <br> Caso: Predicción del incumplimiento de pago de préstamos de la U.S. SBA

[**Alex Castro Gumiel**](https://www.linkedin.com/in/alex-castro-gumiel/)

### Conjunto de Datos

https://www.kaggle.com/datasets/mirbektoktogaraev/should-this-loan-be-approved-or-denied

> **Contexto**

El conjunto de datos es de la Administración de Pequeñas Empresas de EE.UU. (SBA). La SBA de EE.UU. se fundó en 1953 con el principio de promover y ayudar a las pequeñas empresas en el mercado crediticio de EE.UU. Las pequeñas empresas han sido una fuente principal de creación de empleo en los Estados Unidos; por lo tanto, fomentar la formación y el crecimiento de pequeñas empresas tiene beneficios sociales al crear oportunidades laborales y reducir el desempleo.

Ha habido muchas historias de éxito de empresas emergentes que recibieron garantías de préstamos de la SBA, como FedEx y Apple Computer. Sin embargo, también ha habido historias de pequeñas empresas y/o nuevas empresas que han incumplido con sus préstamos garantizados por la SBA.

> **Tarjeta de Datos**

Contiene 899164 instancias y 27 variables.

[Comprensión de los Datos](docs/data_understanding.md)

[Análisis de las Variables](html/sba_national_eda.html)

Conjunto de datos original: "Should This Loan be Approved or Denied?”: A Large Dataset with Class Assignment Guidelines". <br> Por: Min Li, Amy Mickel & Stanley Taylor.

Enlace al artículo: https://doi.org/10.1080/10691898.2018.1434342

### Estructura del Proyecto

    ├── data/
        └── raw/
            └── sba_national.csv                    -> Dataset original completo
            └── sba_train.csv                       -> Dataset original para training
            └── sba_val.csv                         -> Dataset original para validation
            └── sba_test.csv                        -> Dataset original para testing
        └── clean/
            └── data_clean.csv                      -> Dataset limpio y transformado
            └── clean_train.csv                     -> Dataset limpio para training
            └── clean_val.csv                       -> Dataset limpio para validation
            └── clean_test.csv                      -> Dataset limpio para testing
            └── train_subsam.csv                    -> Dataset aplicando SubSampling
            └── train_smote.csv                     -> Dataset aplicando SMOTE
            └── data_results.csv                    -> Muestreo y division de Datos
    ├── docs/
        └── data_understanding.md                   -> Comprensión de los datos
        └── dataset_guidelines.pdf                  -> Pautas del conjunto de datos
    ├── html/
        └── sba_national_eda.html                   -> EDA generado por ProfileReport
    ├── images/
        └── viu_cabecera.webp                       -> Logo de la VIU para cabeceras
    ├── models/
        └── preprocessing.joblib                    -> Pipeline del preprocesamiento
        └── logreg_model.joblib                     -> Modelo de Regresion Logistica
        └── knn_model.joblib                        -> Modelo de K Vecinos Cercanos
        └── dectree_model.joblib                    -> Modelo de Arboles de Decision
        └── ranfor_model.joblib                     -> Modelo de Bosque Aleatorio
        └── xgboost_model.joblib                    -> Modelo XGBoost Clasificador
    ├── notebooks/
        └── transform/
            └── data_exploration.ipynb              -> Análisis Exploratorio de Datos
            └── preprocessing.ipynb                 -> Preprocesamiento de Datos
            └── data_pipeline.ipynb                 -> Pipeline del Preprocesamiento
        └── classifier/
            └── logistic_regression.ipynb           -> Modelo de Regresion Logistica
            └── k-nearest_neighbors.ipynb           -> Modelo de K Vecinos Cercanos
            └── decision_tree.ipynb                 -> Modelo de Arboles de Decision
            └── random_forest.ipynb                 -> Modelo de Bosque Aleatorio
            └── xgb_classifier.ipynb                -> Modelo XGBoost Clasificador
            └── model_stacking.ipynb                -> Meta-Ensamble de Modelos
            └── split_data_sampling.ipynb           -> Muestreo y division de datos
            └── mlp_classifier.ipynb                -> Modelo Multi-layer Perceptron
            └── mlp_ensemble_voting.ipynb           -> MLP Meta-Ensamble por Votacion
        └── clustering/
            └── dim_reduction.ipynb                 -> Reduccion de Dimensionalidad
            └── clustering_models.ipynb             -> Modelos de Agrupamiento
        └── mlops/
            └── mlflow_tracking.ipynb               -> MLflow de Random Forest
            └── fastapi_request.ipynb               -> Consumir FastAPI XGBoost
            └── docker_fastapi.ipynb                -> Consumir Contenedor Docker
    ├── src/
        └── transform/
            └── __init__.py                         -> Convertir directorio en paquete
            └── data_transform.py                   -> Preprocesamiento de variables
            └── data_pipelines.py                   -> Pipelines de preprocesamiento
        └── classifier/
            └── __init__.py                         -> Convertir directorio en paquete
            └── custom_classifier.py                -> Transformaciones personalizadas
            └── classifier_models.py                -> Modelos para Clasificacion
        └── transform_main.py                       -> Transformacion de datasets
        └── logreg_model.py                         -> Modelo de Regresion Logistica
        └── knn_model.py                            -> Modelo de K Vecinos Cercanos
        └── dectree_model.py                        -> Modelo de Arboles de Decision
        └── ranfor_model.py                         -> Modelo de Bosque Aleatorio
        └── xgb_model.py                            -> FastAPI Modelo de XGBoost
    ├── apps/
        └── explore_data.py                         -> Streamlit para explorar datos
        └── predict_data.py                         -> Streamlit para predicciones
    ├── docker/
        └── app/
            └── main.py                             -> Test de Contenedor FastAPI
        └── Dockerfile                              -> Configuracion del Contenedor
        └── requirements.txt                        -> Librerias para el Contenedor
    ├── .gitignore                                  -> Archivos y carpetas ignorados
    ├── config.yaml                                 -> Valores de configuracion
    ├── LICENSE                                     -> Licencia de codigo abierto
    ├── Makefile                                    -> Comandos para actualizaciones
    ├── README.md                                   -> Informacion sobre el proyecto
    ├── requirements.txt                            -> Versiones de librerias necesarias

<!-- ```
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
``` -->
<!-- # . /opt/anaconda3/bin/activate && conda activate /Users/zorromac/.conda/envs/Master_IA -->
