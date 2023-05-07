# DL4H - Reproducibility Project

## Paper chosen for replication

The paper [Unifying Heterogeneous Electronic Health Records Systems via Text-Based Code Embedding](https://proceedings.mlr.press/v174/hur22a/hur22a.pdf), authored by Kyunghoon Hur, Jiyoung Lee, Jungwoo Oh, Wesley Price, Younghak Kim, and Edward Choi, was published in the Proceedings of Machine Learning Research of Conference on Health, Inference, and Learning in 2022.

Citation: K. Hur, J. Lee, J. Oh, W. Price, Y. Kim, and E. Choi. Unifying heterogeneous electronic health records systems via text-based code embedding. In Conference on Health, Inference, and Learning, pages 183–203. PMLR, 2022. https://proceedings.mlr.press/v174/hur22a/hur22a.pdf

Git repository of the original paper is https://github.com/hoon9405/DescEmb. 


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

The requirements.txt lists all dependencies required for the replication. 

## Data preparation

First, download the dataset (links can be referred to the README_orig.md):

MIMIC-III

eICU

ccs_multi_dx_tool_2015

icd10cmtoicd9gem


Second, make directory sturcture like below:

```data prep structure
data_input_path
├─ mimic
│  ├─ ADMISSIONS.csv
│  ├─ PATIENTS.csv
│  ├─ ICUSYAYS.csv
│  ├─ LABEVENTES.csv
│  ├─ PRESCRIPTIONS.csv
│  ├─ PROCEDURES.csv
│  ├─ INPUTEVENTS_CV.csv
│  ├─ INPUTEVENTS_MV.csv
│  ├─ D_ITEMDS.csv
│  ├─ D_ICD_PROCEDURES.csv
│  └─ D_LABITEMBS.csv
├─ eicu
│  ├─ diagnosis.csv
│  ├─ infusionDrug.csv
│  ├─ lab.csv
│  ├─ medication.csv
│  └─ patient.csv
├─ ccs_multi_dx_tool_2015.csv
└─ icd10cmtoicd9gem.csv
```

```
data_output_path
├─mimic
├─eicu
├─pooled
├─label
└─fold
```


## Preprocessing

```preprocessing
python3 preprocess_main.py 
    --data_input_path $csv_directory
    --data_output_path $run_ready_directory 
```

```preprocessing supplement
python3 ./preprocess/main_supplement.py 
    --data_input_path './data_input_path/preprocessed' 
    --data_output_path './data_output_path' 
    --train_valid_fold 5
```

Note: train_valid_fold default value is 10, and we set up as 5 to generate train test split folds. Random seed setup in default is 2021. 


## Training and Evaluation

To train the models in the paper, run this command:

```train codeemb model
python3 main.py \
    --distributed_world_size $WORLDSIZE \
    --input_path /path/to/data \
    --model ehr_model \
    --embed_model codeemb \
    --pred_model rnn \
    --data $data \
    --ratio $ratio \
    --value_embed_type $value \
    --task $task
    --n_epochs $n_epochs (default value: 1000)
    --lr $lr (default value: 1e-4)
```

```train descemb model
python3 main.py \
    --disrtibuted_world_size $WORLDSIZE \
    --input_path /path/to/data \
    --model ehr_model \
    --embed_model $descemb \
    --pred_model rnn \
    --data $data \
    --ratio $ratio \
    --value_embed_type $value \
    --task $task
    --n_epochs $n_epochs (default value: 1000)
    --lr $lr (default value: 1e-4)
```


Below is an example: 
```
python3 main.py 
    --distributed_world_size 1 
    --input_path data_output_path 
    --data eicu 
    --task mortality 
    --model ehr_model 
    --embed_model codeemb 
    --pred_model rnn 
    --ratio 100 
    --value_embed_type DSVA
    --n_epochs 10
    --lr 0.01
```

Other configurations will set to be default, which were used in the paper except for n_epochs and lr due to computation power and time restrictions. The n_epochs is set to 10. lr is set to 0.01. 

$data should be set to 'mimic' or 'eicu'

$descemb should be 'descemb_bert' or 'descemb_rnn'

$ratio should be set to one of [10, 30, 50, 70, 100] (default: 100), in the replication, we use 100 only. 

$value should be set to one of ['DSVA', 'VC']. Other values mentioned in the original paper git repository do not work due to value embedding codes do not generate all required data outputs to be used in the model. 

$task should be set to one of ['readmission', 'mortality', 'los_3day', 'los_7day', 'diagnosis']. In the replication, we tested mortality only. 

Note that --input-path should be the root directory containing preprocessed data.

## Running on AWS SageMaker

The above commands can also be run on AWS SageMaker. The steps below assumes you have the appropriate permissions set up along with AWS CLI and Docker installed locally.

To run pre-processing, `cd` into `preprocess` and run `./build_and_push` to create a docker image and push it to Amazon ECR. This image can then be used to kick off a SageMaker processing job with the same arguments as the example under the Preprocessing section.

To run training, run `./build_and_push` at the top level to create another docker image and push it to ECR. This image can then be used to kick off a SageMaker training job with the same arguments as the examples under the Training and Evaluation section above.

Refer to the [SageMaker documentation](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ProcessingJob.html) for more details on how to structure the S3 inputs and outputs.

## Pre-trained Models

Pre-trained models do not work based on the original code. 

## Results

The models achieve the following performance (after 10 epochs):

Train

| Task       | Value Embedding  | Model        |  Loss    | AUROC    | AUPRC   |
| -----------|------------------|------------- | -------- | -------- | ------- |
| mortality  | DSVA             | CodeEmb      | 0.266    | 0.498    | 0.088   |
| mortality  | DSVA             | DescEmb RNN  | 0.267    | 0.517    | 0.075   |
| mortality  | VC               | DescEmb RNN  | 0.268    | 0.506    | 0.075   |

Validation

| Task       | Value Embedding  | Model        |  Loss    | AUROC    | AUPRC   |
| -----------|------------------|------------- | -------- | -------- | ------- |
| mortality  | DSVA             | CodeEmb      | 0.272    | 0.504    | 0.083   |
| mortality  | DSVA             | DescEmb RNN  | 0.274    | 0.491    | 0.071   |
| mortality  | VC               | DescEmb RNN  | 0.271    | 0.509    | 0.078   |

Test

| Task       | Value Embedding  | Model        |  Loss    | AUROC    | AUPRC   |
| -----------|------------------|------------- | -------- | -------- | ------- |
| mortality  | DSVA             | CodeEmb      | 0.464    | 0.503    | 0.083   |
| mortality  | DSVA             | DescEmb RNN  | 0.465    | 0.490    | 0.073   |
| mortality  | VC               | DescEmb RNN  | 0.460    | 0.509    | 0.081   |



## License

This repository is MIT-licensed.
