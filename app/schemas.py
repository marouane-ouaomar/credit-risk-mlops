from typing import List, Literal
from pydantic import BaseModel, Field, ConfigDict


class ApplicantFeatures(BaseModel):
    AMT_INCOME_TOTAL: float = Field(..., gt=0, description="Applicant's total income")
    AMT_CREDIT: float = Field(..., gt=0, description="Requested credit/loan amount")
    AMT_ANNUITY: float = Field(..., gt=0, description="Loan annuity payment")
    DAYS_BIRTH: int = Field(..., lt=0, description="Days before application (negative); age proxy")
    DAYS_EMPLOYED: int = Field(..., description="Days employed before application (negative), or 365243 if not employed")
    DAYS_ID_PUBLISH: int = Field(..., description="Days since ID document was published")
    DAYS_LAST_PHONE_CHANGE: int = Field(..., description="Days since applicant last changed phone")
    CNT_CHILDREN: int = Field(..., ge=0)
    EXT_SOURCE_1: float = Field(..., ge=0, le=1, description="Normalized external credit bureau score")
    EXT_SOURCE_2: float = Field(..., ge=0, le=1)
    EXT_SOURCE_3: float = Field(..., ge=0, le=1)
    CODE_GENDER: Literal["M", "F", "Unknown"]
    FLAG_OWN_CAR: Literal["Y", "N"]
    FLAG_OWN_REALTY: Literal["Y", "N"]
    NAME_EDUCATION_TYPE: str

    class Config:
        json_schema_extra = {
            "example": {
                "AMT_INCOME_TOTAL": 180000,
                "AMT_CREDIT": 500000,
                "AMT_ANNUITY": 25000,
                "DAYS_BIRTH": -14000,
                "DAYS_EMPLOYED": -2500,
                "DAYS_ID_PUBLISH": -3000,
                "DAYS_LAST_PHONE_CHANGE": -500,
                "CNT_CHILDREN": 1,
                "EXT_SOURCE_1": 0.55,
                "EXT_SOURCE_2": 0.60,
                "EXT_SOURCE_3": 0.50,
                "CODE_GENDER": "F",
                "FLAG_OWN_CAR": "N",
                "FLAG_OWN_REALTY": "Y",
                "NAME_EDUCATION_TYPE": "Higher education",
            }
        }


class ReasonCode(BaseModel):
    feature: str
    impact: float


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    probability_of_default: float
    risk_score: int
    decision: Literal["APPROVE", "REVIEW", "REJECT"]
    top_reasons: List[ReasonCode]
    model_version: str


class HealthResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_version: str
    auc: float
    ks_statistic: float
    trained_on: str
    n_train: int
