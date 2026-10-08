export type TrainTaskData = {
  id: number;
  jobId: string;
  status: string;
  description: string;
  config: Record<string, unknown>;
  createTime: string;
  updateTime: string;
};

export type ValTaskData = {
  id: number;
  trainJobId: string;
  valJobId: string;
  status: string;
  config: Record<string, unknown>;
  result?: {
    Loss?: number;
    Metrics?: {
      OA?: number;
      F1?: number;
      Recall?: number;
      Precision?: number;
      Kappa?: number;
      Class_Accuracy?: number[];
    };
  };
  createTime: string;
  updateTime: string;
};

export type TrainingLogs = {
  log: string;
};

export type Metric = {
  classAccuracy: number[];
  f1: number;
  kappa: number;
  oa: number;
  precision: number;
  recall: number;
  loss: number;
  epoch: number;
};

export type TrainingResult = {
  best: {
    oa: number;
    epoch: number;
    kappa: number;
  };
  metrics: Metric[];
  testLoss: number[];
  trainLoss: number[];
};

export type ValidationMetrics = {
  classAccuracy: number[];
  f1: number;
  kappa: number;
  oa: number;
  precision: number;
  recall: number;
};

export type ValidationResult = {
  loss: number[];
  metrics: ValidationMetrics;
};

export type ValidateJobMetrics = {
  classAccuracy: number[];
  f1: number;
  kappa: number;
  oa: number;
  precision: number;
  recall: number;
};

export type ValidateJobResult = {
  id: string;
  metrics: ValidateJobMetrics;
};
