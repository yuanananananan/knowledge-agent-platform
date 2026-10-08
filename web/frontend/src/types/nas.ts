export type NasSearchJobData = {
  id: number;
  jobId: string;
  userId: number;
  datasetId: number;
  modality: string;
  description: string;
  config: Record<string, unknown>;
  status: string;
  progress: number;
  genotype?: Record<string, unknown> | null;
  result?: Record<string, unknown> | null;
  backboneId?: number | null;
  checkpointDir?: string | null;
  logPath?: string | null;
  createTime?: string | null;
  updateTime?: string | null;
};

export type StartNasSearchRequest = {
  userId: number;
  datasetId: number;
  modality: string;
  description?: string;
  config: {
    epochs: number;
    batch_size: number;
    init_channels: number;
    layers: number;
    learning_rate: number;
    learning_rate_min: number;
    momentum: number;
    weight_decay: number;
    arch_learning_rate: number;
    arch_weight_decay: number;
    train_portion: number;
    grad_clip: number;
    unrolled: boolean;
  };
};

export type NasSearchLogs = {
  log: string;
};