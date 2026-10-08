import { httpService } from '@/services/http';
import type { TrainTaskData, TrainingLogs, TrainingResult } from '@/types/task';
import type { ExtraConfigDTO } from './task';

export interface StartTrainingRequest {
  userId: number;
  datasetId: number;
  nodes?: Record<string, unknown>;
  backboneId: number;
  classifierId: number;
  featureIds: number[];
  isGNN: boolean;
  isNas: boolean;
  extra: ExtraConfigDTO[];
}

type TrainTaskDTO = {
  id: number;
  job_id: string;
  val_job_id?: string | null;
  status: string;
  description?: string;
  backbone_id?: number;
  classifier_id?: number;
  dataset_id?: number;
  feature_ids?: number;
  model_path?: string;
  log_path?: string;
  nodes?: Record<string, unknown>;
  config: Record<string, unknown>;
  progress: number;
  result?: Record<string, unknown>;
  createTime: string;
  updateTime: string;
  user_id?: number;
};

type TrainingLogsDTO = {
  log: string;
};

type TrainMetricDTO = {
  Class_Accuracy: number[];
  F1: number;
  Kappa: number;
  OA: number;
  Precision: number;
  Recall: number;
  Test_Loss: number;
  Train_Loss: number;
  epoch: number;
};

type TrainingResultDTO = {
  result: {
    Best: {
      OA: number;
      ep: number;
      kappa: number;
    };
    Metrics: TrainMetricDTO[];
    Test_Loss: number[];
    Train_Loss: number[];
  };
};

export const trainApi = {
  async getTrainTaskList(): Promise<TrainTaskData[]> {
    const data = await httpService.get<TrainTaskDTO[]>('/api/train/');
    return data.map((item) => ({
      id: item.id,
      jobId: item.job_id,
      status: item.status,
      description: item.description ?? '',
      config: item.config,
      createTime: item.createTime,
      updateTime: item.updateTime,
    }));
  },

  async getTrainingLogs(jobId: string): Promise<TrainingLogs> {
    const data = await httpService.get<TrainingLogsDTO>(`/api/train/logs_file/${jobId}`);
    return {
      log: data.log,
    };
  },

  async getTrainingResult(jobId: string): Promise<TrainingResult> {
    const data = await httpService.get<TrainingResultDTO>(`/api/train/result/${jobId}`);
    return {
      best: {
        oa: data.result.Best.OA,
        epoch: data.result.Best.ep,
        kappa: data.result.Best.kappa,
      },
      metrics: data.result.Metrics.map((metric) => ({
        classAccuracy: metric.Class_Accuracy,
        f1: metric.F1,
        kappa: metric.Kappa,
        oa: metric.OA,
        precision: metric.Precision,
        recall: metric.Recall,
        loss: metric.Train_Loss,
        epoch: metric.epoch,
      })),
      testLoss: data.result.Test_Loss,
      trainLoss: data.result.Train_Loss,
    };
  },

  async startTraining(request: StartTrainingRequest): Promise<void> {
    await httpService.post('/api/train/start', {
      user_id: request.userId,
      dataset_id: request.datasetId,
      nodes: request.nodes || {},
      backbone_id: request.backboneId,
      classifier_id: request.classifierId,
      feature_ids: request.featureIds,
      is_GNN: request.isGNN,
      is_nas: request.isNas,
      extra: request.extra,
    });
  },

  async deleteTrainingTask(jobId: string): Promise<void> {
    await httpService.delete(`/api/train/${jobId}`);
  },
};
