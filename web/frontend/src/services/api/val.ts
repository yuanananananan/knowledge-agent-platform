import { httpService } from '@/services/http';
import type { ValTaskData, ValidationResult, TrainingLogs, ValidateJobResult } from '@/types/task';

type ValTaskDTO = {
  id: number;
  train_job_id: string;
  val_job_id: string;
  status: string;
  config: Record<string, unknown>;
  result?: Record<string, unknown>;
  createTime: string;
  updateTime: string;
  user_id?: number;
};

type ValidationResultDTO = {
  result: {
    Loss: number[];
    Metrics: {
      Class_Accuracy: number[];
      F1: number;
      Kappa: number;
      OA: number;
      Precision: number;
      Recall: number;
    };
  };
};

type ValidationLogsDTO = {
  log: string;
};

interface StartValidationRequest {
  userId: number;
  trainJobId: string;
}

interface ValidateJobsRequest {
  jobIds: string[];
}

type ValidateJobsResponseDTO = {
  results: Array<{
    id: string;
    Metrics: {
      Class_Accuracy: number[];
      F1: number;
      Kappa: number;
      OA: number;
      Precision: number;
      Recall: number;
    };
  }>;
};

export const valApi = {
  async getValTaskList(): Promise<ValTaskData[]> {
    const data = await httpService.get<ValTaskDTO[]>('/api/val/');
    return data.map((item) => ({
      id: item.id,
      trainJobId: item.train_job_id,
      valJobId: item.val_job_id,
      status: item.status,
      config: item.config,
      result: item.result,
      createTime: item.createTime,
      updateTime: item.updateTime,
    }));
  },

  async getValidationResult(jobId: string): Promise<ValidationResult> {
    const data = await httpService.get<ValidationResultDTO>(`/api/val/result/${jobId}`);
    return {
      loss: data.result.Loss,
      metrics: {
        classAccuracy: data.result.Metrics.Class_Accuracy,
        f1: data.result.Metrics.F1,
        kappa: data.result.Metrics.Kappa,
        oa: data.result.Metrics.OA,
        precision: data.result.Metrics.Precision,
        recall: data.result.Metrics.Recall,
      },
    };
  },

  async getValidationLogs(jobId: string): Promise<TrainingLogs> {
    const data = await httpService.get<ValidationLogsDTO>(`/api/val/logs_file/${jobId}`);
    return {
      log: data.log,
    };
  },

  async startValidation(request: StartValidationRequest): Promise<void> {
    await httpService.post('/api/val/start', {
      user_id: request.userId,
      train_job_id: request.trainJobId,
    });
  },

  async validateJobs(request: ValidateJobsRequest): Promise<ValidateJobResult[]> {
    const data = await httpService.post<ValidateJobsResponseDTO>('/api/val/validate', {
      job_ids: request.jobIds,
    });

    return data.results.map((item) => ({
      id: item.id,
      metrics: {
        classAccuracy: item.Metrics.Class_Accuracy,
        f1: item.Metrics.F1,
        kappa: item.Metrics.Kappa,
        oa: item.Metrics.OA,
        precision: item.Metrics.Precision,
        recall: item.Metrics.Recall,
      },
    }));
  },

  async deleteValJob(jobId: string): Promise<void> {
    await httpService.delete(`/api/val/${jobId}`);
  },
};