import { httpService } from '@/services/http';
import type {
  NasSearchJobData,
  NasSearchLogs,
  StartNasSearchRequest,
} from '@/types/nas';

type NasSearchJobDTO = {
  id: number;
  job_id: string;
  user_id: number;
  dataset_id: number;
  modality: string;
  description?: string;
  config: Record<string, unknown>;
  status: string;
  progress: number;
  genotype?: Record<string, unknown> | null;
  result?: Record<string, unknown> | null;
  backbone_id?: number | null;
  checkpoint_dir?: string | null;
  log_path?: string | null;
  createTime?: string | null;
  updateTime?: string | null;
};

type NasSearchLogsDTO = {
  log: string;
};

export const nasApi = {
  async getNasSearchJobList(): Promise<NasSearchJobData[]> {
    const data = await httpService.get<NasSearchJobDTO[]>('/api/nas/search');

    return data.map((item) => ({
      id: item.id,
      jobId: item.job_id,
      userId: item.user_id,
      datasetId: item.dataset_id,
      modality: item.modality,
      description: item.description ?? '',
      config: item.config,
      status: item.status,
      progress: item.progress,
      genotype: item.genotype,
      result: item.result,
      backboneId: item.backbone_id,
      checkpointDir: item.checkpoint_dir,
      logPath: item.log_path,
      createTime: item.createTime,
      updateTime: item.updateTime,
    }));
  },

  async startNasSearch(request: StartNasSearchRequest): Promise<void> {
    await httpService.post('/api/nas/search/start', {
      user_id: request.userId,
      dataset_id: request.datasetId,
      modality: request.modality,
      description: request.description,
      config: request.config,
    });
  },

  async getNasSearchLogs(jobId: string): Promise<NasSearchLogs> {
    const data = await httpService.get<NasSearchLogsDTO>(
      `/api/nas/logs/${jobId}`,
    );

    return {
      log: data.log,
    };
  },

  async deleteNasSearchJob(id: number): Promise<void> {
    await httpService.delete(`/api/nas/search/${id}`);
  },
};