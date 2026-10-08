import { httpService } from '@/services/http';
import type {
  EffectivenessAnalysisData,
  EffectivenessAnalysisListData,
  EffectivenessMetricKey,
} from '@/types/analysis';

type ApiResp<T> = {
  code: number;
  message: string;
  data: T;
};

type EffectivenessRowDTO = {
  model_id: string;
  group_id: string;
  role: 'baseline' | 'knowledge';
  model_name: string;
  backbone: string;
  data_ratio: number | null;
  adaptive?: boolean;
  train_job_id: string;
  val_job_id: string;
  oa: number | null;
  precision: number | null;
  recall: number | null;
  f1: number | null;
  kappa: number | null;
  loss: number | null;
  uplift_rate: number | null;
  per_sample_gain: number | null;
};

type EffectivenessResultSnapshotDTO = {
  group_id: string;
  metric_key: EffectivenessMetricKey;
  backbone: string;
  rows: EffectivenessRowDTO[];
};

type EffectivenessAnalysisDTO = {
  analysis_id: string;
  analysis_name: string;
  user_id: number;
  metric_key: EffectivenessMetricKey;
  backbone: string;
  baseline_val_job_id: string;
  knowledge_val_job_id: string;
  baseline_train_job_id: string;
  knowledge_train_job_id: string;
  status: string;
  result: EffectivenessResultSnapshotDTO | null;
  remark: string;
  createTime: string | null;
  updateTime: string | null;
};

type EffectivenessAnalysisListDTO = {
  items: EffectivenessAnalysisDTO[];
  total: number;
  page: number;
  page_size: number;
};

export interface CreateEffectivenessAnalysisRequest {
  userId: number;
  analysisName: string;
  baselineValJobId: string;
  knowledgeValJobId: string;
  metricKey: EffectivenessMetricKey;
  remark?: string;
}

function mapRow(row: EffectivenessRowDTO) {
  return {
    modelId: row.model_id,
    groupId: row.group_id,
    role: row.role,
    modelName: row.model_name,
    backbone: row.backbone,
    dataRatio: row.data_ratio,
    adaptive: row.adaptive,
    trainJobId: row.train_job_id,
    valJobId: row.val_job_id,
    oa: row.oa,
    precision: row.precision,
    recall: row.recall,
    f1: row.f1,
    kappa: row.kappa,
    loss: row.loss,
    upliftRate: row.uplift_rate,
    perSampleGain: row.per_sample_gain,
  };
}

function mapAnalysis(item: EffectivenessAnalysisDTO): EffectivenessAnalysisData {
  return {
    analysisId: item.analysis_id,
    analysisName: item.analysis_name,
    userId: item.user_id,
    metricKey: item.metric_key,
    backbone: item.backbone,
    baselineValJobId: item.baseline_val_job_id,
    knowledgeValJobId: item.knowledge_val_job_id,
    baselineTrainJobId: item.baseline_train_job_id,
    knowledgeTrainJobId: item.knowledge_train_job_id,
    status: item.status,
    result: item.result
      ? {
          groupId: item.result.group_id,
          metricKey: item.result.metric_key,
          backbone: item.result.backbone,
          rows: item.result.rows.map(mapRow),
        }
      : null,
    remark: item.remark ?? '',
    createTime: item.createTime,
    updateTime: item.updateTime,
  };
}

export const analysisApi = {
  async createAnalysis(request: CreateEffectivenessAnalysisRequest): Promise<EffectivenessAnalysisData> {
    const resp = await httpService.post<ApiResp<EffectivenessAnalysisDTO>>(
      '/api/analysis/effectiveness/start',
      {
        user_id: request.userId,
        analysis_name: request.analysisName,
        baseline_val_job_id: request.baselineValJobId,
        knowledge_val_job_id: request.knowledgeValJobId,
        metric_key: request.metricKey,
        remark: request.remark ?? '',
      }
    );
    return mapAnalysis(resp.data);
  },

  async getAnalysisList(userId: number, page = 1, pageSize = 10): Promise<EffectivenessAnalysisListData> {
    const resp = await httpService.get<ApiResp<EffectivenessAnalysisListDTO>>(
      `/api/analysis/effectiveness?user_id=${userId}&page=${page}&page_size=${pageSize}`
    );

    return {
      items: resp.data.items.map(mapAnalysis),
      total: resp.data.total,
      page: resp.data.page,
      pageSize: resp.data.page_size,
    };
  },

  async getAnalysisDetail(analysisId: string): Promise<EffectivenessAnalysisData> {
    const resp = await httpService.get<ApiResp<EffectivenessAnalysisDTO>>(
      `/api/analysis/effectiveness/${analysisId}`
    );
    return mapAnalysis(resp.data);
  },

  async deleteAnalysis(analysisId: string): Promise<void> {
    await httpService.delete(`/api/analysis/effectiveness/${analysisId}`);
  },
};