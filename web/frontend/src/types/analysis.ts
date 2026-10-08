export type EffectivenessMetricKey =
  | 'oa'
  | 'precision'
  | 'recall'
  | 'f1'
  | 'kappa'
  | 'loss';

export type EffectivenessRow = {
  modelId: string;
  groupId: string;
  role: 'baseline' | 'knowledge';
  modelName: string;
  backbone: string;
  dataRatio: number | null;
  adaptive?: boolean;
  trainJobId: string;
  valJobId: string;
  oa: number | null;
  precision: number | null;
  recall: number | null;
  f1: number | null;
  kappa: number | null;
  loss: number | null;
  upliftRate: number | null;
  perSampleGain: number | null;
};

export type EffectivenessResultSnapshot = {
  groupId: string;
  metricKey: EffectivenessMetricKey;
  backbone: string;
  rows: EffectivenessRow[];
};

export type EffectivenessAnalysisData = {
  analysisId: string;
  analysisName: string;
  userId: number;
  metricKey: EffectivenessMetricKey;
  backbone: string;
  baselineValJobId: string;
  knowledgeValJobId: string;
  baselineTrainJobId: string;
  knowledgeTrainJobId: string;
  status: string;
  result: EffectivenessResultSnapshot | null;
  remark: string;
  createTime: string | null;
  updateTime: string | null;
};

export type EffectivenessAnalysisListData = {
  items: EffectivenessAnalysisData[];
  total: number;
  page: number;
  pageSize: number;
};