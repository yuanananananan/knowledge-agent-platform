import { useEffect, useMemo, useState } from 'react';
import type { ValTaskData } from '@/types/task';
import type { EffectivenessAnalysisData, EffectivenessMetricKey } from '@/types/analysis';
import { useCreateEffectivenessAnalysis } from '@/hooks/analysis';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';

interface Props {
  open: boolean;
  onOpenChange: (_open: boolean) => void;
  selectedTasks: ValTaskData[];
  userId: number;
  onSuccess?: (_record: EffectivenessAnalysisData) => void;
}

const metricOptions: EffectivenessMetricKey[] = ['oa', 'precision', 'recall', 'f1', 'kappa', 'loss'];

type TaskSummary = {
  valJobId: string;
  trainJobId: string;
  datasetName: string;
  backbone: string;
  metricOA: number | null;
  metricF1: number | null;
  metricRecall: number | null;
  metricPrecision: number | null;
  metricKappa: number | null;
  metricLoss: number | null;
  status: string;
};

function shortId(id?: string | null, left = 8, right = 4) {
  if (!id) return '-';
  if (id.length <= left + right + 3) return id;
  return `${id.slice(0, left)}...${id.slice(-right)}`;
}

function safeNumber(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string' && value.trim() !== '' && !Number.isNaN(Number(value))) {
    return Number(value);
  }
  return null;
}

function readMetrics(task: any) {
  const metrics = task?.result?.Metrics || task?.result?.metrics || {};
  return {
    oa: safeNumber(metrics.OA ?? metrics.oa),
    f1: safeNumber(metrics.F1 ?? metrics.f1),
    recall: safeNumber(metrics.Recall ?? metrics.recall),
    precision: safeNumber(metrics.Precision ?? metrics.precision),
    kappa: safeNumber(metrics.Kappa ?? metrics.kappa),
    loss: safeNumber(task?.result?.Loss ?? task?.result?.loss ?? metrics.Loss ?? metrics.loss),
  };
}

function normalizeTask(task: any): TaskSummary {
  const metrics = readMetrics(task);

  return {
    valJobId: task?.valJobId ?? task?.val_job_id ?? '',
    trainJobId: task?.trainJobId ?? task?.train_job_id ?? '',
    datasetName:
      task?.datasetName ??
      task?.dataset_name ??
      task?.config?.data?.name ??
      task?.config?.data?.dataset_name ??
      '未知数据集',
    backbone:
      task?.backbone ??
      task?.config?.model?.backbone ??
      task?.model?.backbone ??
      '未知骨干',
    metricOA: metrics.oa,
    metricF1: metrics.f1,
    metricRecall: metrics.recall,
    metricPrecision: metrics.precision,
    metricKappa: metrics.kappa,
    metricLoss: metrics.loss,
    status: task?.status ?? '-',
  };
}

function buildDefaultAnalysisName(
  baseline: TaskSummary | undefined,
  knowledge: TaskSummary | undefined,
  metricKey: EffectivenessMetricKey
) {
  if (!baseline || !knowledge) return '';

  const sameDataset = baseline.datasetName === knowledge.datasetName;
  const sameBackbone = baseline.backbone === knowledge.backbone;

  if (sameDataset && sameBackbone) {
    return `${baseline.datasetName}-${baseline.backbone}-${metricKey}分析`;
  }

  return `${baseline.datasetName}-${knowledge.datasetName}-${metricKey}分析`;
}

function getMetricDisplay(task: TaskSummary, metricKey: EffectivenessMetricKey) {
  const mapping: Record<EffectivenessMetricKey, number | null> = {
    oa: task.metricOA,
    precision: task.metricPrecision,
    recall: task.metricRecall,
    f1: task.metricF1,
    kappa: task.metricKappa,
    loss: task.metricLoss,
  };

  const value = mapping[metricKey];
  return value == null ? '-' : value;
}

function SummaryCard({
  title,
  task,
  metricKey,
}: {
  title: string;
  task?: TaskSummary;
  metricKey: EffectivenessMetricKey;
}) {
  if (!task) return null;

  return (
    <div className="rounded-md border bg-muted/20 p-4">
      <div className="mb-3 text-sm font-semibold">{title}</div>

      <div className="grid grid-cols-[90px_minmax(0,1fr)] gap-x-3 gap-y-2 text-sm">
        <div className="text-muted-foreground">数据集</div>
        <div className="truncate" title={task.datasetName}>
          {task.datasetName}
        </div>

        <div className="text-muted-foreground">Backbone</div>
        <div className="truncate" title={task.backbone}>
          {task.backbone}
        </div>

        <div className="text-muted-foreground">测试任务ID</div>
        <div className="truncate" title={task.valJobId}>
          {shortId(task.valJobId)}
        </div>

        <div className="text-muted-foreground">训练任务ID</div>
        <div className="truncate" title={task.trainJobId}>
          {shortId(task.trainJobId)}
        </div>

        <div className="text-muted-foreground">{metricKey.toUpperCase()}</div>
        <div>{getMetricDisplay(task, metricKey)}</div>

        <div className="text-muted-foreground">状态</div>
        <div>{task.status}</div>
      </div>
    </div>
  );
}

export default function CreateEffectivenessAnalysisDialog({
  open,
  onOpenChange,
  selectedTasks,
  userId,
  onSuccess,
}: Props) {
  const { mutate: createAnalysis, isPending } = useCreateEffectivenessAnalysis();

  const normalizedTasks = useMemo(
    () => selectedTasks.map((task) => normalizeTask(task)),
    [selectedTasks]
  );

  const [analysisName, setAnalysisName] = useState('');
  const [metricKey, setMetricKey] = useState<EffectivenessMetricKey>('oa');
  const [remark, setRemark] = useState('');
  const [baselineValJobId, setBaselineValJobId] = useState('');
  const [knowledgeValJobId, setKnowledgeValJobId] = useState('');

  const baselineTask = useMemo(
    () => normalizedTasks.find((task) => task.valJobId === baselineValJobId),
    [normalizedTasks, baselineValJobId]
  );

  const knowledgeTask = useMemo(
    () => normalizedTasks.find((task) => task.valJobId === knowledgeValJobId),
    [normalizedTasks, knowledgeValJobId]
  );

  useEffect(() => {
    if (normalizedTasks.length >= 2) {
      const baseline = normalizedTasks[0];
      const knowledge = normalizedTasks[1];

      setBaselineValJobId(baseline.valJobId);
      setKnowledgeValJobId(knowledge.valJobId);
      setAnalysisName(buildDefaultAnalysisName(baseline, knowledge, 'oa'));
    }
  }, [normalizedTasks]);

  useEffect(() => {
    if (baselineTask && knowledgeTask) {
      setAnalysisName((prev) => {
        if (!prev || prev.includes('-分析') || prev.includes('有效性分析')) {
          return buildDefaultAnalysisName(baselineTask, knowledgeTask, metricKey);
        }
        return prev;
      });
    }
  }, [baselineTask, knowledgeTask, metricKey]);

  const datasetMismatch =
    baselineTask && knowledgeTask && baselineTask.datasetName !== knowledgeTask.datasetName;

  const backboneMismatch =
    baselineTask && knowledgeTask && baselineTask.backbone !== knowledgeTask.backbone;

  const taskOptions = useMemo(
    () =>
      normalizedTasks.map((task) => ({
        value: task.valJobId,
        label: `${task.datasetName} | ${task.backbone} | ${metricKey.toUpperCase()}:${getMetricDisplay(task, metricKey)} | ${shortId(task.valJobId)}`,
      })),
    [normalizedTasks, metricKey]
  );

  const handleSubmit = () => {
    if (!baselineValJobId || !knowledgeValJobId) return;
    if (baselineValJobId === knowledgeValJobId) return;

    createAnalysis(
      {
        userId,
        analysisName: analysisName.trim(),
        baselineValJobId,
        knowledgeValJobId,
        metricKey,
        remark,
      },
      {
        onSuccess: (record) => {
          onSuccess?.(record);
          onOpenChange(false);
        },
      }
    );
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-none w-[82vw] max-w-[900px] max-h-[88vh] overflow-y-auto p-6">
        <DialogHeader className="pb-1">
          <DialogTitle className="text-xl font-bold">创建有效性分析</DialogTitle>
        </DialogHeader>

        <div className="space-y-5">
          {(datasetMismatch || backboneMismatch) && (
            <div className="rounded-md border border-yellow-500/40 bg-yellow-500/10 px-4 py-2.5 text-sm text-yellow-200">
              {datasetMismatch && <div>当前两个测试任务的数据集不一致，分析结果可能没有可比性。</div>}
              {backboneMismatch && <div>当前两个测试任务的 Backbone 不一致，分析结果可能没有可比性。</div>}
            </div>
          )}

          <div className="grid grid-cols-[minmax(0,1fr)_180px] gap-4 items-start">
            <div className="space-y-2">
              <Label className="text-sm">分析名称</Label>
              <Input
                value={analysisName}
                onChange={(e) => setAnalysisName(e.target.value)}
                placeholder="例如：FUSAR-resnet18-oa分析"
                className="h-10 text-sm"
              />
            </div>

            <div className="space-y-2">
              <Label className="text-sm">分析指标</Label>
              <Select value={metricKey} onValueChange={(v) => setMetricKey(v as EffectivenessMetricKey)}>
                <SelectTrigger className="w-full h-10 text-sm">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {metricOptions.map((item) => (
                    <SelectItem key={item} value={item}>
                      {item.toUpperCase()}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label className="text-sm">基准模型（baseline）</Label>
              <div className="text-xs text-muted-foreground">作为对照组的测试任务</div>
              <Select value={baselineValJobId} onValueChange={setBaselineValJobId}>
                <SelectTrigger className="w-full h-10 text-sm">
                  <SelectValue placeholder="请选择 baseline" />
                </SelectTrigger>
                <SelectContent>
                  {taskOptions.map((task) => (
                    <SelectItem key={task.value} value={task.value}>
                      {task.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label className="text-sm">知识模型（knowledge）</Label>
              <div className="text-xs text-muted-foreground">加入知识嵌入后的测试任务</div>
              <Select value={knowledgeValJobId} onValueChange={setKnowledgeValJobId}>
                <SelectTrigger className="w-full h-10 text-sm">
                  <SelectValue placeholder="请选择 knowledge" />
                </SelectTrigger>
                <SelectContent>
                  {taskOptions.map((task) => (
                    <SelectItem key={task.value} value={task.value}>
                      {task.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <SummaryCard title="基准模型摘要" task={baselineTask} metricKey={metricKey} />
            <SummaryCard title="知识模型摘要" task={knowledgeTask} metricKey={metricKey} />
          </div>

          <div className="space-y-2">
            <Label className="text-sm">备注</Label>
            <Textarea
              value={remark}
              onChange={(e) => setRemark(e.target.value)}
              placeholder="可选：补充说明这次分析的目的"
              className="min-h-[120px] text-sm"
            />
          </div>

          <div className="flex justify-end gap-3 pt-1">
            <Button
              variant="outline"
              onClick={() => onOpenChange(false)}
              disabled={isPending}
              className="h-10 px-5 text-sm"
            >
              取消
            </Button>
            <Button
              onClick={handleSubmit}
              disabled={
                isPending ||
                !analysisName.trim() ||
                !baselineValJobId ||
                !knowledgeValJobId ||
                baselineValJobId === knowledgeValJobId
              }
              className="h-10 px-5 text-sm"
            >
              {isPending ? '创建中...' : '确认创建'}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}