import {
  AlertTriangle,
  Check,
  Clock3,
  LoaderCircle,
  RefreshCw,
  ShieldCheck,
  X,
} from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import type { AgentActionDetail, AgentActionProposal } from '@/types/agent';

const actionLabels: Record<string, string> = {
  create_train_job: '创建训练任务',
  stop_train_job: '停止训练任务',
  retry_train_job: '重试训练任务',
  clone_train_job: '克隆训练任务',
  start_validation: '启动验证任务',
};

const executionStatusLabels: Record<string, string> = {
  pending: '等待执行',
  running: '执行中',
  retry: '等待重试',
  executing: '执行中',
  completed: '已完成',
  succeeded: '已完成',
  cancelled: '已取消',
  rejected: '已拒绝',
  expired: '已过期',
};

function parseUtcTimestamp(value: string) {
  const hasTimezone = /(?:z|[+-]\d{2}:?\d{2})$/i.test(value);
  return new Date(hasTimezone ? value : `${value}Z`);
}

function formatTimestamp(value: string) {
  const date = parseUtcTimestamp(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function isExpired(proposal: AgentActionProposal) {
  return parseUtcTimestamp(proposal.expires_at).getTime() <= Date.now();
}

type ActionApprovalPanelProps = {
  proposals: AgentActionProposal[];
  executions: AgentActionDetail[];
  loading: boolean;
  processingId: string | null;
  onApprove: (proposal: AgentActionProposal) => void;
  onReject: (proposal: AgentActionProposal) => void;
  onDismissFailure: (proposalId: string) => void;
  onRefresh: () => void;
};

export function ActionApprovalPanel({
  proposals,
  executions,
  loading,
  processingId,
  onApprove,
  onReject,
  onDismissFailure,
  onRefresh,
}: ActionApprovalPanelProps) {
  return (
    <div className="space-y-3">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="flex items-center gap-2 text-sm font-medium">
            <ShieldCheck className="size-4 text-primary" /> 待审批操作
          </p>
          <p className="mt-1 text-xs text-muted-foreground">
            批准后会立即执行，请先核对参数和影响。
          </p>
        </div>
        <Button
          aria-label="刷新待审批操作"
          disabled={loading}
          onClick={onRefresh}
          size="icon"
          title="刷新"
          variant="ghost"
        >
          <RefreshCw className={`size-4 ${loading ? 'animate-spin' : ''}`} />
        </Button>
      </div>

      {loading && proposals.length === 0 && (
        <div className="flex items-center justify-center gap-2 rounded-lg border border-dashed py-12 text-sm text-muted-foreground">
          <LoaderCircle className="size-4 animate-spin" /> 正在加载草案…
        </div>
      )}

      {!loading && proposals.length === 0 && (
        <div className="rounded-lg border border-dashed px-4 py-12 text-center">
          <ShieldCheck className="mx-auto mb-3 size-7 text-muted-foreground" />
          <p className="text-sm font-medium">暂无待审批操作</p>
          <p className="mt-1 text-xs text-muted-foreground">Agent 生成操作草案后会显示在这里。</p>
        </div>
      )}

      {proposals.map((proposal) => {
        const expired = isExpired(proposal);
        const processing = processingId === proposal.proposal_id;
        return (
          <Card className="gap-3 py-4 shadow-none" key={proposal.proposal_id}>
            <CardHeader className="gap-2 px-4">
              <div className="flex items-start justify-between gap-2">
                <CardTitle className="text-sm leading-5">
                  {actionLabels[proposal.action_type] ?? proposal.action_type}
                </CardTitle>
                <Badge variant={expired ? 'destructive' : 'secondary'}>
                  {expired ? '已过期' : '等待审批'}
                </Badge>
              </div>
              <p className="break-all font-mono text-[11px] text-muted-foreground">
                {proposal.proposal_id}
              </p>
            </CardHeader>
            <CardContent className="space-y-3 px-4 text-xs">
              {proposal.reason && (
                <div>
                  <p className="mb-1 font-medium text-foreground">操作原因</p>
                  <p className="whitespace-pre-wrap text-muted-foreground">{proposal.reason}</p>
                </div>
              )}

              <div className="rounded-md border border-amber-500/30 bg-amber-500/10 p-3">
                <p className="mb-1 flex items-center gap-1.5 font-medium text-foreground">
                  <AlertTriangle className="size-3.5 text-amber-600" /> 影响
                </p>
                <p className="leading-5 text-muted-foreground">{proposal.impact}</p>
              </div>

              <div>
                <p className="mb-1 font-medium text-foreground">操作参数</p>
                <pre className="max-h-52 overflow-auto whitespace-pre-wrap break-all rounded-md bg-muted p-3 font-mono text-[11px] leading-5 text-muted-foreground">
                  {JSON.stringify(proposal.payload, null, 2)}
                </pre>
              </div>

              <p
                className={`flex items-center gap-1.5 ${expired ? 'text-destructive' : 'text-muted-foreground'}`}
              >
                <Clock3 className="size-3.5" />
                {expired ? '已于' : '有效期至'} {formatTimestamp(proposal.expires_at)}
              </p>

              <div className="grid grid-cols-2 gap-2 pt-1">
                <Button
                  disabled={Boolean(processingId)}
                  onClick={() => onReject(proposal)}
                  size="sm"
                  variant="outline"
                >
                  {processing ? (
                    <LoaderCircle className="size-4 animate-spin" />
                  ) : (
                    <X className="size-4" />
                  )}
                  拒绝
                </Button>
                <Button
                  disabled={Boolean(processingId) || expired}
                  onClick={() => onApprove(proposal)}
                  size="sm"
                >
                  {processing ? (
                    <LoaderCircle className="size-4 animate-spin" />
                  ) : (
                    <Check className="size-4" />
                  )}
                  批准并执行
                </Button>
              </div>
            </CardContent>
          </Card>
        );
      })}

      {executions.length > 0 && (
        <div className="space-y-2 pt-2">
          <div>
            <p className="text-sm font-medium">最近执行</p>
            <p className="mt-1 text-xs text-muted-foreground">
              异步任务会自动刷新，失败原因保留在这里。
            </p>
          </div>
          {executions.map(({ proposal, verification }) => {
            const status = String(verification.status ?? proposal.status);
            const result = verification.result;
            const resultObject =
              result && typeof result === 'object' ? (result as Record<string, unknown>) : null;
            const failureReason =
              status === 'failed'
                ? String(
                    resultObject?.error ?? proposal.error_message ?? '任务执行失败，未返回具体原因'
                  )
                : null;
            const targetId = verification.job_id ?? verification.val_job_id;
            const active = ['pending', 'running', 'retry', 'executing'].includes(status);
            const completed = status === 'completed' || status === 'succeeded';

            return (
              <Card className="gap-2 py-3 shadow-none" key={`execution-${proposal.proposal_id}`}>
                <CardHeader className="px-3">
                  <div className="flex items-start justify-between gap-2">
                    <CardTitle className="text-sm">
                      {actionLabels[proposal.action_type] ?? proposal.action_type}
                    </CardTitle>
                    <Badge
                      variant={failureReason ? 'destructive' : completed ? 'default' : 'secondary'}
                    >
                      {failureReason
                        ? '执行失败'
                        : (executionStatusLabels[status] ?? (active ? '执行中' : status))}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent className="space-y-2 px-3 text-xs">
                  {targetId != null && (
                    <p className="break-all font-mono text-[11px] text-muted-foreground">
                      {String(targetId)}
                    </p>
                  )}
                  {failureReason && (
                    <div className="rounded-md border border-destructive/30 bg-destructive/10 p-3 text-destructive">
                      <div className="mb-1 flex items-center justify-between gap-2">
                        <p className="font-medium">失败原因</p>
                        <Button
                          className="h-6 px-2 text-[11px] text-destructive hover:text-destructive"
                          onClick={() => onDismissFailure(proposal.proposal_id)}
                          size="sm"
                          variant="ghost"
                        >
                          清除此提示
                        </Button>
                      </div>
                      <p className="whitespace-pre-wrap break-words leading-5">{failureReason}</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}
