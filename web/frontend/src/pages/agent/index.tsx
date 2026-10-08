import { useCallback, useEffect, useRef, useState } from 'react';
import {
  Bot,
  CircleDot,
  FlaskConical,
  History,
  Plus,
  Send,
  ShieldCheck,
  Square,
  Trash2,
  Wrench,
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { toast } from 'sonner';
import { useNavigate, useParams } from 'react-router';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Textarea } from '@/components/ui/textarea';
import { agentApi } from '@/services/api/agent';
import type {
  AgentActionDetail,
  AgentActionProposal,
  AgentConversation,
  AgentMessage,
  AgentTraceEvent,
  ActiveRun,
  EvalRun,
  GoldenCase,
} from '@/types/agent';
import { ActionApprovalPanel } from './action-approval-panel';

const eventLabels: Record<AgentTraceEvent['type'], string> = {
  'run.accepted': '任务已接收',
  'run.created': '任务已创建',
  'run.collecting': '收集证据',
  'run.reasoning': '分析推理',
  'run.started': '开始运行',
  'model.delta': '模型输出',
  'tool.started': '调用工具',
  'tool.completed': '工具完成',
  'workflow.node.completed': '工作流节点完成',
  'run.completed': '运行完成',
  'run.failed': '运行失败',
  'run.cancelled': '运行已取消',
};

function MarkdownMessage({ content }: { content: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        h1: ({ children }) => <h1 className="mb-3 mt-1 text-xl font-bold">{children}</h1>,
        h2: ({ children }) => <h2 className="mb-2 mt-4 text-lg font-semibold">{children}</h2>,
        h3: ({ children }) => <h3 className="mb-2 mt-3 font-semibold">{children}</h3>,
        p: ({ children }) => <p className="my-2 first:mt-0 last:mb-0">{children}</p>,
        ul: ({ children }) => <ul className="my-2 list-disc space-y-1 pl-5">{children}</ul>,
        ol: ({ children }) => <ol className="my-2 list-decimal space-y-1 pl-5">{children}</ol>,
        li: ({ children }) => <li className="pl-0.5">{children}</li>,
        blockquote: ({ children }) => (
          <blockquote className="my-3 border-l-4 border-primary/40 pl-3 text-muted-foreground">
            {children}
          </blockquote>
        ),
        a: ({ children, href }) => (
          <a
            className="text-primary underline underline-offset-4"
            href={href}
            rel="noreferrer"
            target="_blank"
          >
            {children}
          </a>
        ),
        pre: ({ children }) => (
          <pre className="my-3 max-w-full overflow-x-auto rounded-lg bg-muted p-3 text-xs leading-5 [&>code]:bg-transparent [&>code]:p-0">
            {children}
          </pre>
        ),
        code: ({ children, className }) => (
          <code
            className={`rounded bg-muted px-1.5 py-0.5 font-mono text-[0.85em] ${className ?? ''}`}
          >
            {children}
          </code>
        ),
        table: ({ children }) => (
          <div className="my-3 max-w-full overflow-x-auto rounded-lg border">
            <table className="w-full border-collapse text-left text-xs">{children}</table>
          </div>
        ),
        th: ({ children }) => (
          <th className="border-b bg-muted px-3 py-2 font-semibold">{children}</th>
        ),
        td: ({ children }) => (
          <td className="border-b px-3 py-2 align-top last:border-b-0">{children}</td>
        ),
        hr: () => <hr className="my-4 border-border" />,
      }}
    >
      {content}
    </ReactMarkdown>
  );
}

export default function AgentPage() {
  const navigate = useNavigate();
  const { conversationId: routeConversationId } = useParams<{ conversationId: string }>();
  const [conversations, setConversations] = useState<AgentConversation[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<AgentMessage[]>([]);
  const [events, setEvents] = useState<AgentTraceEvent[]>([]);
  const [prompt, setPrompt] = useState('');
  const [activeRuns, setActiveRuns] = useState<Record<string, ActiveRun>>(() => {
    try {
      return JSON.parse(localStorage.getItem('agent-active-runs') || '{}') as Record<string, ActiveRun>;
    } catch {
      return {};
    }
  });
  const [goldenCases, setGoldenCases] = useState<GoldenCase[]>([]);
  const [evalRuns, setEvalRuns] = useState<EvalRun[]>([]);
  const [evaluating, setEvaluating] = useState(false);
  const [pendingActions, setPendingActions] = useState<AgentActionProposal[]>([]);
  const [recentExecutions, setRecentExecutions] = useState<AgentActionDetail[]>([]);
  const [approvalsLoading, setApprovalsLoading] = useState(false);
  const [processingActionId, setProcessingActionId] = useState<string | null>(null);
  const [approvalDialogOpen, setApprovalDialogOpen] = useState(false);
  const [monitorSummary, setMonitorSummary] = useState<{ runs: Record<string, number>; tasks: Record<string, number> } | null>(null);
  const [selectedEvaluation, setSelectedEvaluation] = useState<{ evaluation: EvalRun; results: unknown[] } | null>(null);
  const [dismissedFailures, setDismissedFailures] = useState<Set<string>>(() => {
    try {
      return new Set(JSON.parse(localStorage.getItem('agent-dismissed-failures') || '[]'));
    } catch {
      return new Set();
    }
  });
  const [deleteTarget, setDeleteTarget] = useState<AgentConversation | null>(null);
  const [deletePassword, setDeletePassword] = useState('');
  const [deleting, setDeleting] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const currentConversationRef = useRef<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const activeRun = conversationId ? activeRuns[conversationId] : undefined;
  const running = Boolean(activeRun && ['pending', 'collecting_evidence', 'reasoning'].includes(activeRun.status));
  const evaluationRunning = evalRuns.some((item) => item.status === 'running');

  useEffect(() => {
    currentConversationRef.current = conversationId;
  }, [conversationId]);

  useEffect(() => {
    localStorage.setItem('agent-active-runs', JSON.stringify(activeRuns));
  }, [activeRuns]);

  useEffect(() => {
    if (routeConversationId && routeConversationId !== conversationId) {
      setConversationId(routeConversationId);
    }
  }, [conversationId, routeConversationId]);

  const refreshSidebar = useCallback(async () => {
    const [conversationData, caseData, evaluationData, monitorData] = await Promise.all([
      agentApi.listConversations(),
      agentApi.listGoldenCases(),
      agentApi.listEvaluations(),
      agentApi.getMonitorSummary().catch(() => null),
    ]);
    setConversations(conversationData.items);
    const latestRuns = await Promise.all(
      conversationData.items.map(async (conversation) => ({
        conversationId: conversation.conversation_id,
        run: (await agentApi.listConversationRuns(conversation.conversation_id).catch(() => ({ items: [] }))).items[0],
      }))
    );
    setActiveRuns((current) => {
      const next = { ...current };
      for (const { conversationId: id, run } of latestRuns) {
        if (run) next[id] = {
          runId: run.run_id,
          conversationId: id,
          lastEventId: current[id]?.runId === run.run_id ? current[id].lastEventId : 0,
          status: run.status,
        };
      }
      return next;
    });
    setGoldenCases(caseData.items);
    setEvalRuns(evaluationData.items);
    setMonitorSummary(monitorData);
  }, []);

  const refreshApprovals = useCallback(async (silent = false) => {
    if (!silent) setApprovalsLoading(true);
    try {
      const data = await agentApi.listActionProposals(undefined, 20, conversationId);
      setPendingActions(data.items.filter((item) => item.status === 'pending_approval'));

      const inspectable = data.items
        .filter((item) => item.status === 'succeeded' || item.status === 'failed')
        .slice(0, 8);
      const details = (
        await Promise.all(
          inspectable.map(async (item) => {
            try {
              return await agentApi.getActionProposal(item.proposal_id, true);
            } catch {
              return null;
            }
          })
        )
      ).filter((item): item is AgentActionDetail => item !== null);

      setRecentExecutions(
        details.filter(({ proposal }) => !dismissedFailures.has(proposal.proposal_id))
      );
    } finally {
      if (!silent) setApprovalsLoading(false);
    }
  }, [conversationId, dismissedFailures]);

  const refreshEvaluations = useCallback(async () => {
    const data = await agentApi.listEvaluations();
    setEvalRuns(data.items);
    const selectedId = selectedEvaluation?.evaluation.eval_run_id;
    if (selectedId) {
      const detail = await agentApi.getEvaluationDetail(selectedId);
      setSelectedEvaluation({ evaluation: detail.evaluation, results: detail.results });
    }
  }, [selectedEvaluation?.evaluation.eval_run_id]);

  const dismissFailure = (proposalId: string) => {
    setDismissedFailures((current) => {
      const next = new Set(current);
      next.add(proposalId);
      localStorage.setItem('agent-dismissed-failures', JSON.stringify([...next]));
      return next;
    });
  };

  useEffect(() => {
    Promise.all([refreshSidebar(), refreshApprovals()]).catch(() => undefined);
  }, [refreshApprovals, refreshSidebar]);

  const hasActiveExecution = recentExecutions.some(({ verification }) =>
    ['pending', 'running', 'retry', 'executing'].includes(String(verification.status ?? ''))
  );

  useEffect(() => {
    if (!hasActiveExecution) return;
    const timer = window.setInterval(() => {
      void refreshApprovals(true).catch(() => undefined);
    }, 3_000);
    return () => window.clearInterval(timer);
  }, [hasActiveExecution, refreshApprovals]);

  useEffect(() => {
    if (!evalRuns.some((item) => item.status === 'running')) return;
    const timer = window.setInterval(() => {
      void refreshEvaluations().catch(() => undefined);
    }, 2_000);
    return () => window.clearInterval(timer);
  }, [evalRuns, refreshEvaluations]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'auto', block: 'end' });
  }, [messages]);

  // Reload server-backed approvals whenever the user returns to this tab/window.
  useEffect(() => {
    const refresh = () => {
      if (document.visibilityState === 'visible') {
        void refreshApprovals(true).catch(() => undefined);
        void refreshEvaluations().catch(() => undefined);
      }
    };
    window.addEventListener('focus', refresh);
    document.addEventListener('visibilitychange', refresh);
    return () => {
      window.removeEventListener('focus', refresh);
      document.removeEventListener('visibilitychange', refresh);
    };
  }, [refreshApprovals, refreshEvaluations]);

  const applyRunEvent = useCallback((event: AgentTraceEvent, runId: string, displayedConversation: string) => {
    if (event.run_id !== runId || currentConversationRef.current !== displayedConversation) return;
    setEvents((current) => current.some((item) => item.sequence === event.sequence && item.run_id === event.run_id)
      ? current : [...current, event]);
    setActiveRuns((current) => {
      const previous = current[displayedConversation];
      const status = event.type === 'run.completed' ? 'succeeded'
        : event.type === 'run.failed' ? 'failed'
          : event.type === 'run.cancelled' ? 'cancelled'
            : previous?.status ?? 'reasoning';
      return {
        ...current,
        [displayedConversation]: {
          runId,
          conversationId: displayedConversation,
          lastEventId: Math.max(previous?.lastEventId ?? 0, event.sequence),
          status,
        },
      };
    });
    if (event.type === 'model.delta') {
      const delta = String(event.payload.content ?? '');
      setMessages((current) => {
        const last = current[current.length - 1];
        if (last?.role === 'assistant') {
          return current.map((item, index) => index === current.length - 1
            ? { ...item, content: item.content + delta } : item);
        }
        return [...current, { role: 'assistant', content: delta }];
      });
    } else if (event.type === 'run.completed') {
      const answer = String(event.payload.answer ?? '');
      if (answer) setMessages((current) => {
        const last = current[current.length - 1];
        return last?.role === 'assistant'
          ? current.map((item, index) => index === current.length - 1 ? { ...item, content: answer } : item)
          : [...current, { role: 'assistant', content: answer }];
      });
      void Promise.all([refreshSidebar(), refreshApprovals(true)]).catch(() => undefined);
    } else if (event.type === 'run.failed') {
      const visibleFailure = String(
        event.payload.partial_answer
          ?? `本次 Agent 运行失败：${event.payload.message ?? event.payload.error ?? '未知错误'}`
      );
      setMessages((current) => {
        const last = current[current.length - 1];
        return last?.role === 'assistant'
          ? current.map((item, index) => index === current.length - 1
            ? { ...item, content: visibleFailure } : item)
          : [...current, { role: 'assistant', content: visibleFailure }];
      });
      toast.error(String(event.payload.message ?? event.payload.error ?? 'Agent 运行失败'));
    }
  }, [refreshApprovals, refreshSidebar]);

  const resumeRun = useCallback((run: ActiveRun) => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    void agentApi.resumeRun(
      run.runId,
      run.lastEventId,
      (event) => applyRunEvent(event, run.runId, run.conversationId),
      controller.signal
    ).catch((error) => {
      if (!controller.signal.aborted) toast.error(error instanceof Error ? error.message : '恢复运行失败');
    });
  }, [applyRunEvent]);

  const loadConversation = useCallback(async (id: string) => {
    abortRef.current?.abort();
    currentConversationRef.current = id;
    const data = await agentApi.getConversation(id);
    setConversationId(id);
    setMessages(data.messages);
    const runs = await agentApi.listConversationRuns(id).catch(() => ({ items: [] }));
    const latestRun = runs.items[0];
    if (latestRun) {
      const trace = await agentApi.getRunEvents(latestRun.run_id).catch(() => null);
      setEvents(trace?.events ?? []);
      const lastEventId = Math.max(0, ...(trace?.events ?? []).map((event) => event.sequence));
      const restored: ActiveRun = {
        runId: latestRun.run_id,
        conversationId: id,
        lastEventId,
        status: latestRun.status,
      };
      setActiveRuns((current) => ({ ...current, [id]: restored }));
      const serverEndsWithUser = data.messages[data.messages.length - 1]?.role === 'user';
      const deltas = (trace?.events ?? [])
        .filter((event) => event.type === 'model.delta')
        .map((event) => String(event.payload.content ?? ''))
        .join('');
      if (serverEndsWithUser && latestRun.status === 'failed') {
        const failedEvent = [...(trace?.events ?? [])].reverse().find((event) => event.type === 'run.failed');
        const visibleFailure = String(
          failedEvent?.payload.partial_answer
            ?? (deltas
              ? `${deltas}\n\n草案未创建：${failedEvent?.payload.message ?? failedEvent?.payload.error ?? '参数校验未通过'}`
              : `本次 Agent 运行失败：${failedEvent?.payload.message ?? failedEvent?.payload.error ?? '未知错误'}`)
        );
        setMessages((current) => [...current, { role: 'assistant', content: visibleFailure }]);
      }
      if (['pending', 'collecting_evidence', 'reasoning'].includes(latestRun.status)) {
        if (deltas) setMessages((current) => {
          const last = current[current.length - 1];
          return last?.role === 'assistant' ? current : [...current, { role: 'assistant', content: deltas }];
        });
        resumeRun(restored);
      }
    } else {
      setEvents([]);
    }
  }, [resumeRun]);

  useEffect(() => {
    if (routeConversationId) void loadConversation(routeConversationId).catch(() => undefined);
  }, [loadConversation, routeConversationId]);

  useEffect(() => {
    const restore = () => {
      if (document.visibilityState === 'visible' && currentConversationRef.current) {
        void loadConversation(currentConversationRef.current).catch(() => undefined);
      }
    };
    window.addEventListener('focus', restore);
    document.addEventListener('visibilitychange', restore);
    return () => {
      window.removeEventListener('focus', restore);
      document.removeEventListener('visibilitychange', restore);
    };
  }, [loadConversation]);

  const openConversation = async (id: string) => {
    navigate(`/agent/${encodeURIComponent(id)}`);
  };

  const newConversation = async () => {
    abortRef.current?.abort();
    try {
      const data = await agentApi.createConversation();
      setConversationId(data.conversation.conversation_id);
      navigate(`/agent/${encodeURIComponent(data.conversation.conversation_id)}`);
      setMessages([]);
      setEvents([]);
      setPrompt('');
      await Promise.all([refreshSidebar(), refreshApprovals()]);
    } catch {
      // The HTTP interceptor already shows the server error.
    }
  };

  const requestDelete = (conversation: AgentConversation) => {
    setDeleteTarget(conversation);
    setDeletePassword('');
  };

  const deleteConversation = async () => {
    if (!deleteTarget || !deletePassword || deleting) return;
    setDeleting(true);
    try {
      await agentApi.deleteConversation(deleteTarget.conversation_id, deletePassword);
      if (conversationId === deleteTarget.conversation_id) {
        setConversationId(null);
        navigate('/agent');
        setMessages([]);
        setEvents([]);
        setPrompt('');
      }
      setDeleteTarget(null);
      setDeletePassword('');
      await Promise.all([refreshSidebar(), refreshApprovals()]);
      toast.success('会话已永久删除');
    } finally {
      setDeleting(false);
    }
  };

  const send = async () => {
    const text = prompt.trim();
    if (!text || running) return;
    let targetConversation = conversationId;
    if (!targetConversation) {
      const created = await agentApi.createConversation();
      targetConversation = created.conversation.conversation_id;
      currentConversationRef.current = targetConversation;
      setConversationId(targetConversation);
      navigate(`/agent/${encodeURIComponent(targetConversation)}`);
    }
    setPrompt('');
    setEvents([]);
    setMessages((current) => [
      ...current,
      { role: 'user', content: text },
      { role: 'assistant', content: '' },
    ]);
    const controller = new AbortController();
    abortRef.current = controller;
    let acceptedRunId = '';
    try {
      await agentApi.streamChat(
        text,
        targetConversation,
        (event) => {
          if (event.type === 'run.accepted') {
            acceptedRunId = event.run_id;
            setActiveRuns((current) => ({
              ...current,
              [targetConversation]: {
                runId: event.run_id,
                conversationId: targetConversation,
                lastEventId: event.sequence,
                status: 'pending',
              },
            }));
          }
          if (acceptedRunId) applyRunEvent(event, acceptedRunId, targetConversation);
        },
        controller.signal
      );
      await Promise.all([refreshSidebar(), refreshApprovals()]);
    } catch (error) {
      if (!controller.signal.aborted)
        toast.error(error instanceof Error ? error.message : '发送失败');
    } finally {
      abortRef.current = null;
    }
  };

  const stop = async () => {
    if (!activeRun) return;
    try {
      await agentApi.cancelRun(activeRun.runId);
      abortRef.current?.abort();
      setActiveRuns((current) => ({
        ...current,
        [activeRun.conversationId]: { ...activeRun, status: 'cancelled' },
      }));
      toast.success('已取消后台 Agent 运行');
    } catch {
      // The HTTP interceptor already shows the server error.
    }
  };

  const runEvaluation = async () => {
    setEvaluating(true);
    try {
      const result = await agentApi.runEvaluation();
      setEvalRuns((current) => [
        result.evaluation,
        ...current.filter((item) => item.eval_run_id !== result.evaluation.eval_run_id),
      ]);
      setSelectedEvaluation({ evaluation: result.evaluation, results: [] });
      toast.success(`评测已转入后台：0/${result.evaluation.total}`);
    } finally {
      setEvaluating(false);
    }
  };

  const approveAction = async (proposal: AgentActionProposal) => {
    if (processingActionId) return;
    setProcessingActionId(proposal.proposal_id);
    try {
      const result = await agentApi.approveActionProposal(proposal.proposal_id);
      const targetStatus = result.verification.status;
      toast.success(
        targetStatus ? `操作已执行，目标状态：${String(targetStatus)}` : '操作已批准并执行'
      );
    } catch {
      // The HTTP interceptor already shows the server error.
    } finally {
      await refreshApprovals().catch(() => undefined);
      setProcessingActionId(null);
    }
  };

  const rejectAction = async (proposal: AgentActionProposal) => {
    if (processingActionId) return;
    setProcessingActionId(proposal.proposal_id);
    try {
      await agentApi.rejectActionProposal(proposal.proposal_id);
      toast.success('操作草案已拒绝');
    } catch {
      // The HTTP interceptor already shows the server error.
    } finally {
      await refreshApprovals().catch(() => undefined);
      setProcessingActionId(null);
    }
  };

  const approvalPanel = (
    <ActionApprovalPanel
      executions={recentExecutions}
      loading={approvalsLoading}
      onApprove={(proposal) => void approveAction(proposal)}
      onDismissFailure={dismissFailure}
      onRefresh={() => void refreshApprovals().catch(() => undefined)}
      onReject={(proposal) => void rejectAction(proposal)}
      processingId={processingActionId}
      proposals={pendingActions}
    />
  );
  const failedExecutionCount = recentExecutions.filter(
    ({ verification, proposal }) => String(verification.status ?? proposal.status) === 'failed'
  ).length;

  return (
    <div className="grid h-[calc(100vh-3.75rem)] grid-cols-[260px_minmax(0,1fr)] bg-muted/30 xl:grid-cols-[260px_minmax(420px,1fr)_360px]">
      <aside className="block border-r bg-background p-3">
        <Button className="mb-3 w-full" onClick={() => void newConversation()}>
          <Plus className="size-4" /> 新对话
        </Button>
        <div className="mb-2 flex items-center gap-2 px-2 text-xs font-medium text-muted-foreground">
          <History className="size-3.5" /> 持久化会话
        </div>
        <ScrollArea className="h-[calc(100vh-9rem)]">
          <div className="space-y-1 pr-2">
            {conversations.map((item) => (
              <div
                className={`group grid w-full grid-cols-[minmax(0,1fr)_40px] items-stretch overflow-hidden rounded-lg text-sm transition-colors ${
                  conversationId === item.conversation_id
                    ? 'bg-accent text-accent-foreground'
                    : 'hover:bg-muted'
                }`}
                key={item.conversation_id}
              >
                <button
                  className="min-w-0 px-3 py-2 text-left"
                  onClick={() => void openConversation(item.conversation_id)}
                  type="button"
                >
                  <span className="block truncate font-medium">{item.title}</span>
                  <span className="block text-xs text-muted-foreground">
                    Checkpoint v{item.checkpoint_version}
                    {activeRuns[item.conversation_id] && (
                      <> · {({
                        pending: '运行中', collecting_evidence: '运行中', reasoning: '运行中',
                        succeeded: '完成', failed: '失败', cancelled: '已取消',
                      } as const)[activeRuns[item.conversation_id].status]}</>
                    )}
                  </span>
                </button>
                <button
                  aria-label={`删除会话 ${item.title}`}
                  className="m-1 flex items-center justify-center rounded-md text-destructive opacity-0 transition-opacity hover:bg-destructive/10 focus:opacity-100 group-hover:opacity-100"
                  data-testid={`delete-conversation-${item.conversation_id}`}
                  onClick={() => requestDelete(item)}
                  title="删除会话"
                  type="button"
                >
                  <Trash2 className="size-4" />
                </button>
              </div>
            ))}
          </div>
        </ScrollArea>
      </aside>

      <section className="flex min-h-0 flex-col">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b bg-background px-5 py-3">
          <div>
            <h1 className="flex items-center gap-2 font-semibold">
              <Bot className="size-5 text-primary" /> 实验 Agent
            </h1>
            <p className="text-xs text-muted-foreground">
              对话可恢复 · 工具调用需审批 · 轨迹实时可见
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Dialog
              open={approvalDialogOpen}
              onOpenChange={(open) => {
                setApprovalDialogOpen(open);
                if (open) void refreshApprovals().catch(() => undefined);
              }}
            >
              <DialogTrigger asChild>
                <Button data-testid="open-action-approvals" size="sm" variant="outline">
                  <ShieldCheck className="size-4" />
                  待审批
                  {pendingActions.length > 0 && <Badge>{pendingActions.length}</Badge>}
                  {failedExecutionCount > 0 && (
                    <Badge variant="destructive">失败 {failedExecutionCount}</Badge>
                  )}
                </Button>
              </DialogTrigger>
              <DialogContent className="sm:max-w-2xl">
                <DialogHeader>
                  <DialogTitle>操作审批</DialogTitle>
                  <DialogDescription>这是 Agent 副作用操作的人工确认边界。</DialogDescription>
                </DialogHeader>
                <ScrollArea className="h-[65vh] pr-3">{approvalPanel}</ScrollArea>
              </DialogContent>
            </Dialog>
            {running && (
              <Badge variant="secondary">
                <CircleDot className="animate-pulse" /> Running
              </Badge>
            )}
            {monitorSummary && (
              <Badge variant="secondary" title="当前账号任务监控">
                运行中 {monitorSummary.tasks.running ?? 0}
              </Badge>
            )}
          </div>
        </div>
        <ScrollArea className="flex-1 p-5">
          {messages.length === 0 ? (
            <div className="mx-auto mt-24 max-w-xl text-center">
              <Bot className="mx-auto mb-4 size-10 text-primary" />
              <h2 className="text-xl font-semibold">从实验数据出发，而不是猜测</h2>
              <p className="mt-2 text-sm text-muted-foreground">
                查询任务、诊断日志、比较指标，或生成下一轮单变量实验方案。
              </p>
            </div>
          ) : (
            <div className="mx-auto max-w-3xl space-y-4 pb-4">
              {messages.map((item, index) => (
                <div
                  className={`flex ${item.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  key={`${item.role}-${index}`}
                >
                  <div
                    className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-6 ${
                      item.role === 'user'
                        ? 'bg-primary text-primary-foreground'
                        : 'border bg-background shadow-sm'
                    }`}
                  >
                    {item.content ? (
                      item.role === 'assistant' ? (
                        <MarkdownMessage content={item.content} />
                      ) : (
                        <span className="whitespace-pre-wrap">{item.content}</span>
                      )
                    ) : (
                      <span className="animate-pulse text-muted-foreground">思考中…</span>
                    )}
                  </div>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}
        </ScrollArea>
        <div className="border-t bg-background p-4">
          <div className="mx-auto flex max-w-3xl gap-2">
            <Textarea
              className="min-h-12 resize-none"
              onChange={(event) => setPrompt(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault();
                  void send();
                }
              }}
              placeholder="输入实验问题，Enter 发送，Shift+Enter 换行"
              value={prompt}
            />
            {running ? (
              <Button size="icon" variant="outline" onClick={() => void stop()}>
                <Square className="size-4" />
              </Button>
            ) : (
              <Button size="icon" onClick={() => void send()} disabled={!prompt.trim()}>
                <Send className="size-4" />
              </Button>
            )}
          </div>
        </div>
      </section>

      <aside className="hidden min-h-0 border-l bg-background xl:block">
        <Tabs defaultValue="trace" className="flex h-full flex-col">
          <TabsList className="m-3 grid grid-cols-3">
            <TabsTrigger value="trace">执行轨迹</TabsTrigger>
            <TabsTrigger value="eval">Golden Eval</TabsTrigger>
            <TabsTrigger value="approval">
              待审批{pendingActions.length > 0 ? ` ${pendingActions.length}` : ''}
            </TabsTrigger>
          </TabsList>
          <TabsContent value="trace" className="min-h-0 flex-1 px-3 pb-3">
            <ScrollArea className="h-full">
              <div className="space-y-2 pr-2">
                {events.length === 0 && (
                  <p className="py-12 text-center text-sm text-muted-foreground">
                    发送消息后实时展示轨迹
                  </p>
                )}
                {events
                  .filter((event) => event.type !== 'model.delta')
                  .map((event) => (
                    <Card
                      className="gap-2 py-3 shadow-none"
                      key={`${event.run_id}-${event.sequence}`}
                    >
                      <CardHeader className="px-3">
                        <CardTitle className="flex items-center gap-2 text-sm">
                          {event.type.startsWith('tool.') ? (
                            <Wrench className="size-4 text-primary" />
                          ) : (
                            <CircleDot className="size-4 text-primary" />
                          )}
                          {eventLabels[event.type]}
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="px-3 text-xs text-muted-foreground">
                        {event.payload.name ? String(event.payload.name) : `#${event.sequence}`}
                      </CardContent>
                    </Card>
                  ))}
              </div>
            </ScrollArea>
          </TabsContent>
          <TabsContent value="eval" className="min-h-0 flex-1 px-3 pb-3">
            <div className="mb-3 flex items-center justify-between">
              <div>
                <p className="text-sm font-medium">{goldenCases.length} 条基准样例</p>
                <p className="text-xs text-muted-foreground">答案规则与安全回归</p>
              </div>
              <Button
                size="sm"
                onClick={() => void runEvaluation()}
                disabled={evaluating || evaluationRunning || goldenCases.length === 0}
              >
                <FlaskConical className="size-4" /> {evaluating || evaluationRunning ? '后台运行中' : '运行评测'}
              </Button>
            </div>
            <ScrollArea className="h-[calc(100%-3.5rem)]">
              <div className="space-y-2 pr-2">
                {evalRuns.map((item) => (
                  <Card className="cursor-pointer gap-2 py-3 shadow-none hover:bg-muted/50" key={item.eval_run_id} onClick={() => {
                    void agentApi.getEvaluationDetail(item.eval_run_id).then((detail) => setSelectedEvaluation({ evaluation: detail.evaluation, results: detail.results }));
                  }}>
                    <CardHeader className="px-3">
                      <CardTitle className="text-sm">
                        {item.status === 'running'
                          ? `进度 ${item.metrics?.completed_cases ?? 0}/${item.total} · 通过 ${item.passed}`
                          : item.status === 'failed'
                            ? `已完成 ${item.metrics?.completed_cases ?? 0}/${item.total} · 通过 ${item.passed}`
                            : `通过 ${item.passed}/${item.total}`}
                      </CardTitle>
                      <p className="text-xs text-muted-foreground">
                        {item.status === 'running' ? '后台运行中 · ' : item.status === 'failed' ? '已中断 · ' : ''}
                        工具 {Math.round((item.metrics?.tool_accuracy ?? 0) * 100)}%
                        {' · '}P95 {item.metrics?.p95_latency_ms ?? 0} ms
                        {' · '}失败 {Math.round((item.metrics?.failure_rate ?? 0) * 100)}%
                      </p>
                    </CardHeader>
                    <CardContent className="flex items-center justify-between px-3 text-xs text-muted-foreground">
                      <span>{new Date(item.created_at).toLocaleString()}</span>
                      <Badge variant={item.score === 1 ? 'default' : 'secondary'}>
                        {item.status === 'running'
                          ? `${Math.round((item.metrics?.progress ?? 0) * 100)}%`
                          : `${Math.round(item.score * 100)}%`}
                      </Badge>
                    </CardContent>
                  </Card>
                ))}
                {selectedEvaluation && (
                  <Card className="gap-2 border-primary/30 py-3 shadow-none">
                    <CardHeader className="px-3"><CardTitle className="text-sm">评测用例详情</CardTitle></CardHeader>
                    <CardContent className="space-y-1 px-3 text-xs text-muted-foreground">
                      <p>运行 ID：{selectedEvaluation.evaluation.eval_run_id}</p>
                      <div className="grid grid-cols-2 gap-1 rounded bg-muted p-2">
                        <span>任务成功率：{Math.round((selectedEvaluation.evaluation.metrics?.task_success_rate ?? 0) * 100)}%</span>
                        <span>工具准确率：{Math.round((selectedEvaluation.evaluation.metrics?.tool_accuracy ?? 0) * 100)}%</span>
                        <span>平均延迟：{selectedEvaluation.evaluation.metrics?.average_latency_ms ?? 0} ms</span>
                        <span>P95 延迟：{selectedEvaluation.evaluation.metrics?.p95_latency_ms ?? 0} ms</span>
                        <span>失败率：{Math.round((selectedEvaluation.evaluation.metrics?.failure_rate ?? 0) * 100)}%</span>
                      </div>
                      <p>共 {selectedEvaluation.results.length} 条用例结果</p>
                      {selectedEvaluation.evaluation.metrics?.error && (
                        <p className="rounded bg-destructive/10 p-2 text-destructive">
                          {selectedEvaluation.evaluation.metrics.error}
                        </p>
                      )}
                      <pre className="max-h-48 overflow-auto whitespace-pre-wrap rounded bg-muted p-2 text-[11px]">{JSON.stringify(selectedEvaluation.results, null, 2)}</pre>
                    </CardContent>
                  </Card>
                )}
              </div>
            </ScrollArea>
          </TabsContent>
          <TabsContent value="approval" className="min-h-0 flex-1 px-3 pb-3">
            <ScrollArea className="h-full pr-2">{approvalPanel}</ScrollArea>
          </TabsContent>
        </Tabs>
      </aside>

      <Dialog
        open={Boolean(deleteTarget)}
        onOpenChange={(open) => {
          if (!open && !deleting) {
            setDeleteTarget(null);
            setDeletePassword('');
          }
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>永久删除会话</DialogTitle>
            <DialogDescription>
              将删除“{deleteTarget?.title}”及其全部
              Checkpoint、运行记录和执行轨迹。此操作无法撤销，请输入当前账号密码确认。
            </DialogDescription>
          </DialogHeader>
          <Input
            autoComplete="current-password"
            autoFocus
            onChange={(event) => setDeletePassword(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') void deleteConversation();
            }}
            placeholder="当前账号密码"
            type="password"
            value={deletePassword}
          />
          <DialogFooter>
            <Button
              disabled={deleting}
              onClick={() => {
                setDeleteTarget(null);
                setDeletePassword('');
              }}
              variant="outline"
            >
              取消
            </Button>
            <Button
              disabled={!deletePassword || deleting}
              onClick={() => void deleteConversation()}
              variant="destructive"
            >
              {deleting ? '删除中…' : '确认永久删除'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
