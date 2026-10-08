import { apiConfig } from '@/config/env';
import httpService from '@/services/http';
import type {
  AgentActionDetail,
  AgentActionProposal,
  AgentConversation,
  AgentMessage,
  AgentTraceEvent,
  AgentRunSummary,
  EvalRun,
  GoldenCase,
} from '@/types/agent';

export const agentApi = {
  listConversations: () =>
    httpService.get<{ items: AgentConversation[] }>('/api/agent/conversations'),

  createConversation: (title = '新对话') =>
    httpService.post<{ conversation: AgentConversation }>('/api/agent/conversations', { title }),

  getConversation: (conversationId: string) =>
    httpService.get<{ conversation: AgentConversation; messages: AgentMessage[] }>(
      `/api/agent/conversations/${conversationId}`
    ),
  getRunEvents: (runId: string) =>
    httpService.get<{ run: { run_id: string; status: string }; events: AgentTraceEvent[] }>(
      `/api/agent/runs/${runId}/events`
    ),
  listConversationRuns: (conversationId: string) =>
    httpService.get<{ items: AgentRunSummary[] }>(
      `/api/agent/conversations/${conversationId}/runs`
    ),

  deleteConversation: (conversationId: string, password: string) =>
    httpService.delete<{
      message: string;
      conversation_id: string;
      deleted: { checkpoints: number; runs: number; trace_events: number };
    }>(`/api/agent/conversations/${conversationId}`, { data: { password } }),

  cancelRun: (runId: string) =>
    httpService.delete<AgentRunSummary>(`/api/agent/runs/${runId}`),

  listGoldenCases: () => httpService.get<{ items: GoldenCase[] }>('/api/agent/golden-cases'),

  listEvaluations: () => httpService.get<{ items: EvalRun[] }>('/api/agent/evaluations'),

  runEvaluation: (caseIds?: string[]) =>
    httpService.post<{ evaluation: EvalRun }>(
      '/api/agent/evaluations',
      caseIds ? { case_ids: caseIds } : {},
      { timeout: 15_000 }
    ),
  getEvaluationDetail: (evalRunId: string) =>
    httpService.get<{ evaluation: EvalRun; results: unknown[] }>(
      `/api/agent/evaluations/${evalRunId}`
    ),
  getMonitorSummary: () =>
    httpService.get<{ runs: Record<string, number>; tasks: Record<string, number> }>(
      '/api/agent/monitor/summary'
    ),
  compareExperiments: (jobIds: string[]) =>
    httpService.post<{ items: unknown[]; metrics: string[] }>('/api/agent/insights/compare', {
      job_ids: jobIds,
    }),
  analyzeExperiment: (jobId: string) =>
    httpService.get<{ summary: string; suggestions: string[]; metrics: Record<string, number> }>(
      `/api/agent/insights/analyze/${jobId}`
    ),
  checkDatasetQuality: (datasetId: number) =>
    httpService.get<{ healthy: boolean; issues: string[]; splits: Record<string, unknown> }>(
      `/api/agent/datasets/${datasetId}/quality`
    ),

  listActionProposals: (status?: AgentActionProposal['status'], limit = 20, conversationId?: string | null) =>
    httpService.get<{ items: AgentActionProposal[] }>('/api/agent/actions', {
      params: { status, limit, conversation_id: conversationId || undefined },
    }),

  getActionProposal: (proposalId: string, silent = false) =>
    httpService.get<AgentActionDetail>(`/api/agent/actions/${proposalId}`, { silent }),

  approveActionProposal: (proposalId: string) =>
    httpService.post<AgentActionDetail>(`/api/agent/actions/${proposalId}/approve`),

  rejectActionProposal: (proposalId: string) =>
    httpService.post<{ proposal: AgentActionProposal }>(`/api/agent/actions/${proposalId}/reject`),

  async streamChat(
    message: string,
    conversationId: string | null,
    onEvent: (event: AgentTraceEvent) => void,
    signal?: AbortSignal
  ) {
    const response = await fetch(apiConfig.getUrl('/api/agent/chat/stream'), {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
      body: JSON.stringify({ message, conversation_id: conversationId }),
      signal,
    });
    if (!response.ok || !response.body) {
      const body = (await response.json().catch(() => ({}))) as { error?: string };
      throw new Error(body.error || `Agent 请求失败 (${response.status})`);
    }

    await readEventStream(response, onEvent);
  },

  async resumeRun(
    runId: string,
    after: number,
    onEvent: (event: AgentTraceEvent) => void,
    signal?: AbortSignal
  ) {
    const response = await fetch(
      apiConfig.getUrl(`/api/agent/runs/${encodeURIComponent(runId)}/stream?after=${Math.max(0, after)}`),
      { credentials: 'include', headers: { Accept: 'text/event-stream' }, signal }
    );
    if (!response.ok || !response.body) throw new Error(`恢复 Agent 事件流失败 (${response.status})`);
    await readEventStream(response, onEvent);
  },
};

async function readEventStream(
  response: Response,
  onEvent: (event: AgentTraceEvent) => void
) {
    if (!response.body) return;
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      const blocks = buffer.split('\n\n');
      buffer = blocks.pop() ?? '';
      for (const block of blocks) {
        const data = block
          .split('\n')
          .find((line) => line.startsWith('data: '))
          ?.slice(6);
        if (data) onEvent(JSON.parse(data) as AgentTraceEvent);
      }
      if (done) break;
    }
}
